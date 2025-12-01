import gc
import hashlib
import os
import queue
import threading
import warnings
import time
from datetime import datetime
from multiprocessing import cpu_count
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}


class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(self, model_path: str, params: MDXModel, processor=0):
        self.start_time = time.time()
        model_name = os.path.basename(model_path)
        
        # Set the device and the provider (CPU or CUDA)
        self.device = (
            torch.device(f"cuda:{processor}")
            if processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"]
            if processor >= 0
            else ["CPUExecutionProvider"]
        )

        print(f"\n{'='*60}")
        print(f"MDX Model Initialization")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Execution Provider: {self.provider}")
        print(f"Stem Name: {params.stem_name}")
        print(f"Model Parameters: dim_f={params.dim_f}, dim_t={params.dim_t}, "
              f"n_fft={params.n_fft}, hop={params.hop}")
        print(f"{'='*60}")

        self.model = params

        # Load the ONNX model using ONNX Runtime
        print(f"Loading ONNX model...")
        load_start = time.time()
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        load_time = time.time() - load_start
        
        # Preload the model for faster performance
        print(f"Warming up model...")
        warmup_start = time.time()
        self.ort.run(None, {'input': torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        warmup_time = time.time() - warmup_start
        
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]
        
        init_time = time.time() - self.start_time
        print(f"Model loaded in {load_time:.2f}s (warmup: {warmup_time:.2f}s, total: {init_time:.2f}s)")
        print(f"{'='*60}\n")

        self.prog = None
        self.total_chunks = 0
        self.processed_chunks = 0

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        """
        Segment or join segmented wave array

        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array. If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)

        Returns:
            numpy array: Segmented or joined wave array
        """

        if combine:
            processed_wave = None  # Initializing as None instead of [] for later numpy array concatenation
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size

        Args:
            wave: (np.array) Wave array to be padded

        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Padded wave
        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int, thread_id: int):
        """
        Process each wave segment in a multi-threaded environment

        Args:
            mix_waves: (torch.Tensor) Wave segments to be processed
            trim: (int) Number of samples trimmed during padding
            pad: (int) Number of samples padded during padding
            q: (queue.Queue) Queue to hold the processed wave segments
            _id: (int) Identifier of the processed wave segment
            thread_id: (int) Thread identifier for logging
        """
        mix_waves = mix_waves.split(1)
        thread_chunks = len(mix_waves)
        
        with torch.no_grad():
            pw = []
            for i, mix_wave in enumerate(mix_waves):
                self.prog.update()
                self.processed_chunks += 1
                
                # Process STFT
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
                
                # Log progress every 10% or at least every 10 chunks
                if thread_chunks > 10 and (i + 1) % max(1, thread_chunks // 10) == 0:
                    print(f"  Thread {thread_id}: Processed {i+1}/{thread_chunks} chunks "
                          f"({(i+1)/thread_chunks*100:.0f}%)")
                    
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        """
        Process the wave array in a multi-threaded environment

        Args:
            wave: (np.array) Wave array to be processed
            mt_threads: (int) Number of threads to be used for processing

        Returns:
            numpy array: Processed wave array
        """
        print(f"\nStarting audio processing with {mt_threads} thread(s)...")
        print(f"Input shape: {wave.shape}, Duration: {wave.shape[1]/44100:.2f}s")
        
        processing_start = time.time()
        
        # Calculate total chunks for progress tracking
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)
        
        self.total_chunks = 0
        for batch in waves:
            mix_waves, pad, trim = self.pad_wave(batch)
            self.total_chunks += len(mix_waves)
        
        self.processed_chunks = 0
        self.prog = tqdm(total=self.total_chunks, desc="Processing chunks", 
                        unit="chunk", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        # Create a queue to hold the processed wave segments
        q = queue.Queue()
        threads = []
        
        print(f"Total chunks to process: {self.total_chunks}")
        print(f"Launching {len(waves)} processing threads...")
        
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            thread = threading.Thread(target=self._process_wave, 
                                    args=(mix_waves, trim, pad, q, c, c+1),
                                    name=f"Processing-Thread-{c+1}")
            thread.start()
            threads.append(thread)
            
            # Log thread creation
            print(f"  Started thread {c+1} with {len(mix_waves)} chunks")
        
        # Monitor thread completion
        for i, thread in enumerate(threads):
            thread.join()
            print(f"  Thread {i+1} completed")
        
        self.prog.close()
        
        # Collect results
        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        
        processed_batches = [list(wave.values())[0] for wave in
                           sorted(processed_batches, key=lambda d: list(d.keys())[0])]
        
        if len(processed_batches) != len(waves):
            print(f"⚠️  Warning: Expected {len(waves)} batches, got {len(processed_batches)}")
            raise Exception('Incomplete processed batches, please reduce batch size!')
        
        processing_time = time.time() - processing_start
        print(f"✓ Processing completed in {processing_time:.2f}s "
              f"({self.total_chunks/processing_time:.1f} chunks/sec)")
        
        return self.segment(processed_batches, True, chunk)


def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, 
            exclude_inversion=False, suffix=None, invert_suffix=None, 
            denoise=False, keep_orig=True, m_threads=2, base_device="cuda"):
    
    print(f"\n{'#'*60}")
    print(f"STARTING MDX PROCESSING")
    print(f"{'#'*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {os.path.basename(filename)}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"{'#'*60}")
    
    total_start_time = time.time()
    
    # Determine device and VRAM
    if base_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_properties = torch.cuda.get_device_properties(device)
        vram_gb = device_properties.total_memory / 1024**3
        vram_free = torch.cuda.memory_reserved(device) / 1024**3
        vram_used = vram_gb - vram_free
        
        # Adjust threads based on VRAM
        if vram_gb < 4:
            m_threads = 1
            print(f"⚠️  Low VRAM detected ({vram_gb:.1f}GB), using single thread")
        elif vram_gb < 8:
            m_threads = 2
        elif vram_gb > 32:
            m_threads = 8
        else:
            m_threads = 4
            
        processor_num = 0
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: Total={vram_gb:.1f}GB, Used={vram_used:.1f}GB, Free={vram_free:.1f}GB")
    else:
        device = torch.device("cpu")
        m_threads = min(cpu_count(), 8)  # Use up to 8 CPU threads
        processor_num = -1
        vram_gb = 0
        print(f"Using CPU with {m_threads} thread(s)")
        print(f"CPU cores: {cpu_count()}")
    
    # Get model hash and parameters
    model_hash = MDX.get_hash(model_path)
    print(f"Model hash: {model_hash}")
    
    if model_hash not in model_params:
        print(f"❌ Error: Model hash {model_hash} not found in parameters!")
        return None, None
    
    mp = model_params[model_hash]
    
    # Log model parameters
    print(f"\nModel Parameters:")
    print(f"  Primary stem: {mp.get('primary_stem', 'Unknown')}")
    print(f"  dim_f: {mp.get('mdx_dim_f_set', 'Unknown')}")
    print(f"  dim_t: {2 ** mp.get('mdx_dim_t_set', 0)}")
    print(f"  n_fft: {mp.get('mdx_n_fft_scale_set', 'Unknown')}")
    print(f"  Compensation: {mp.get('compensate', 1.0)}")
    
    # Load audio file
    print(f"\nLoading audio file...")
    load_start = time.time()
    try:
        wave, sr = librosa.load(filename, mono=False, sr=44100)
        if wave.ndim == 1:
            wave = np.stack([wave, wave])  # Convert mono to stereo
        duration = librosa.get_duration(y=wave, sr=sr)
        load_time = time.time() - load_start
        print(f"✓ Audio loaded: {duration:.2f}s, {sr}Hz, {wave.shape} shape")
        print(f"  Load time: {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return None, None
    
    # Adjust threads based on audio duration
    original_threads = m_threads
    if duration < 30:  # Short files use single thread
        m_threads = 1
        print(f"Short audio ({duration:.1f}s), using single thread")
    elif duration > 300:  # Long files might need more threads
        if m_threads < 4:
            m_threads = min(8, m_threads * 2)
    
    if original_threads != m_threads:
        print(f"Adjusted threads from {original_threads} to {m_threads} based on audio length")
    
    print(f"\nFinal processing configuration:")
    print(f"  Threads: {m_threads}")
    print(f"  Denoise: {denoise}")
    print(f"  Keep original: {keep_orig}")
    print(f"  Exclude main: {exclude_main}")
    print(f"  Exclude inversion: {exclude_inversion}")
    
    # Initialize model
    print(f"\nInitializing MDX model...")
    model_init_start = time.time()
    
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"]
    )
    
    mdx_sess = MDX(model_path, model, processor=processor_num)
    model_init_time = time.time() - model_init_start
    
    # Normalize audio
    print(f"\nNormalizing audio...")
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    print(f"  Peak amplitude: {peak:.4f}")
    
    # Process audio
    print(f"\nProcessing audio with MDX network...")
    process_start = time.time()
    
    try:
        if denoise:
            print("  Applying denoising (bidirectional processing)...")
            wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
            wave_processed *= 0.5
            print("  Denoising completed")
        else:
            wave_processed = mdx_sess.process_wave(wave, m_threads)
            
        # Restore original peak
        wave_processed *= peak
        
        process_time = time.time() - process_start
        print(f"✓ Audio processing completed in {process_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return None, None
    
    # Determine output stem names
    stem_name = model.stem_name if suffix is None else suffix
    
    # Save main stem
    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        print(f"\nSaving main stem: {os.path.basename(main_filepath)}")
        try:
            sf.write(main_filepath, wave_processed.T, sr)
            file_size = os.path.getsize(main_filepath) / (1024 * 1024)  # MB
            print(f"✓ Main stem saved: {file_size:.2f} MB")
        except Exception as e:
            print(f"❌ Error saving main stem: {e}")
            main_filepath = None
    
    # Save inversion stem
    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        invert_stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{invert_stem_name}.wav")
        print(f"\nSaving inversion stem: {os.path.basename(invert_filepath)}")
        try:
            sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)
            file_size = os.path.getsize(invert_filepath) / (1024 * 1024)  # MB
            print(f"✓ Inversion stem saved: {file_size:.2f} MB")
        except Exception as e:
            print(f"❌ Error saving inversion stem: {e}")
            invert_filepath = None
    
    # Clean up original if requested
    if not keep_orig:
        try:
            os.remove(filename)
            print(f"\nRemoved original file: {os.path.basename(filename)}")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove original file: {e}")
    
    # Cleanup
    print(f"\nCleaning up resources...")
    del mdx_sess, wave_processed, wave
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  GPU cache cleared")
    
    total_time = time.time() - total_start_time
    print(f"\n{'#'*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'#'*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Audio duration: {duration:.2f}s")
    print(f"Processing speed: {duration/total_time:.2f}x real-time")
    
    if main_filepath:
        print(f"Main output: {main_filepath}")
    if invert_filepath:
        print(f"Inversion output: {invert_filepath}")
    
    if not main_filepath and not invert_filepath:
        print("⚠️  No output files were created!")
    
    print(f"{'#'*60}\n")
    
    return main_filepath, invert_filepath


# Optional: Add a summary function for batch processing
def print_processing_summary(file_list, results):
    """Print a summary of batch processing results"""
    print(f"\n{'#'*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'#'*60}")
    print(f"Total files processed: {len(file_list)}")
    
    successful = sum(1 for result in results if result[0] or result[1])
    failed = len(file_list) - successful
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed files:")
        for i, (file, result) in enumerate(zip(file_list, results)):
            if not result[0] and not result[1]:
                print(f"  {os.path.basename(file)}")
    
    print(f"{'#'*60}")
