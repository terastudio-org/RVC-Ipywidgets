import os
from audio_separator.separator import Separator
import sys
import torch
import librosa
import logging
import warnings
import numpy as np
import soundfile as sf
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from modules import fairseq
from modules.config import Config
from modules.cut import cut, restore
from modules.pipeline import Pipeline
from modules.utils import clear_gpu_cache
from modules.synthesizers import Synthesizer
from modules.utils import check_predictors, check_embedders, load_audio

for l in ["torch", "faiss", "omegaconf", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "transformers", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)





def separate_audio_tracks(
    input_path: str,
    output_dir: str = "output",
    model_vocals_instrumental: str = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    model_deecho_dereverb: str = 'UVR-DeEcho-DeReverb.pth',
    model_backing_vocals: str = 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
) -> dict:
    """
    Separate audio tracks into multiple components.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save output files
        model_vocals_instrumental: Model filename for vocals/instrumental separation
        model_deecho_dereverb: Model filename for de-echo/de-reverb
        model_backing_vocals: Model filename for backing vocals separation
    
    Returns:
        Dictionary with paths to all separated tracks
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize paths for output files
    paths = {
        'vocals': os.path.join(output_dir, 'Vocals.wav'),
        'instrumental': os.path.join(output_dir, 'Instrumental.wav'),
        'vocals_reverb': os.path.join(output_dir, 'Vocals (Reverb).wav'),
        'vocals_no_reverb': os.path.join(output_dir, 'Vocals (No Reverb).wav'),
        'lead_vocals': os.path.join(output_dir, 'Lead Vocals.wav'),
        'backing_vocals': os.path.join(output_dir, 'Backing Vocals.wav')
    }
    
    try:
        # Initialize separator
        separator = Separator(output_dir=output_dir)
        
        # Step 1: Splitting into Vocal and Instrumental
        print("Step 1: Separating vocals and instrumental...")
        separator.load_model(model_filename=model_vocals_instrumental)
        voc_inst = separator.separate(input_path)
        
        # Rename output files
        os.rename(os.path.join(output_dir, voc_inst[0]), paths['instrumental'])
        os.rename(os.path.join(output_dir, voc_inst[1]), paths['vocals'])
        print(f"✓ Instrumental saved to: {paths['instrumental']}")
        print(f"✓ Vocals saved to: {paths['vocals']}")
        
        # Step 2: Applying DeEcho-DeReverb to Vocals
        print("\nStep 2: Applying DeEcho-DeReverb to vocals...")
        separator.load_model(model_filename=model_deecho_dereverb)
        voc_no_reverb = separator.separate(paths['vocals'])
        
        # Rename output files
        os.rename(os.path.join(output_dir, voc_no_reverb[0]), paths['vocals_no_reverb'])
        os.rename(os.path.join(output_dir, voc_no_reverb[1]), paths['vocals_reverb'])
        print(f"✓ Vocals (No Reverb) saved to: {paths['vocals_no_reverb']}")
        print(f"✓ Vocals (Reverb) saved to: {paths['vocals_reverb']}")
        
        # Step 3: Separating Back Vocals from Main Vocals
        print("\nStep 3: Separating lead and backing vocals...")
        separator.load_model(model_filename=model_backing_vocals)
        backing_voc = separator.separate(paths['vocals_no_reverb'])
        
        # Rename output files
        os.rename(os.path.join(output_dir, backing_voc[0]), paths['backing_vocals'])
        os.rename(os.path.join(output_dir, backing_voc[1]), paths['lead_vocals'])
        print(f"✓ Backing Vocals saved to: {paths['backing_vocals']}")
        print(f"✓ Lead Vocals saved to: {paths['lead_vocals']}")
        
        print("\n✅ Audio separation completed successfully!")
        return paths
        
    except Exception as e:
        print(f"❌ Error during audio separation: {str(e)}")
        raise




def run_inference_script(
    is_half=False, 
    cpu_mode=False,
    pitch=0, 
    filter_radius=3, 
    index_rate=0.5, 
    volume_envelope=1, 
    protect=0.5, 
    hop_length=64, 
    f0_method="rmvpe", 
    input_path=None, 
    output_path="./output.wav", 
    pth_path=None, 
    index_path=None, 
    export_format="wav", 
    embedder_model="contentvec_base", 
    resample_sr=0,  
    f0_autotune=False, 
    f0_autotune_strength=1, 
    split_audio=False,
    clean_audio=False, 
    clean_strength=0.7
):
    check_predictors(f0_method); check_embedders(embedder_model)
    
    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith(".pth"):
        print("[WARNING] Please enter a valid model.")
        return

    config = Config(is_half=is_half, cpu_mode=cpu_mode)
    cvt = VoiceConverter(config, pth_path, 0)

    if os.path.isdir(input_path):
        print("[INFO] Use batch conversion...")
        audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

        if not audio_files: 
            print("[WARNING] No audio files found.")
            return

        print(f"[INFO] Found {len(audio_files)} audio files for conversion.")

        for audio in audio_files:
            audio_path = os.path.join(input_path, audio)
            output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

            print(f"[INFO] Conversion '{audio_path}'...")
            if os.path.exists(output_audio): os.remove(output_audio)

            cvt.convert_audio(
                audio_input_path=audio_path, 
                audio_output_path=output_audio, 
                index_path=index_path, 
                embedder_model=embedder_model, 
                pitch=pitch, 
                f0_method=f0_method, 
                index_rate=index_rate, 
                volume_envelope=volume_envelope, 
                protect=protect, 
                hop_length=hop_length, 
                filter_radius=filter_radius, 
                export_format=export_format, 
                resample_sr=resample_sr, 
                f0_autotune=f0_autotune, 
                f0_autotune_strength=f0_autotune_strength,
                split_audio=split_audio,
                clean_audio=clean_audio,
                clean_strength=clean_strength
            )

        print("[INFO] Conversion complete.")
    else:
        if not os.path.exists(input_path):
            print("[WARNING] No audio files found.")
            return

        print(f"[INFO] Conversion '{input_path}'...")
        if os.path.exists(output_path): os.remove(output_path)

        cvt.convert_audio(
            audio_input_path=input_path, 
            audio_output_path=output_path, 
            index_path=index_path, 
            embedder_model=embedder_model, 
            pitch=pitch, 
            f0_method=f0_method, 
            index_rate=index_rate, 
            volume_envelope=volume_envelope, 
            protect=protect, 
            hop_length=hop_length, 
            filter_radius=filter_radius,  
            export_format=export_format, 
            resample_sr=resample_sr, 
            f0_autotune=f0_autotune, 
            f0_autotune_strength=f0_autotune_strength,
            split_audio=split_audio,
            clean_audio=clean_audio,
            clean_strength=clean_strength
        )

        print("[INFO] Conversion complete.")

class VoiceConverter:
    def __init__(self, config, model_path, sid = 0):
        self.config = config
        self.device = config.device
        self.hubert_model = None
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
        self.vocoder = "Default"
        self.sample_rate = 16000
        self.sid = sid
        self.get_vc(model_path, sid)

    def convert_audio(
        self, 
        audio_input_path, 
        audio_output_path, 
        index_path, 
        embedder_model, 
        pitch, 
        f0_method, 
        index_rate, 
        volume_envelope, 
        protect, 
        hop_length, 
        filter_radius, 
        export_format, 
        resample_sr = 0, 
        f0_autotune=False, 
        f0_autotune_strength=1,
        split_audio=False,
        clean_audio=False,
        clean_strength=0.5
    ):
        try:
            audio = load_audio(audio_input_path, self.sample_rate)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1: audio /= audio_max

            if not self.hubert_model:
                embedder_model_path = os.path.join("models", embedder_model + ".pt")
                if not os.path.exists(embedder_model_path): raise FileNotFoundError(f"[ERROR] Not found embeddeder: {embedder_model}")

                models = fairseq.load_model(embedder_model_path).to(self.device).eval()
                self.hubert_model = models.half() if self.config.is_half else models.float()

            if split_audio:
                chunks = cut(
                    audio, 
                    self.sample_rate, 
                    db_thresh=-60, 
                    min_interval=500
                )  
                print(f"Split Total: {len(chunks)}")
            else: chunks = [(audio, 0, 0)]

            converted_chunks = [
                (
                    start, 
                    end, 
                    self.vc.pipeline(
                        model=self.hubert_model, 
                        net_g=self.net_g, 
                        sid=self.sid, 
                        audio=waveform, 
                        f0_up_key=pitch, 
                        f0_method=f0_method, 
                        file_index=(
                            index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
                        ), 
                        index_rate=index_rate, 
                        pitch_guidance=self.use_f0, 
                        filter_radius=filter_radius, 
                        volume_envelope=volume_envelope, 
                        version=self.version, 
                        protect=protect, 
                        hop_length=hop_length, 
                        energy_use=self.energy,
                        f0_autotune=f0_autotune, 
                        f0_autotune_strength=f0_autotune_strength
                    )
                ) for waveform, start, end in chunks
            ]

            audio_output = restore(
                converted_chunks, 
                total_len=len(audio), 
                dtype=converted_chunks[0][2].dtype
            ) if split_audio else converted_chunks[0][2]

            if self.tgt_sr != resample_sr and resample_sr > 0: 
                audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")
                self.tgt_sr = resample_sr

            if clean_audio:
                from modules.noisereduce import reduce_noise
                audio_output = reduce_noise(
                    y=audio_output, 
                    sr=self.tgt_sr, 
                    prop_decrease=clean_strength, 
                    device=self.device
                ) 

            sf.write(audio_output_path, audio_output, self.tgt_sr, format=export_format)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"[ERROR] An error has occurred: {e}")

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup()
            clear_gpu_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.loaded_model = weight_root
            self.load_model()
            if self.cpt is not None: self.setup()

    def cleanup(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            clear_gpu_cache()

        del self.net_g, self.cpt
        clear_gpu_cache()
        self.cpt = None

    def load_model(self):
        if os.path.isfile(self.loaded_model): self.cpt = torch.load(self.loaded_model, map_location="cpu")  
        else: self.cpt = None

    def setup(self):
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]

            self.use_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            self.vocoder = self.cpt.get("vocoder", "Default")
            self.energy = self.cpt.get("energy", False)

            if self.vocoder != "Default": self.config.is_half = False
            self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=768 if self.version == "v2" else 256, vocoder=self.vocoder, energy=self.energy)
            del self.net_g.enc_q

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.device)
            self.net_g = (self.net_g.half() if self.config.is_half else self.net_g.float())
            self.n_spk = self.cpt["config"][-3]

            self.vc = Pipeline(self.tgt_sr, self.config)
