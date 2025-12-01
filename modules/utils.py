import os
import gc
import sys
import torch
import codecs
import librosa
import requests
import numpy as np
import soundfile as sf
import torch.nn.functional as F

sys.path.append(os.getcwd())

from modules import opencl
from pathlib import Path

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'


mdxnet_models_dir =  os.path.join(os.getcwd(), 'models')


def dl_model(link, model_name, dir_name):
    model_path = dir_name / model_name
    if model_path.exists():
        # print(f"{model_name} already exists, skipping download.")
        return

    print(f"Downloading {model_name}...")
    with requests.get(f'{link}{model_name}', stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_mdx():
    mdx_model_names = ['UVR-MDX-NET-Inst_HQ_4.onnx', 'UVR-MDX-NET-Voc_FT.onnx', 
                      'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)


def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(
        torch.from_numpy(
            librosa.feature.rms(
                y=target_audio, 
                frame_length=target_rate // 2 * 2, 
                hop_length=target_rate // 2
            )
        ).float().unsqueeze(0), 
        size=target_audio.shape[0], 
        mode="linear"
    ).squeeze()
    
    source_rms = F.interpolate(
        torch.from_numpy(
            librosa.feature.rms(
                y=source_audio, 
                frame_length=source_rate // 2 * 2, 
                hop_length=source_rate // 2
            )
        ).float().unsqueeze(0), 
        size=target_audio.shape[0], 
        mode="linear"
    ).squeeze()
    
    return target_audio * (
        torch.pow(
            source_rms, 
            1 - rate
        ) * torch.pow(
            torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), 
            rate - 1
        )
    ).numpy()


def clear_gpu_cache():
    gc.collect()
    
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): 
        torch.mps.empty_cache()
    elif opencl.is_available(): 
        opencl.pytorch_ocl.empty_cache()


def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    
    if output_path is None:
        output_path = os.path.basename(url)
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(url))
    
    response = requests.get(url, stream=True, timeout=300)
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                f.write(chunk)
        return output_path
    else: 
        raise ValueError(response.status_code)


def check_predictors(method):
    def download(predictors):
        if not os.path.exists(os.path.join("models", predictors)):
            HF_download_file(
                codecs.decode(
                    "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", 
                    "rot13"
                ) + predictors, 
                os.path.join("models", predictors)
            )
    
    model_dict = {
        **dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), 
        **dict.fromkeys(["fcpe"], "fcpe.pt"), 
        **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"), 
        **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), 
        **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), 
        **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), 
        **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), 
        **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), 
    }
    
    if method in model_dict: 
        download(model_dict[method])


def check_embedders(hubert):
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", 
                  "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "spin"]:
        hubert += ".pt"
        model_path = os.path.join("models", hubert)
        
        if not os.path.exists(model_path):
            HF_download_file(
                "".join([
                    codecs.decode(
                        "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", 
                        "rot13"
                    ), 
                    "fairseq/", 
                    hubert
                ]), 
                model_path
            )


def load_audio(file, sample_rate=16000):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): 
            raise FileNotFoundError(f"[ERROR] Not found audio: {file}")
        # Note: You might want to add audio loading logic here
    except Exception as e:
        raise e        try:
            audio, sr = sf.read(file, dtype=np.float32)
        except:
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Error reading audio file: {e}")
    
    return audio.flatten()

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0
