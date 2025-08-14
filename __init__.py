
import io, os, wave, base64, requests, numpy as np

try:
    import torch
except Exception:
    torch = None

CATEGORY = "I/O → URL"

def _upload_bytes(filename: str, data: bytes) -> str:
    r = requests.post("https://0x0.st", files={"file": (filename, data)}, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _audio_to_wav_bytes(samples, sample_rate, filename="audio.wav"):
    arr = _to_numpy(samples)
    if arr.ndim == 1:  # [T] -> [1,T]
        arr = arr[np.newaxis, :]
    channels = arr.shape[0]
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr.T * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes(order="C"))
    return (filename if str(filename).strip() else "audio.wav"), bio.getvalue()

def _maybe_get_path_from_dict(d):
    for k in ["path","filepath","file","tmp_path","audio_path","local_path","audio_file","filename"]:
        v = d.get(k)
        if isinstance(v, str) and os.path.isfile(v):
            return v
    return None

def _maybe_get_bytes_from_dict(d):
    for k in ["bytes","data","content"]:
        v = d.get(k)
        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v)
        if isinstance(v, str) and v.startswith("data:audio"):
            b64 = v.split(",", 1)[-1]
            try:
                return base64.b64decode(b64)
            except Exception:
                pass
    return None

def _maybe_get_samples_sr_from_dict(d):
    sample_keys = ["samples","waveform","audio","array","tensor"]
    sr_keys = ["sample_rate","sr","rate","sampleRate"]
    samples = None
    sr = None
    for k in sample_keys:
        if k in d:
            samples = d[k]
            break
    for k in sr_keys:
        if k in d:
            sr = d[k]
            break
    if samples is not None and sr is not None:
        return samples, sr
    return None, None

class AudioToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": ""}),
                "debug": ("BOOLEAN", {"default": False})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, filename="", debug=False):
        # 0) If it's already a remote URL, pass-through
        if isinstance(audio, str) and (audio.startswith("http://") or audio.startswith("https://") or audio.startswith("s3://")):
            return (audio,)

        # dict-like inputs
        if isinstance(audio, dict):
            if debug:
                print("[AudioToURL_0x0] dict keys:", list(audio.keys()))
            # (1) direct samples+sr
            samples, sr = _maybe_get_samples_sr_from_dict(audio)
            if samples is not None:
                fname, data = _audio_to_wav_bytes(samples, sr, filename or "audio.wav")
                return (_upload_bytes(fname, data),)
            # (2) embedded file path
            p = _maybe_get_path_from_dict(audio)
            if p:
                with open(p, "rb") as f:
                    payload = f.read()
                base = os.path.basename(p)
                return (_upload_bytes(base, payload),)
            # (3) embedded raw bytes / data URL
            b = _maybe_get_bytes_from_dict(audio)
            if b:
                fname = filename if filename.strip() else "audio.bin"
                return (_upload_bytes(fname, b),)

        # tuple/list -> (samples, sr)
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            samples, sr = audio
            fname, data = _audio_to_wav_bytes(samples, sr, filename or "audio.wav")
            return (_upload_bytes(fname, data),)

        # local path
        if isinstance(audio, str) and os.path.isfile(audio):
            with open(audio, "rb") as f:
                payload = f.read()
            base = os.path.basename(audio)
            return (_upload_bytes(base, payload),)

        # raw bytes
        if isinstance(audio, (bytes, bytearray, memoryview)):
            fname = filename if filename.strip() else "audio.bin"
            return (_upload_bytes(fname, bytes(audio)),)

        # object with attributes (e.g., audio.samples, audio.sample_rate)
        if hasattr(audio, "samples") and hasattr(audio, "sample_rate"):
            fname, data = _audio_to_wav_bytes(getattr(audio, "samples"), getattr(audio, "sample_rate"), filename or "audio.wav")
            return (_upload_bytes(fname, data),)

        raise RuntimeError("AudioToURL_0x0: unsupported AUDIO object. Enable 'debug' to print keys/shape.")
