
import os, io, re, base64, wave, requests

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

# ----------------- helpers -----------------

AUDIO_EXTS = (".mp3",".wav",".m4a",".ogg",".flac",".aac",".opus",".webm")

def _upload_bytes(filename: str, data: bytes) -> str:
    r = requests.post("https://0x0.st", files={"file": (filename, data)}, timeout=60)
    r.raise_for_status()
    url = r.text.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise RuntimeError(f"0x0.st bad response: {url[:120]}")
    return url

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if np is not None and isinstance(x, np.ndarray):
        return x.astype("float32")
    raise RuntimeError("Unsupported audio sample array type")

def _samples_to_wav(samples, sr: int) -> bytes:
    arr = _to_numpy(samples)
    if arr.ndim == 1:
        arr = arr[:, None]          # [T] -> [T,1]
    if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
        arr = arr.T                 # [C,T] -> [T,C]
    arr = arr.clip(-1.0, 1.0)
    if arr.dtype.kind == "f":
        arr = (arr * 32767.0).round().astype("int16")
    elif arr.dtype != "int16":
        arr = arr.astype("int16")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(arr.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sr or 44100))
        wf.writeframes(arr.tobytes(order="C"))
    return bio.getvalue()

def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("s3://"))

def _is_data_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:audio")

def _first_url_in_dict(d: dict) -> str | None:
    # check common keys first
    for k in ("url","audio_url","href","link","src"):
        v = d.get(k)
        if _is_url(v):
            return v
    # otherwise scan any str value that looks like a URL (and likely audio)
    for v in d.values():
        if isinstance(v, str) and _is_url(v):
            return v
    return None

def _first_path_in_dict(d: dict) -> str | None:
    for k in ("path","filepath","file","tmp_path","audio_path","local_path","audio_file","filename"):
        v = d.get(k)
        if isinstance(v, str) and os.path.isfile(v):
            return v
    return None

def _first_bytes_in_dict(d: dict) -> bytes | None:
    for k in ("bytes","data","content"):
        v = d.get(k)
        if isinstance(v, (bytes, bytearray, io.BytesIO)):
            return v if isinstance(v, (bytes, bytearray)) else v.getvalue()
        if isinstance(v, str) and _is_data_uri(v):
            m = re.match(r"data:(audio/[A-Za-z0-9.+-]+);base64,(.*)$", v, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    return base64.b64decode(m.group(2))
                except Exception:
                    pass
    return None

# ----------------- node -----------------

class AudioToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),         # keep single type to satisfy validator
                "filename": ("STRING", {"default": "audio.wav"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, audio, filename="audio.wav", force_upload=False, debug=False):
        # 1) If it's actually a plain string (some toolchains wrap URL as AUDIO)
        if isinstance(audio, str):
            if debug: print("[AudioToURL_0x0] got STRING:", audio[:120])
            if _is_url(audio) and not force_upload:
                return (audio,)
            if os.path.isfile(audio):
                with open(audio, "rb") as f:
                    data = f.read()
                fn = os.path.basename(audio)
                return (_upload_bytes(fn, data),)
            if _is_data_uri(audio):
                m = re.match(r"data:(audio/[A-Za-z0-9.+-]+);base64,(.*)$", audio, re.IGNORECASE | re.DOTALL)
                if m:
                    ext = m.group(1).split("/")[-1]
                    bb = base64.b64decode(m.group(2))
                    fn = filename.strip() or f"audio.{ext}"
                    if not fn.lower().endswith(f".{ext}"):
                        fn = f"{fn}.{ext}"
                    return (_upload_bytes(fn, bb),)
            raise RuntimeError("AudioToURL_0x0: STRING provided but not URL/path/data URI")

        # 2) Dict-like AUDIO
        if hasattr(audio, "get"):
            d = audio
            if debug: print("[AudioToURL_0x0] dict keys:", list(d.keys()))
            u = _first_url_in_dict(d)
            if u and not force_upload:
                return (u,)
            p = _first_path_in_dict(d)
            if p:
                with open(p, "rb") as f:
                    data = f.read()
                fn = os.path.basename(p)
                return (_upload_bytes(fn, data),)
            b = _first_bytes_in_dict(d)
            if b:
                fn = filename.strip() or "audio.bin"
                return (_upload_bytes(fn, b),)
            # samples + sr variants
            samples = None
            for sk in ("samples","waveform","audio","array","tensor"):
                if sk in d:
                    samples = d[sk]; break
            sr = None
            for rk in ("sample_rate","sr","rate","sampleRate"):
                if rk in d:
                    sr = d[rk]; break
            if samples is not None:
                wav = _samples_to_wav(samples, int(sr or 44100))
                fn = filename.strip() or "audio.wav"
                if not fn.lower().endswith(".wav"):
                    fn = f"{fn}.wav"
                return (_upload_bytes(fn, wav),)

        # 3) tuple (samples, sr)
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            samples, sr = audio
            wav = _samples_to_wav(samples, int(sr or 44100))
            fn = filename.strip() or "audio.wav"
            if not fn.lower().endswith(".wav"):
                fn = f"{fn}.wav"
            return (_upload_bytes(fn, wav),)

        # 4) object with attrs
        if hasattr(audio, "samples") and hasattr(audio, "sample_rate"):
            wav = _samples_to_wav(getattr(audio, "samples"), int(getattr(audio, "sample_rate") or 44100))
            fn = filename.strip() or "audio.wav"
            if not fn.lower().endswith(".wav"):
                fn = f"{fn}.wav"
            return (_upload_bytes(fn, wav),)

        raise RuntimeError("AudioToURL_0x0: unsupported AUDIO payload")
        
NODE_CLASS_MAPPINGS = {"AudioToURL_0x0": AudioToURL_0x0}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioToURL_0x0": "Audio → URL (0x0.st)"}
