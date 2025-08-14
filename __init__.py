
import os
import io
import re
import base64
import tempfile
import requests
from urllib.parse import urlparse
from typing import Tuple, Union

# Optional deps
try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

# --- helpers ---------------------------------------------------------------

def _is_url(s: str) -> bool:
    try:
        u = urlparse(str(s))
        return u.scheme in ("http", "https", "s3")
    except Exception:
        return False

def _is_data_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:")

def _ensure_bytes(x: Union[bytes, bytearray, io.BytesIO]) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, io.BytesIO):
        return x.getvalue()
    raise RuntimeError("Cannot convert to bytes")

def _upload_bytes_0x0(filename: str, data: bytes) -> str:
    resp = requests.post("https://0x0.st", files={"file": (filename, data)},
                         timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"0x0.st upload failed: {resp.status_code} {resp.text[:200]}")
    url = resp.text.strip()
    # 0x0 returns short URL per line
    if not _is_url(url):
        raise RuntimeError(f"0x0.st returned unexpected response: {url[:200]}")
    return url

def _read_local(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _audio_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """
    Accepts samples as torch.Tensor or numpy.ndarray; mono or stereo.
    Returns 16-bit PCM WAV bytes.
    """
    if sample_rate is None:
        sample_rate = 44100

    # to numpy float32 [-1,1]
    if torch is not None and isinstance(samples, torch.Tensor):
        arr = samples.detach().cpu().float().numpy()
    elif np is not None and isinstance(samples, np.ndarray):
        arr = samples.astype("float32")
    else:
        raise RuntimeError("Unsupported audio sample array type")

    # shape: (channels, n) or (n,) -> (n, channels)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
        # likely (channels, n)
        arr = arr.T  # -> (n, channels)

    # clamp
    arr = arr.clip(-1.0, 1.0)

    # float32 [-1,1] -> int16
    if arr.dtype.kind == "f":
        arr = (arr * 32767.0).round().astype("int16")
    elif arr.dtype != "int16":
        arr = arr.astype("int16")

    import wave
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(arr.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(arr.tobytes(order="C"))
    return bio.getvalue()

def _image_to_bytes(image, fmt: str = "png", jpeg_quality: int = 95) -> bytes:
    """
    Converts ComfyUI IMAGE (tensor BCHW or HWC float [0..1]) to bytes via PIL.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to encode images")

    # Comfy tensor -> numpy HWC uint8
    if torch is not None and isinstance(image, torch.Tensor):
        t = image
        # Comfy usually gives BCHW or BHWC normalized 0..1
        if t.ndim == 4:  # B,H,W,C or B,C,H,W
            if t.shape[1] in (1,3) and t.shape[-1] not in (1,3):
                # B,C,H,W -> B,H,W,C
                t = t.permute(0, 2, 3, 1)
            t = t[0]
        if t.ndim == 3 and t.shape[-1] in (1,3,4):
            arr = t.detach().cpu().clamp(0,1).mul(255).round().byte().numpy()
        else:
            raise RuntimeError("Unexpected IMAGE tensor shape")
    elif np is not None and isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != "uint8":
            arr = np.clip(arr,0,1)
            arr = (arr * 255.0).round().astype("uint8")
        if arr.ndim == 4:
            arr = arr[0]
    else:
        raise RuntimeError("Unsupported IMAGE type")

    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    pil = Image.fromarray(arr, mode=mode)
    out = io.BytesIO()
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        pil = pil.convert("RGB")
        pil.save(out, format="JPEG", quality=int(jpeg_quality))
    elif fmt == "png":
        pil.save(out, format="PNG")
    elif fmt == "webp":
        pil.save(out, format="WEBP", quality=int(jpeg_quality))
    else:
        raise RuntimeError(f"Unsupported image format: {fmt}")
    return out.getvalue()

def _maybe_passthrough_url(x, force_upload: bool) -> Union[str, None]:
    if isinstance(x, str) and (_is_url(x) or _is_data_uri(x)):
        return None if force_upload else x
    return None

# --- Nodes -----------------------------------------------------------------

class ImageToURL_0x0:
    """
    INPUT: IMAGE -> uploads to 0x0.st (PNG/JPEG/WEBP) and returns URL.
    If you already have a http/https/s3 URL STRING, use PathToURL_0x0 instead.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "filename_hint": ("STRING", {"default": "image.png"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, image, image_format, jpeg_quality, filename_hint, force_upload=False, debug=False):
        # IMAGE cannot be URL-string; always encode and upload
        data = _image_to_bytes(image, image_format, jpeg_quality)
        # pick filename
        ext = {"png": "png", "jpeg":"jpg", "webp":"webp"}[image_format]
        fname = filename_hint if filename_hint.strip() else f"image.{ext}"
        if not fname.lower().endswith(f".{ext}"):
            fname = f"{fname}.{ext}"
        url = _upload_bytes_0x0(fname, data)
        return (url,)


class PathToURL_0x0:
    """
    INPUT: STRING path or URL.
      - If http/https/s3: passthrough (unless force_upload=True)
      - If local path: upload bytes to 0x0.st
      - If data:URI: decode and upload
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "filename_hint": ("STRING", {"default": "file.bin"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, path, filename_hint, force_upload=False, debug=False):
        if not isinstance(path, str) or not path:
            raise RuntimeError("PathToURL_0x0 expects a non-empty STRING")

        passthrough = _maybe_passthrough_url(path, force_upload)
        if passthrough:
            return (passthrough,)

        # data URI?
        if _is_data_uri(path):
            m = re.match(r"data:([^;]+);base64,(.*)$", path, re.IGNORECASE)
            if not m:
                raise RuntimeError("Unsupported data: URI")
            mime = m.group(1)
            data = base64.b64decode(m.group(2))
            # pick extension rough
            ext = mime.split("/")[-1]
            fname = filename_hint if filename_hint.strip() else f"file.{ext}"
            if not fname.lower().endswith(f".{ext}"):
                fname = f"{fname}.{ext}"
            url = _upload_bytes_0x0(fname, data)
            return (url,)

        # local file?
        if os.path.isfile(path):
            data = _read_local(path)
            fname = filename_hint if filename_hint.strip() else os.path.basename(path)
            url = _upload_bytes_0x0(fname, data)
            return (url,)

        raise RuntimeError("Invalid path or unsupported scheme")

class AudioToURL_0x0:
    """
    INPUT: AUDIO or STRING
      - If STRING http/https/s3: passthrough (unless force_upload=True)
      - If STRING local path: upload
      - If data:audio/*;base64: decode and upload
      - If AUDIO object (dict/tuple/attrs): encode to WAV and upload
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (["AUDIO","STRING","ANY"],),
                "filename": ("STRING", {"default": "audio.wav"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def _extract_audio(self, audio) -> Tuple[bytes, str]:
        """
        Returns (bytes, suggested_filename)
        """
        # STRING?
        if isinstance(audio, str):
            # passthrough URL?
            if _is_url(audio):
                return None, audio  # None means passthrough
            if _is_data_uri(audio):
                m = re.match(r"data:(audio/[A-Za-z0-9.+-]+);base64,(.*)$", audio, re.IGNORECASE | re.DOTALL)
                if not m:
                    raise RuntimeError("Unsupported data:audio URI")
                mime = m.group(1)
                data = base64.b64decode(m.group(2))
                ext = mime.split("/")[-1].lower()
                return data, f"audio.{ext}"
            # local file path
            if os.path.isfile(audio):
                data = _read_local(audio)
                return data, os.path.basename(audio)
            raise RuntimeError("AudioToURL_0x0: STRING is not a URL nor an existing file")

        # dict-like (common AUDIO type)
        if hasattr(audio, "get"):
            d = audio
            # Already have URL?
            if "url" in d and _is_url(d["url"]):
                return None, d["url"]
            if "bytes" in d:
                return _ensure_bytes(d["bytes"]), d.get("filename","audio.bin")
            if "path" in d and isinstance(d["path"], str) and os.path.isfile(d["path"]):
                return _read_local(d["path"]), os.path.basename(d["path"])
            if "samples" in d:
                sr = int(d.get("sample_rate", 44100) or 44100)
                data = _audio_to_wav_bytes(d["samples"], sr)
                return data, "audio.wav"

        # tuple (samples, sr)
        if isinstance(audio, tuple) and len(audio) == 2:
            samples, sr = audio
            data = _audio_to_wav_bytes(samples, int(sr or 44100))
            return data, "audio.wav"

        # object with attrs .samples and .sample_rate
        if hasattr(audio, "samples"):
            sr = int(getattr(audio, "sample_rate", 44100))
            data = _audio_to_wav_bytes(getattr(audio, "samples"), sr)
            return data, "audio.wav"

        # raw bytes
        if isinstance(audio, (bytes, bytearray, io.BytesIO)):
            return _ensure_bytes(audio), "audio.bin"

        raise RuntimeError("AudioToURL_0x0: unsupported AUDIO object. Pass dict{'samples','sample_rate'}, (samples,sr), local path STRING, URL STRING, data: URI, or bytes.")

    def run(self, audio, filename, force_upload=False, debug=False):
        # If URL string passthrough
        if isinstance(audio, str) and _is_url(audio) and not force_upload:
            return (audio,)

        # Extract
        data_fname = self._extract_audio(audio)
        if data_fname[0] is None and _is_url(data_fname[1]) and not force_upload:
            return (data_fname[1],)

        data, fname_guess = data_fname
        fname = filename.strip() or fname_guess
        # Ensure extension if guess has one
        if "." not in os.path.basename(fname) and "." in fname_guess:
            fname = f"{fname}.{fname_guess.split('.')[-1]}"
        url = _upload_bytes_0x0(fname, data)
        return (url,)


NODE_CLASS_MAPPINGS = {
    "ImageToURL_0x0": ImageToURL_0x0,
    "AudioToURL_0x0": AudioToURL_0x0,
    "PathToURL_0x0": PathToURL_0x0,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToURL_0x0": "Image → URL (0x0.st)",
    "AudioToURL_0x0": "Audio → URL (0x0.st)",
    "PathToURL_0x0": "Path → URL (0x0.st)",
}
