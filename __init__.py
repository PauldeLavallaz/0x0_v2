
import os, io, re, base64, wave, requests

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

# ----------------- utils -----------------

def _upload_bytes(filename: str, data: bytes) -> str:
    r = requests.post("https://0x0.st", files={"file": (filename, data)}, timeout=60)
    r.raise_for_status()
    url = r.text.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise RuntimeError(f"0x0.st bad response: {url[:120]}")
    return url

def _img_to_bytes(image, fmt="png", quality=95) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow required")
    if torch is not None and isinstance(image, torch.Tensor):
        t = image
        if t.ndim == 4:
            # B,C,H,W or B,H,W,C -> make HWC
            if t.shape[1] in (1,3) and t.shape[-1] not in (1,3):
                t = t.permute(0,2,3,1)
            t = t[0]
        if t.ndim != 3:
            raise RuntimeError("Unexpected IMAGE tensor shape")
        arr = t.detach().cpu().clamp(0,1).mul(255).round().byte().numpy()
    else:
        raise RuntimeError("Unsupported IMAGE type")
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    pil = Image.fromarray(arr, mode=mode)
    bio = io.BytesIO()
    if fmt == "jpeg":
        pil = pil.convert("RGB")
        pil.save(bio, format="JPEG", quality=int(quality))
    elif fmt == "png":
        pil.save(bio, format="PNG")
    elif fmt == "webp":
        pil.save(bio, format="WEBP", quality=int(quality))
    else:
        raise RuntimeError(f"Unsupported image format: {fmt}")
    return bio.getvalue()

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

def _is_data_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:")

# ----------------- nodes -----------------

class ImageToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_format": (["png","jpeg","webp"], {"default": "png"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "filename_hint": ("STRING", {"default": "image.png"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, image, image_format="png", jpeg_quality=95, filename_hint="image.png"):
        data = _img_to_bytes(image, fmt=image_format, quality=jpeg_quality)
        ext = {"png":"png","jpeg":"jpg","webp":"webp"}[image_format]
        fn = filename_hint.strip() or f"image.{ext}"
        if not fn.lower().endswith(f".{ext}"):
            fn = f"{fn}.{ext}"
        url = _upload_bytes(fn, data)
        return (url,)

class PathToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "filename_hint": ("STRING", {"default": "file.bin"}),
                "force_upload": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, path, filename_hint="file.bin", force_upload=False):
        if not isinstance(path, str) or not path:
            raise RuntimeError("PathToURL_0x0 expects STRING")
        # passthrough URLs
        if not force_upload and (path.startswith("http://") or path.startswith("https://") or path.startswith("s3://")):
            return (path,)
        # data URI
        if _is_data_uri(path):
            m = re.match(r"data:([^;]+);base64,(.*)$", path, re.IGNORECASE | re.DOTALL)
            if not m:
                raise RuntimeError("Unsupported data URI")
            mime, b64 = m.groups()
            data = base64.b64decode(b64)
            ext = mime.split("/")[-1]
            fn = filename_hint.strip() or f"file.{ext}"
            if not fn.lower().endswith(f".{ext}"):
                fn = f"{fn}.{ext}"
            return (_upload_bytes(fn, data),)
        # local file
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            fn = filename_hint.strip() or os.path.basename(path)
            return (_upload_bytes(fn, data),)
        raise RuntimeError("Invalid path for PathToURL_0x0")

class AudioToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),         # STRICT: Comfy expects a single type here
                "filename": ("STRING", {"default": "audio.wav"}),
                "force_upload": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, audio, filename="audio.wav", force_upload=False):
        # 1) dict AUDIO from Deploy? handle common shapes
        if hasattr(audio, "get"):
            d = audio
            # pass-through if already brings a public URL and we don't force re-upload
            url = d.get("url")
            if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://") or url.startswith("s3://")) and not force_upload:
                return (url,)
            # path on disk?
            for k in ("path","filepath","file","tmp_path","audio_path","local_path","audio_file","filename"):
                p = d.get(k)
                if isinstance(p, str) and os.path.isfile(p):
                    with open(p, "rb") as f:
                        data = f.read()
                    fn = os.path.basename(p)
                    return (_upload_bytes(fn, data),)
            # raw bytes?
            for k in ("bytes","data","content"):
                b = d.get(k)
                if isinstance(b, (bytes, bytearray, io.BytesIO)):
                    bb = b if isinstance(b, (bytes, bytearray)) else b.getvalue()
                    fn = filename.strip() or "audio.bin"
                    return (_upload_bytes(fn, bb),)
                if isinstance(b, str) and _is_data_uri(b):
                    m = re.match(r"data:(audio/[A-Za-z0-9.+-]+);base64,(.*)$", b, re.IGNORECASE | re.DOTALL)
                    if m:
                        ext = m.group(1).split("/")[-1]
                        bb = base64.b64decode(m.group(2))
                        fn = filename.strip() or f"audio.{ext}"
                        if not fn.lower().endswith(f".{ext}"):
                            fn = f"{fn}.{ext}"
                        return (_upload_bytes(fn, bb),)
            # samples + sample_rate
            if "samples" in d:
                sr = int(d.get("sample_rate", 44100) or 44100)
                wav = _samples_to_wav(d["samples"], sr)
                fn = filename.strip() or "audio.wav"
                if not fn.lower().endswith(".wav"):
                    fn = f"{fn}.wav"
                return (_upload_bytes(fn, wav),)

        # 2) tuple (samples, sr)
        if isinstance(audio, tuple) and len(audio) == 2:
            samples, sr = audio
            wav = _samples_to_wav(samples, int(sr or 44100))
            fn = filename.strip() or "audio.wav"
            if not fn.lower().endswith(".wav"):
                fn = f"{fn}.wav"
            return (_upload_bytes(fn, wav),)

        # 3) object with attrs
        if hasattr(audio, "samples") and hasattr(audio, "sample_rate"):
            wav = _samples_to_wav(getattr(audio, "samples"), int(getattr(audio, "sample_rate") or 44100))
            fn = filename.strip() or "audio.wav"
            if not fn.lower().endswith(".wav"):
                fn = f"{fn}.wav"
            return (_upload_bytes(fn, wav),)

        # 4) as fallback: if dict brought a URL under an unusual key, reject clearly
        raise RuntimeError("AudioToURL_0x0: unsupported AUDIO object. Pasá el AUDIO del External Audio o (samples,sr).")

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
