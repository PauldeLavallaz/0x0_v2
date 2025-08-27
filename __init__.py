import os, io, re, base64, wave, requests, ast

try:
    import torch
except Exception:
        torch = None

try:
    import numpy as np
except Exception:
    np = None


# ============ helpers de red ============

def _ensure_https(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url

def _verify_accessible(url: str, timeout: int = 20) -> bool:
    try:
        h = requests.head(url, allow_redirects=True, timeout=timeout)
        if 200 <= h.status_code < 300:
            return True
        if h.status_code in (403, 405):
            g = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
            ok = 200 <= g.status_code < 300
            try:
                next(g.iter_content(chunk_size=1))
            except Exception:
                pass
            g.close()
            return ok
        return False
    except Exception:
        return False


# ============ uploaders (con fallback) ============

def _upload_0x0(filename: str, data: bytes) -> str:
    r = requests.post("https://0x0.st", files={"file": (filename, data)},
                      timeout=60, headers={"User-Agent": "curl/8.0"})
    r.raise_for_status()
    return _ensure_https(r.text.strip())

def _upload_transfer_sh(filename: str, data: bytes) -> str:
    r = requests.put(f"https://transfer.sh/{filename}", data=data,
                     timeout=120, headers={"User-Agent": "curl/8.0"})
    r.raise_for_status()
    return _ensure_https(r.text.strip().split()[0])

def _upload_tmpfiles(filename: str, data: bytes) -> str:
    r = requests.post("https://tmpfiles.org/api/v1/upload",
                      files={"file": (filename, data)}, timeout=120)
    r.raise_for_status()
    j = r.json()
    url = j.get("data", {}).get("url", "")
    url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    return _ensure_https(url)

def _upload_bytes(filename: str, data: bytes, uploader: str = "auto") -> str:
    order = {
        "auto":        (_upload_0x0, _upload_transfer_sh, _upload_tmpfiles),
        "0x0":         (_upload_0x0,),
        "transfer.sh": (_upload_transfer_sh,),
        "tmpfiles":    (_upload_tmpfiles,),
    }.get(uploader, (_upload_transfer_sh,))
    last = None
    for fn in order:
        try:
            url = fn(filename, data)
            url = _ensure_https(url)
            if _verify_accessible(url):
                return url
            last = RuntimeError(f"uploaded but not accessible: {url}")
        except Exception as e:
            last = e
            continue
    raise last if last else RuntimeError("upload failed")


# ============ utils comunes ============

def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("s3://"))

def _is_data_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:")

def _img_to_bytes(image, fmt="png", quality=95) -> bytes:
    from PIL import Image as PILImage
    if torch is not None and isinstance(image, torch.Tensor):
        t = image
        if t.ndim == 4:
            if t.shape[1] in (1, 3) and t.shape[-1] not in (1, 3):
                t = t.permute(0, 2, 3, 1)
            t = t[0]
        if t.ndim != 3:
            raise RuntimeError("Unexpected IMAGE tensor shape")
        arr = t.detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()
    else:
        raise RuntimeError("Unsupported IMAGE type")
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    pil = PILImage.fromarray(arr, mode=mode)
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
        arr = arr[:, None]
    if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
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

def _decode_bytes_string(s: str) -> bytes:
    """
    Acepta:
      - Literal de bytes estilo Python:  b'ID3\\x04...'
      - data URI: data:audio/mpeg;base64,....
      - Base64 “puro”
      - Hex (con o sin 0x)
      - Texto crudo (se interpreta como latin-1)
    """
    if not isinstance(s, str):
        raise RuntimeError("Expected STRING with byte content")
    s = s.strip()

    # 1) bytes literal estilo Python
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        try:
            val = ast.literal_eval(s)  # -> bytes
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
        except Exception:
            pass

    # 2) data URI
    if _is_data_uri(s):
        m = re.match(r"data:([^;]+);base64,(.*)$", s, re.IGNORECASE | re.DOTALL)
        if m:
            b64 = m.group(2)
            return base64.b64decode(b64)
        # data:...;charset=...;... no soportado acá
        raise RuntimeError("Unsupported data URI for bytes string")

    # 3) Base64 “crudo”
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        pass

    # 4) Hex
    hex_s = s[2:] if s.lower().startswith("0x") else s
    try:
        # quitar espacios
        hex_s_clean = re.sub(r"\s+", "", hex_s)
        return bytes.fromhex(hex_s_clean)
    except Exception:
        pass

    # 5) Fallback: interpretar como texto que representa bytes (latin-1)
    return s.encode("latin-1", errors="ignore")


# ============ nodos ============

class ImageToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "filename_hint": ("STRING", {"default": "image.png"}),
                "uploader": (["auto", "0x0", "transfer.sh", "tmpfiles"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, image, image_format="png", jpeg_quality=95, filename_hint="image.png", uploader="auto"):
        data = _img_to_bytes(image, fmt=image_format, quality=jpeg_quality)
        ext = {"png": "png", "jpeg": "jpg", "webp": "webp"}[image_format]
        fn = filename_hint.strip() or f"image.{ext}"
        if not fn.lower().endswith(f".{ext}"):
            fn = f"{fn}.{ext}"
        url = _upload_bytes(fn, data, uploader=uploader)
        return (_ensure_https(url),)


class AudioToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "audio.wav"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "uploader": (["auto", "0x0", "transfer.sh", "tmpfiles"], {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    # Tu lógica original va aquí. Si no usás este nodo, podés ignorarlo.
    def run(self, audio, filename="audio.wav", force_upload=False, uploader="auto", debug=False):
        raise RuntimeError("AudioToURL_0x0.run no implementado en este archivo.")


class PathToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "filename_hint": ("STRING", {"default": "file.bin"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "uploader": (["auto", "0x0", "transfer.sh", "tmpfiles"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, path, filename_hint="file.bin", force_upload=False, uploader="auto"):
        if not isinstance(path, str) or not path:
            raise RuntimeError("PathToURL_0x0 expects STRING")
        if not force_upload and _is_url(path):
            return (_ensure_https(path),)
        if _is_data_uri(path):
            m = re.match(r"data:([^;]+);base64,(.*)$", path, re.IGNORECASE | re.DOTALL)
            if not m: raise RuntimeError("Unsupported data URI")
            mime, b64 = m.groups()
            data = base64.b64decode(b64)
            ext = mime.split("/")[-1]
            fn = filename_hint.strip() or f"file.{ext}"
            if not fn.lower().endswith(f".{ext}"):
                fn = f"{fn}.{ext}"
            return (_upload_bytes(fn, data, uploader=uploader),)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            fn = filename_hint.strip() or os.path.basename(path)
            return (_upload_bytes(fn, data, uploader=uploader),)
        raise RuntimeError("Invalid path for PathToURL_0x0")


class VideoToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),   # acepta salida de Kling u otros nodos
                "filename_hint": ("STRING", {"default": "video.mp4"}),
                "force_upload": ("BOOLEAN", {"default": False}),
                "uploader": (["auto", "0x0", "transfer.sh", "tmpfiles"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, video, filename_hint="video.mp4", force_upload=False, uploader="auto"):
        # caso 1: string
        if isinstance(video, str):
            s = video.strip()
            if not force_upload and _is_url(s):
                return (_ensure_https(s),)
            if _is_data_uri(s) and s.lower().startswith("data:video"):
                m = re.match(r"data:(video/[A-Za-z0-9.+-]+);base64,(.*)$", s, re.IGNORECASE | re.DOTALL)
                if not m:
                    raise RuntimeError("Unsupported video data URI")
                mime, b64 = m.groups()
                data = base64.b64decode(b64)
                ext = mime.split("/")[-1] or "mp4"
                fn = filename_hint.strip() or f"video.{ext}"
                if not fn.lower().endswith(f".{ext}"):
                    fn = f"{fn}.{ext}"
                return (_upload_bytes(fn, data, uploader=uploader),)
            if os.path.isfile(s):
                with open(s, "rb") as f:
                    data = f.read()
                fn = filename_hint.strip() or os.path.basename(s)
                return (_upload_bytes(fn, data, uploader=uploader),)
            raise RuntimeError("Invalid string input for VideoToURL_0x0")

        # caso 2: dict con path
        if hasattr(video, "get"):
            d = video
            if "path" in d and os.path.isfile(d["path"]):
                with open(d["path"], "rb") as f:
                    data = f.read()
                fn = filename_hint.strip() or os.path.basename(d["path"])
                return (_upload_bytes(fn, data, uploader=uploader),)

        # caso 3: objeto con atributo video_path
        if hasattr(video, "video_path") and os.path.isfile(video.video_path):
            with open(video.video_path, "rb") as f:
                data = f.read()
            fn = filename_hint.strip() or os.path.basename(video.video_path)
            return (_upload_bytes(fn, data, uploader=uploader),)

        raise RuntimeError("VideoToURL_0x0: unsupported VIDEO payload")


# ======== NUEVO NODO: ByteString (MP3) → URL ========

class ByteStringMP3ToURL_0x0:
    """
    Recibe bytes en formato STRING (por ejemplo:  b'ID3\\x04...') y sube el MP3,
    devolviendo una URL pública (0x0.st / transfer.sh / tmpfiles).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bytes_string": ("STRING", {"multiline": True, "default": ""}),
                "filename": ("STRING", {"default": "audio.mp3"}),
                "uploader": (["auto", "0x0", "transfer.sh", "tmpfiles"], {"default": "auto"}),
                "force_upload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "run"
    CATEGORY = "I/O → URL"

    def run(self, bytes_string, filename="audio.mp3", uploader="auto", force_upload=True):
        if not isinstance(bytes_string, str) or not bytes_string.strip():
            raise RuntimeError("ByteStringMP3ToURL_0x0: expected non-empty STRING")

        s = bytes_string.strip()

        # Si ya es una URL y no forzamos upload, devolvemos tal cual
        if not force_upload and _is_url(s):
            return (_ensure_https(s),)

        # Decodificar a bytes
        data = _decode_bytes_string(s)

        # Asegurar extensión .mp3
        fn = (filename or "audio.mp3").strip()
        if not fn.lower().endswith(".mp3"):
            fn = f"{fn}.mp3"

        url = _upload_bytes(fn, data, uploader=uploader)
        return (_ensure_https(url),)


# ============ registros ============

NODE_CLASS_MAPPINGS = {
    "ImageToURL_0x0": ImageToURL_0x0,
    "AudioToURL_0x0": AudioToURL_0x0,
    "PathToURL_0x0": PathToURL_0x0,
    "VideoToURL_0x0": VideoToURL_0x0,
    "ByteStringMP3ToURL_0x0": ByteStringMP3ToURL_0x0,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToURL_0x0": "Image → URL (0x0.st)",
    "AudioToURL_0x0": "Audio → URL (0x0.st)",
    "PathToURL_0x0": "Path → URL (0x0.st)",
    "VideoToURL_0x0": "Video → URL (0x0.st)",
    "ByteStringMP3ToURL_0x0": "ByteString (MP3) → URL (0x0.st)",
}
