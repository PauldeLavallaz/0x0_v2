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

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm")

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
    for k in ("url", "audio_url", "href", "link", "src"):
        v = d.get(k)
        if _is_url(v):
            return v
    for v in d.values():
        if isinstance(v, str) and _is_url(v):
            return v
    return None

def _first_path_in_dict(d: dict) -> str | None:
    for k in ("path", "filepath", "file", "tmp_path", "audio_path", "local_path", "audio_file", "filename"):
        v = d.get(k)
        if isinstance(v, str) and os.path.isfile(v):
            return v
    return None

def _first_bytes_in_dict(d: dict) -> bytes | None:
    for k in ("bytes", "data", "content"):
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

def _get_attr(o, names):
    for n in names:
        if hasattr(o, n):
            v = getattr(o, n)
            if v is not None:
                return v
    return None

def _upload_wav_from(samples, sr, filename):
    wav = _samples_to_wav(samples, int(sr or 44100))
    fn = (filename or "audio.wav").strip()
    if not fn.lower().endswith(".wav"):
        fn = f"{fn}.wav"
    return _upload_bytes(fn, wav)


# ----------------- node -----------------

class AudioToURL_0x0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),   # admite objetos de Deploy, dicts, str, (samples,sr)
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
        if audio is None:
            raise RuntimeError("AudioToURL_0x0: no audio provided")

        # ---------- 0) Unwrap contenedores comunes ----------
        # Muchos wrappers (incl. Deploy) guardan el verdadero payload en .audio/.value/etc.
        for attr in ("audio", "value", "payload", "data"):
            if hasattr(audio, attr):
                inner = getattr(audio, attr)
                if inner is not None:
                    audio = inner
                    break

        # ---------- 1) STRING ----------
        if isinstance(audio, str):
            if debug: print("[AudioToURL_0x0] got STRING:", audio[:120])
            # URL directa
            if _is_url(audio) and not force_upload:
                return (audio,)
            # Path local
            if os.path.isfile(audio):
                with open(audio, "rb") as f:
                    return (_upload_bytes(os.path.basename(audio), f.read()),)
            # Data URI
            if _is_data_uri(audio):
                m = re.match(r"data:(audio/[A-Za-z0-9.+-]+);base64,(.*)$", audio, re.IGNORECASE | re.DOTALL)
                if m:
                    ext = m.group(1).split("/")[-1].lower()
                    bb  = base64.b64decode(m.group(2))
                    fn  = (filename or f"audio.{ext}").strip()
                    if not fn.lower().endswith(f".{ext}"):
                        fn = f"{fn}.{ext}"
                    return (_upload_bytes(fn, bb),)
            raise RuntimeError("AudioToURL_0x0: STRING provided but not URL/path/data URI")

        # ---------- 2) DICT-LIKE (incl. Deploy que emite dict) ----------
        if hasattr(audio, "get"):  # mapping
            d = audio
            if debug: print("[AudioToURL_0x0] dict keys:", list(d.keys()))
            u = _first_url_in_dict(d)
            if u and not force_upload:
                return (u,)
            p = _first_path_in_dict(d)
            if p:
                with open(p, "rb") as f:
                    return (_upload_bytes(os.path.basename(p), f.read()),)
            b = _first_bytes_in_dict(d)
            if b:
                fn = (filename or "audio.bin").strip()
                return (_upload_bytes(fn, b),)
            # (samples, sr) embebido en dict
            samples = None
            for sk in ("samples", "waveform", "audio", "array", "tensor"):
                if sk in d:
                    samples = d[sk]; break
            sr = None
            for rk in ("sample_rate", "sr", "rate", "sampleRate"):
                if rk in d:
                    sr = d[rk]; break
            if samples is not None:
                return (_upload_wav_from(samples, sr, filename),)

        # ---------- 2.5) OBJETO con atributos (Deploy wrapper) ----------
        # Acepta atributos típicos: url / path / bytes / file-like / data URI / samples+sr
        if not hasattr(audio, "get"):
            # URL en atributo
            u = _get_attr(audio, ("url", "audio_url", "href", "link", "src"))
            if isinstance(u, str) and _is_url(u) and not force_upload:
                if debug: print("[AudioToURL_0x0] attr URL:", u)
                return (u,)

            # Path en atributo
            p = _get_attr(audio, ("path", "filepath", "file", "tmp_path", "audio_path", "local_path", "audio_file", "filename"))
            if isinstance(p, str) and os.path.isfile(p):
                if debug: print("[AudioToURL_0x0] attr PATH:", p)
                with open(p, "rb") as f:
                    return (_upload_bytes(os.path.basename(p), f.read()),)

            # File-like (stream)
            fobj = _get_attr(audio, ("fileobj", "fp", "stream"))
            if fobj and hasattr(fobj, "read"):
                bb = fobj.read()
                if isinstance(bb, (bytes, bytearray)):
                    if debug: print("[AudioToURL_0x0] attr FILEOBJ bytes")
                    fn = (filename or "audio.bin").strip()
                    return (_upload_bytes(fn, bb if isinstance(bb, (bytes, bytearray)) else bytes(bb)),)

            # Bytes en atributo o Data URI
            bb = _get_attr(audio, ("bytes", "data", "content"))
            if isinstance(bb, io.BytesIO):
                bb = bb.getvalue()
            if isinstance(bb, (bytes, bytearray)):
                if debug: print("[AudioToURL_0x0] attr BYTES")
                fn = (filename or "audio.bin").strip()
                return (_upload_bytes(fn, bb),)
            durl = _get_attr(audio, ("data_url", "datauri"))
            if isinstance(durl, str) and _is_data_uri(durl):
                m = re.match(r"data:(audio/[A-Za-z0-9.+-]+);base64,(.*)$", durl, re.IGNORECASE | re.DOTALL)
                if m:
                    ext = m.group(1).split("/")[-1].lower()
                    bb  = base64.b64decode(m.group(2))
                    fn  = (filename or f"audio.{ext}").strip()
                    if not fn.lower().endswith(f".{ext}"):
                        fn = f"{fn}.{ext}"
                    return (_upload_bytes(fn, bb),)

            # (samples, sr) por atributos
            samples = _get_attr(audio, ("samples", "waveform", "array", "tensor"))
            sr      = _get_attr(audio, ("sample_rate", "sr", "rate", "sampleRate"))
            if samples is not None:
                if debug: print("[AudioToURL_0x0] attr SAMPLES+SR")
                return (_upload_wav_from(samples, sr, filename),)

        # ---------- 3) tuple/list (samples, sr) ----------
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            samples, sr = audio
            if debug: print("[AudioToURL_0x0] tuple/list SAMPLES+SR")
            return (_upload_wav_from(samples, sr, filename),)

        # ---------- 4) objeto con .samples/.sample_rate ----------
        if hasattr(audio, "samples") and hasattr(audio, "sample_rate"):
            if debug: print("[AudioToURL_0x0] object SAMPLES+SR attrs")
            return (_upload_wav_from(getattr(audio, "samples"), int(getattr(audio, "sample_rate") or 44100), filename),)

        raise RuntimeError("AudioToURL_0x0: unsupported AUDIO payload")


NODE_CLASS_MAPPINGS = {"AudioToURL_0x0": AudioToURL_0x0}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioToURL_0x0": "Audio → URL (0x0.st)"}
