# comfyui-asset-to-url-0x0 (v5)

**Audio → URL (0x0.st)** con _pass-through_ de URLs HTTP/HTTPS/S3.

Ahora, si tu `External Audio (ComfyUI Deploy)` llega como
`https://comfy-deploy-output.s3.../audio.MP3`, el nodo devuelve **esa misma URL**.
Si recibe audio crudo, lo codifica a WAV y lo sube a 0x0.st (devuelve URL).

Soporta además:
- dict con `samples/sample_rate` o `path/file/...` o `bytes/data/content (incluye data:audio base64)`
- `(samples, sr)`
- ruta local `STRING`
- `bytes/bytearray`
- objeto con `.samples` y `.sample_rate`
