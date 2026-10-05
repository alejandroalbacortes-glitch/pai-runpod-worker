# PAI Studio · worker de video (RunPod Serverless)

Imagen de ComfyUI para generar video desde PAI Studio con **MiniMax H3**, **LTX-2.5** y VFX,
pagando solo por segundo de GPU. Los modelos viven en el disco de red `pai-video`, no aquí.

## Qué contiene
- `Dockerfile`: parte de la imagen oficial `runpod/worker-comfyui:5.10.0-base` (ComfyUI 0.34.0)
  y agrega los nodos que usan tus flujos de H3 (GGUF, ClipProj, H3 Motion Context, PlagueKind, obvpm).
- `extra_model_paths.yaml`: le dice a ComfyUI dónde están los modelos en el disco.
- `pai_handler.py`: permite mandar y recibir videos grandes a través del disco
  (RunPod limita cada petición a 10 MB).
- `scripts/descargar_modelos.py`: descarga LTX-2.5 y H3 al disco, una sola vez.

## Cómo se usa
1. Sube estos archivos a un repositorio de GitHub (puede ser privado).
2. En RunPod: Serverless → New Endpoint → GitHub → elige el repositorio.
3. Configura GPUs, Workers en 0 y el disco `pai-video`.
4. Carga los modelos con `scripts/descargar_modelos.py` en un Pod temporal.
5. Pega tu API key y el Endpoint ID en PAI → Conexiones → RunPod.
