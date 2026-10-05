# PAI Studio · worker de video para RunPod Serverless (H3, LTX-2.5, VFX).
# Base oficial de RunPod: ComfyUI 0.34.0 (misma versión que tu Mac), que ya trae
# los nodos nativos de MiniMax H3 y LTX-2.5. Los modelos NO van en la imagen:
# viven en el disco de red (/runpod-volume/models).
FROM runpod/worker-comfyui:5.10.0-base

# Nodos extra: los de tus flujos de H3 (probados en la Mac) y los de VFX de LTX-2.5
# (IC-LoRA, tiled fusion, inpaint/outpaint), fijados a una versión concreta.
# De ComfyUI-LTXVideo se instalan sus dependencias sin diffusers: no lo usa y choca
# con huggingface-hub<1.0, que necesita el script de descarga.
RUN set -eux; cd /comfyui/custom_nodes; \
    clone() { git clone --quiet "https://github.com/$1.git" "$2" && git -C "$2" checkout --quiet "$3"; }; \
    clone city96/ComfyUI-GGUF                 ComfyUI-GGUF               6ea2651; \
    clone nicolab28/ComfyUI-ClipProj          ComfyUI-ClipProj           c01ba8f; \
    clone NikoDemon80/ComfyUI-H3-Motion-Context ComfyUI-H3-Motion-Context 5335715; \
    clone PlagueKind/ComfyUI-PlagueKind-Nodes ComfyUI-PlagueKind-Nodes   aaec055; \
    clone obvpm/comfyui-obvpm                 comfyui-obvpm              704fe3e; \
    clone Lightricks/ComfyUI-LTXVideo         ComfyUI-LTXVideo           3bf3ca6; \
    clone kijai/ComfyUI-KJNodes               ComfyUI-KJNodes            d3cfe21; \
    for r in */requirements.txt; do case "$r" in ComfyUI-LTXVideo/*) ;; *) uv pip install -r "$r";; esac; done; \
    uv pip install "colour-science>=0.4.4" einops kornia openimageio "ninja~=1.11.1.4" timm; \
    uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0" hf_transfer

# Rutas de modelos en el disco de red (incluye diffusion_models, text_encoders, etc.).
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# Prueba de arranque en CPU: si algún nodo rompe ComfyUI, la construcción falla aquí
# y no en un worker pagado.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Handler de PAI: envuelve el oficial para videos grandes (entrada/salida por el disco).
RUN mv /handler.py /handler_base.py
COPY pai_handler.py /handler.py
COPY scripts/descargar_modelos.py /pai/descargar_modelos.py
