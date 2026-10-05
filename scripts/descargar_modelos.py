#!/usr/bin/env python3
"""Descarga los modelos de PAI al disco de red de RunPod.

Se corre UNA vez en un Pod temporal (CPU) con el disco `pai-video` montado en /workspace:

    pip install -q "huggingface_hub[cli]<1.0" hf_transfer
    export HF_TOKEN=hf_xxx            # tu token de lectura (no lo compartas)
    python descargar_modelos.py --listar      # muestra qué archivos existen
    python descargar_modelos.py               # descarga el plan de abajo

Los archivos quedan en /workspace/models/<carpeta>/, que el endpoint ve como
/runpod-volume/models/<carpeta>/.
"""
import argparse
import fnmatch
import os
import shutil
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

DEST = os.environ.get("PAI_MODELS", "/workspace/models")

# (repo, carpeta destino, patrones en orden de preferencia: se toma el PRIMERO que exista)
PLAN = [
    # LTX-2.5 (requiere aceptar la licencia en huggingface.co/Lightricks/LTX-2.5)
    ("Lightricks/LTX-2.5", "diffusion_models", ["diffusion_models/*distilled*fp8*", "diffusion_models/*distilled*int8*"]),
    ("Lightricks/LTX-2.5", "text_encoders", ["text_encoders/*fp8*", "text_encoders/*int8*"]),
    ("Lightricks/LTX-2.5", "vae", ["vae/*video-vae*"]),
    ("Lightricks/LTX-2.5", "vae", ["vae/*audio-vae*"]),
    ("Lightricks/LTX-2.5", "latent_upscale_models", ["latent_upscale_models/*spatial*x2*"]),
    # MiniMax H3 (repack de Comfy-Org)
    ("Comfy-Org/MiniMax-H3", "diffusion_models", ["split_files/diffusion_models/*fl2va*fp8_scaled*", "split_files/diffusion_models/*fl2va*bf16*"]),
    ("Comfy-Org/MiniMax-H3", "diffusion_models", ["split_files/diffusion_models/*ref2va*fp8_scaled*", "split_files/diffusion_models/*ref2va*bf16*"]),
    ("Comfy-Org/MiniMax-H3", "text_encoders", ["split_files/text_encoders/*nvfp4*", "split_files/text_encoders/*int8*"]),
    ("Comfy-Org/MiniMax-H3", "vae", ["split_files/vae/*video*"]),
    ("Comfy-Org/MiniMax-H3", "vae", ["split_files/vae/*audio*"]),
    ("Comfy-Org/MiniMax-H3", "loras", ["split_files/loras/*turbo*8*step*", "split_files/loras/*turbo*"]),
]


def pick(files, patterns):
    for pat in patterns:
        hits = sorted(f for f in files if fnmatch.fnmatch(f.lower(), pat.lower()))
        # Prefer non-pruned full models when both exist.
        full = [h for h in hits if "prun" not in h.lower()]
        if hits:
            return (full or hits)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true", help="solo lista los archivos de cada repo")
    ap.add_argument("--simular", action="store_true", help="muestra el plan sin descargar")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Falta HF_TOKEN (export HF_TOKEN=...).")
    api = HfApi(token=token)
    listing = {}
    for repo in sorted({r for r, _, _ in PLAN}):
        try:
            infos = api.list_repo_tree(repo, recursive=True, expand=True)
            listing[repo] = {i.path: getattr(i, "size", 0) or 0 for i in infos if hasattr(i, "size")}
        except Exception as e:  # gated repo without accepted license, typo, etc.
            sys.exit(f"No pude leer {repo}: {e}\n¿Aceptaste su licencia en Hugging Face con esta cuenta?")
    if args.listar:
        for repo, files in listing.items():
            print(f"\n== {repo}")
            for path, size in sorted(files.items()):
                if path.endswith((".safetensors", ".gguf")):
                    print(f"  {size / 1e9:7.2f} GB  {path}")
        return
    plan, total = [], 0
    for repo, folder, patterns in PLAN:
        f = pick(listing[repo], patterns)
        if not f:
            print(f"!! Sin coincidencia en {repo} para {patterns}")
            continue
        size = listing[repo][f]
        total += size
        plan.append((repo, folder, f, size))
        print(f"{size / 1e9:7.2f} GB  {repo}/{f}  ->  models/{folder}/")
    free = shutil.disk_usage(os.path.dirname(DEST) if not os.path.exists(DEST) else DEST).free
    print(f"\nTotal: {total / 1e9:.1f} GB · libre en el disco: {free / 1e9:.1f} GB")
    if args.simular:
        return
    if total > free:
        sys.exit("No cabe. Agranda el disco en RunPod (Storage → pai-video → Expand) y vuelve a correr.")
    for repo, folder, f, size in plan:
        target = os.path.join(DEST, folder, os.path.basename(f))
        if os.path.exists(target) and os.path.getsize(target) == size:
            print(f"ya existe {target}")
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"descargando {f} ...", flush=True)
        path = hf_hub_download(repo, f, token=token, local_dir=os.path.join(DEST, ".hf", repo.replace("/", "__")))
        shutil.move(path, target)
    shutil.rmtree(os.path.join(DEST, ".hf"), ignore_errors=True)
    print("\nListo. Modelos en", DEST)


if __name__ == "__main__":
    main()
