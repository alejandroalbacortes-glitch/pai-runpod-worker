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

DEST = os.environ.get("PAI_MODELS", "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/workspace/models")

# (repo, carpeta destino, patrones en orden de preferencia: se toma el PRIMERO que exista)
PLAN = [
    # LTX-2.5 (requiere aceptar la licencia en huggingface.co/Lightricks/LTX-2.5) · ~39.7 GB
    ("Lightricks/LTX-2.5", "diffusion_models", ["diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"]),
    ("Lightricks/LTX-2.5", "text_encoders", ["text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"]),
    ("Lightricks/LTX-2.5", "vae", ["vae/ltx-2.5-video-vae-bf16.safetensors"]),
    ("Lightricks/LTX-2.5", "vae", ["vae/ltx-2.5-audio-vae-bf16.safetensors"]),
    ("Lightricks/LTX-2.5", "latent_upscale_models", ["latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"]),
    # MiniMax H3 (repack de Comfy-Org), versiones "pruned" fp8 para que todo quepa en 121 GB · ~67 GB
    ("Comfy-Org/MiniMax-H3", "diffusion_models", ["diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"]),
    ("Comfy-Org/MiniMax-H3", "diffusion_models", ["diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"]),
    ("Comfy-Org/MiniMax-H3", "text_encoders", ["text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]),
    ("Comfy-Org/MiniMax-H3", "vae", ["vae/minimax_h3_video_vae_fp16.safetensors"]),
    ("Comfy-Org/MiniMax-H3", "vae", ["vae/minimax_h3_audio_vae_fp32.safetensors"]),
    ("Comfy-Org/MiniMax-H3", "loras", ["loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"]),
    ("Comfy-Org/MiniMax-H3", "loras", ["loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"]),
]


def pick(files, patterns):
    for pat in patterns:
        hits = sorted(f for f in files if fnmatch.fnmatch(f.lower(), pat.lower()))
        # Prefer non-pruned full models when both exist.
        full = [h for h in hits if "prun" not in h.lower()]
        if hits:
            return (full or hits)[0]
    return None


def read_listing(token, extra_repos=None):
    api = HfApi(token=token)
    listing = {}
    for repo in sorted({r for r, _, _ in PLAN} | set(extra_repos or [])):
        try:
            infos = api.list_repo_tree(repo, recursive=True, expand=True)
            listing[repo] = {i.path: getattr(i, "size", 0) or 0 for i in infos if hasattr(i, "size")}
        except Exception as e:  # gated repo without accepted license, typo, etc.
            raise RuntimeError(f"No pude leer {repo}: {e}. ¿Aceptaste su licencia en Hugging Face con esta cuenta?")
    return listing


def make_plan(listing, extra=None):
    plan, missing = [], []
    entries = list(PLAN) + [(e["repo"], e["folder"], [e["file"]]) for e in (extra or [])]
    for repo, folder, patterns in entries:
        if repo not in listing:
            missing.append(f"{repo}: repo no leído")
            continue
        f = pick(listing[repo], patterns)
        if not f:
            missing.append(f"{repo}: {patterns}")
            continue
        plan.append({"repo": repo, "folder": folder, "file": f, "gb": round(listing[repo][f] / 1e9, 2), "size": listing[repo][f]})
    return plan, missing


def free_bytes():
    probe = DEST if os.path.exists(DEST) else os.path.dirname(DEST)
    return shutil.disk_usage(probe).free


def download(plan, token, log=print):
    done = []
    for item in plan:
        target = os.path.join(DEST, item["folder"], os.path.basename(item["file"]))
        if os.path.exists(target) and os.path.getsize(target) == item["size"]:
            log(f"ya existe {target}")
            done.append(target)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        log(f"descargando {item['file']} ({item['gb']} GB) ...")
        path = hf_hub_download(item["repo"], item["file"], token=token, local_dir=os.path.join(DEST, ".hf", item["repo"].replace("/", "__")))
        shutil.move(path, target)
        done.append(target)
    shutil.rmtree(os.path.join(DEST, ".hf"), ignore_errors=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true", help="solo lista los archivos de cada repo")
    ap.add_argument("--simular", action="store_true", help="muestra el plan sin descargar")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Falta HF_TOKEN (export HF_TOKEN=...).")
    try:
        listing = read_listing(token)
    except RuntimeError as e:
        sys.exit(str(e))
    if args.listar:
        for repo, files in listing.items():
            print(f"\n== {repo}")
            for path, size in sorted(files.items()):
                if path.endswith((".safetensors", ".gguf")):
                    print(f"  {size / 1e9:7.2f} GB  {path}")
        return
    plan, missing = make_plan(listing)
    for m in missing:
        print("!! Sin coincidencia:", m)
    for it in plan:
        print(f"{it['gb']:7.2f} GB  {it['repo']}/{it['file']}  ->  models/{it['folder']}/")
    total = sum(i["size"] for i in plan)
    print(f"\nTotal: {total / 1e9:.1f} GB · libre en el disco: {free_bytes() / 1e9:.1f} GB")
    if args.simular:
        return
    if total > free_bytes():
        sys.exit("No cabe. Agranda el disco en RunPod (Storage → pai-video → Expand) y vuelve a correr.")
    download(plan, token)
    print("\nListo. Modelos en", DEST)


if __name__ == "__main__":
    main()
