"""PAI Studio wrapper around the official worker-comfyui handler.

RunPod limits a job's payload (10 MB for /run), which a video reference or a
finished 10-second clip easily exceeds. Large files therefore travel through
the network volume, which PAI reads and writes with RunPod's S3-compatible API:

  input.volume_inputs  = ["pai/in/<id>/clip.mp4", ...]  -> copied to ComfyUI's input/
  outputs larger than INLINE_MAX (or all, with input.outputs_to_volume) are saved to
  pai/out/<job_id>/<file> and returned as {"type": "volume", "path": ...}.

Everything else is the official handler unchanged.
"""
import base64
import os
import shutil
import sys
import time

import runpod

import handler_base as base

VOLUME = os.environ.get("PAI_VOLUME", "/runpod-volume")
COMFY_INPUT = os.environ.get("PAI_COMFY_INPUT", "/comfyui/input")
INLINE_MAX = int(os.environ.get("PAI_INLINE_MAX", str(6 * 1024 * 1024)))
KEEP_SECONDS = int(os.environ.get("PAI_KEEP_SECONDS", str(2 * 24 * 3600)))


def _inside(root, rel):
    path = os.path.realpath(os.path.join(root, rel))
    if not path.startswith(os.path.realpath(root) + os.sep):
        raise ValueError(f"Ruta fuera del disco: {rel}")
    return path


def _cleanup(folder):
    """Old staged files are removed so the disk never fills up."""
    now = time.time()
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(p) > KEEP_SECONDS:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def _models_on_disk():
    root = os.path.join(VOLUME, "models")
    out = {}
    for folder in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, folder)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if not f.startswith(".")]
            if files:
                out[folder] = {f: round(os.path.getsize(os.path.join(p, f)) / 1e9, 2) for f in sorted(files)}
    return out


def maintenance(job, action):
    """Admin jobs: ping, listar (plan without downloading) and descargar (fetch models).

    The Hugging Face token comes from the endpoint's HF_TOKEN environment variable,
    never from the request, so it does not show up in job history.
    """
    usage = shutil.disk_usage(VOLUME) if os.path.isdir(VOLUME) else None
    info = {
        "gpu": os.environ.get("RUNPOD_GPU_NAME") or _gpu_name(),
        "volume_free_gb": round(usage.free / 1e9, 1) if usage else None,
        "volume_total_gb": round(usage.total / 1e9, 1) if usage else None,
        "models": _models_on_disk(),
    }
    if action == "ping":
        return info
    sys.path.insert(0, "/pai")
    import descargar_modelos as dm  # noqa: WPS433 (only needed for admin jobs)

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"error": "Falta la variable HF_TOKEN en el endpoint (Manage → Edit → Environment variables)."}
    # Optional extra files without rebuilding: [{"repo","file","folder"}]
    extra = [e for e in (job.get("input") or {}).get("extra_files") or []
             if isinstance(e, dict) and all(isinstance(e.get(k), str) for k in ("repo", "file", "folder"))
             and e["folder"].replace("_", "").isalnum() and ".." not in e["file"]]
    try:
        listing = dm.read_listing(token, [e["repo"] for e in extra])
    except RuntimeError as e:
        return {"error": str(e)}
    plan, missing = dm.make_plan(listing, extra)
    total = sum(i["size"] for i in plan)
    info.update({
        "plan": [{k: i[k] for k in ("repo", "file", "folder", "gb")} for i in plan],
        "missing": missing,
        "plan_total_gb": round(total / 1e9, 1),
    })
    if action == "listar":
        info["all_files"] = {
            repo: {p: round(s / 1e9, 2) for p, s in files.items() if p.endswith((".safetensors", ".gguf"))}
            for repo, files in listing.items()
        }
        return info
    if action == "descargar":
        if usage and total > usage.free:
            # Partially downloaded files are skipped on the next run, so only the rest counts.
            pending = sum(i["size"] for i in plan if not os.path.exists(os.path.join(dm.DEST, i["folder"], os.path.basename(i["file"]))))
            if pending > usage.free:
                return {**info, "error": f"No cabe: faltan {pending / 1e9:.1f} GB y hay {usage.free / 1e9:.1f} GB libres. Agranda el disco."}
        log = lambda msg: runpod.serverless.progress_update(job, msg)  # noqa: E731
        try:
            dm.download(plan, token, log=log)
        except Exception as e:  # network hiccup: running the job again resumes
            return {**info, "error": f"Descarga interrumpida ({e}). Vuelve a mandar 'descargar' y continúa donde se quedó."}
        info["models"] = _models_on_disk()
        info["status"] = "listo"
        return info
    return {"error": f"Acción desconocida: {action}"}


def _gpu_name():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def handler(job):
    job_input = job.get("input") or {}
    action = job_input.get("pai_action")
    if action:
        return maintenance(job, action)
    for folder in ("pai/in", "pai/out"):
        _cleanup(os.path.join(VOLUME, folder))

    os.makedirs(COMFY_INPUT, exist_ok=True)
    for rel in job_input.get("volume_inputs") or []:
        if not isinstance(rel, str) or not rel.startswith("pai/in/"):
            return {"error": f"volume_inputs inválido: {rel!r}"}
        try:
            src = _inside(os.path.join(VOLUME, "pai", "in"), rel[len("pai/in/"):])
        except ValueError as e:
            return {"error": str(e)}
        if not os.path.isfile(src):
            return {"error": f"No encontré en el disco: {rel}"}
        shutil.copyfile(src, os.path.join(COMFY_INPUT, os.path.basename(src)))

    result = base.handler(job)
    if not isinstance(result, dict) or not result.get("images"):
        return result

    to_volume = bool(job_input.get("outputs_to_volume"))
    out_dir = os.path.join(VOLUME, "pai", "out", str(job.get("id", "job")))
    items = []
    for item in result["images"]:
        if item.get("type") == "base64":
            data = base64.b64decode(item["data"])
            if to_volume or len(data) > INLINE_MAX:
                os.makedirs(out_dir, exist_ok=True)
                name = os.path.basename(item.get("filename") or "output.bin")
                with open(os.path.join(out_dir, name), "wb") as f:
                    f.write(data)
                item = {
                    "filename": name,
                    "type": "volume",
                    "path": os.path.relpath(os.path.join(out_dir, name), VOLUME),
                    "bytes": len(data),
                }
        items.append(item)
    result["images"] = items
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
