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


def handler(job):
    job_input = job.get("input") or {}
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
