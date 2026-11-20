"""ComfyUI entry point for a clone of this whole repository in ComfyUI/custom_nodes.

The node lives in comfyui_nodes/snapref; copying only that folder to custom_nodes/snapref keeps working. When
both are installed, the copy that loads second registers nothing (comfyui_nodes/snapref/__init__.py), so the
nodes and the web extension are registered once.
"""
try:
    from .comfyui_nodes import snapref as _snapref
except ModuleNotFoundError as exc:   # imported outside ComfyUI (training scripts, test runners)
    if (exc.name or "").split(".")[0] not in ("comfy", "folder_paths"):
        raise
    _snapref = None

NODE_CLASS_MAPPINGS = getattr(_snapref, "NODE_CLASS_MAPPINGS", {})
NODE_DISPLAY_NAME_MAPPINGS = getattr(_snapref, "NODE_DISPLAY_NAME_MAPPINGS", {})
# ComfyUI resolves WEB_DIRECTORY against this folder
WEB_DIRECTORY = "./comfyui_nodes/snapref/web" if getattr(_snapref, "WEB_DIRECTORY", None) else None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
