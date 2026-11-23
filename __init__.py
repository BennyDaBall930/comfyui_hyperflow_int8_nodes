# Copyright (C) 2026 the ComfyUI-HyperFlow-INT8 authors. GPL-3.0; see LICENSE.
"""HyperFlow H3 Flash INT8 (convrot) on ComfyUI's native INT8 ops, plus a LoRA node for HyperFlow H3 Flash models;
reuses the official ComfyUI-HyperFlow-Flash pack."""

from .loader import NODE_CLASS_MAPPINGS as _LOADER_NODES
from .loader import NODE_DISPLAY_NAME_MAPPINGS as _LOADER_NAMES
from .lora import NODE_CLASS_MAPPINGS as _LORA_NODES
from .lora import NODE_DISPLAY_NAME_MAPPINGS as _LORA_NAMES
from .pih3 import NODE_CLASS_MAPPINGS as _PIH3_NODES
from .pih3 import NODE_DISPLAY_NAME_MAPPINGS as _PIH3_NAMES

NODE_CLASS_MAPPINGS = {**_LOADER_NODES, **_LORA_NODES, **_PIH3_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_LOADER_NAMES, **_LORA_NAMES, **_PIH3_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
