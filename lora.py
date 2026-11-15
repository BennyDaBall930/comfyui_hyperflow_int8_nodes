# Copyright (C) 2026 the ComfyUI-HyperFlow-INT8 authors. GPL-3.0; see LICENSE.
"""HyperFlow LoRA (bypass): a LoRA trained on the full MiniMax-H3 DiT, applied to a pruned HyperFlow H3 Flash model.

HyperFlow drops source blocks (6, 21 and 22 in fl2va: config.json `dropped_blocks`) and renumbers the kept ones, and
some kept blocks replace their dense linears with low-rank pairs (`LowRankLinear`) and split q/k/v (`SplitQKV`).
A stock LoRA loader therefore patches block N of the LoRA onto the wrong block from the first dropped index on, and
skips every factored linear (no `.weight` on the wrapper module).

This node:
* renumbers the LoRA's `blocks.<source>` keys to the kept order and leaves out the dropped blocks' deltas;
* applies every adapter in ComfyUI's bypass mode (`comfy.weight_adapter.bypass.BypassForwardHook`, unchanged):
  y = f(x) + scale * up(down(x)). On a dense or INT8 linear this is the stock LoraLoaderBypass math. On a factored
  linear f(x) = U(D(x)), so y = (U·D + scale·B·A) x: the LoRA delta added to the weight HyperFlow kept, the same
  thing a weight merge would give. SplitQKV concatenates q|k|v and FactorizedMLP chunks gate|value, which are the
  row orders of the dense `qkv_proj` / `fc1` the LoRA was trained on.
"""

from __future__ import annotations

import logging
import re

import comfy.lora
import comfy.lora_convert
import comfy.utils
import comfy.weight_adapter
import folder_paths
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.bypass import BypassForwardHook

# Both key styles a MiniMax-H3 LoRA ships in: `diffusion_model.blocks.6.attn...` and kohya `lora_unet_blocks_6_attn...`.
BLOCK = re.compile(r"^(?P<prefix>(?:diffusion_model\.|lora_unet_)?)blocks(?P<sep>[._])(?P<index>\d+)(?P=sep)")
LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
INJECTION_KEY = "hyperflow_bypass_lora"
# Source blocks the released HyperFlow H3 Flash variants leave out, by the number of blocks they keep (of 50):
# fl2va keeps 47, ref2va keeps 49 (`dropped_blocks` in each variant's config.json and safetensors header).
KNOWN_DROPPED = {(50, 47): [6, 21, 22], (50, 49): [9], (50, 50): []}


def parse_dropped(text: str) -> list[int] | None:
    """A list such as "6, 21, 22"; "auto" (or empty) -> None: looked up from the model's block count."""
    if text.strip().lower() in ("", "auto"):
        return None
    return sorted({int(part) for part in re.split(r"[,\s]+", text.strip()) if part})


def remap_blocks(lora_sd: dict, dropped: list[int], source_blocks: int) -> tuple[dict, list[int]]:
    """`blocks.<source>.` -> `blocks.<kept index>.`; the dropped blocks' tensors are left out."""
    kept = [index for index in range(source_blocks) if index not in set(dropped)]
    new_index = {old: new for new, old in enumerate(kept)}
    out, skipped = {}, set()
    for key, value in lora_sd.items():
        match = BLOCK.match(key)
        if match is None:
            out[key] = value
            continue
        old = int(match.group("index"))
        if old in new_index:
            sep = match.group("sep")
            out[f"{match.group('prefix')}blocks{sep}{new_index[old]}{sep}{key[match.end():]}"] = value
        else:
            skipped.add(old)
    return out, sorted(skipped)


def _shape(weight) -> tuple[int, ...]:
    return tuple(int(s) for s in getattr(weight, "tensor_shape", weight.shape))


def out_in(module) -> tuple[int, int] | None:
    """(out_features, in_features) of a dense, quantized, low-rank or split-qkv linear."""
    if hasattr(module, "q_proj"):  # SplitQKV
        parts = [out_in(getattr(module, name)) for name in ("q_proj", "k_proj", "v_proj")]
        if any(part is None for part in parts):
            return None
        return sum(o for o, _ in parts), parts[0][1]
    if hasattr(module, "down") and hasattr(module, "up"):  # LowRankLinear
        return _shape(module.up.weight)[0], _shape(module.down.weight)[1]
    weight = getattr(module, "weight", None)
    return None if weight is None else _shape(weight)[:2]


def factored_key_map(diffusion_model) -> dict[str, str]:
    """LoRA prefixes for the factored linears, which have no `.weight` in the model state dict."""
    key_map = {}
    for index, block in enumerate(diffusion_model.blocks):
        for name in LINEARS:
            owner, _, leaf = name.partition(".")
            module = getattr(getattr(block, owner), leaf)
            if hasattr(module, "weight"):
                continue
            model_key = f"diffusion_model.blocks.{index}.{name}.weight"
            key_map[f"diffusion_model.blocks.{index}.{name}"] = model_key
            key_map[f"blocks.{index}.{name}"] = model_key
            key_map[f"lora_unet_blocks_{index}_{name.replace('.', '_')}"] = model_key
    return key_map


def module_at(root, key: str):
    module = root
    for part in key.removesuffix(".weight").split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def apply_hyperflow_lora(model, lora_sd: dict, strength: float, dropped: list[int] | None, source_blocks: int):
    diffusion_model = model.model.diffusion_model
    kept_blocks = len(diffusion_model.blocks)
    if dropped is None:
        dropped = KNOWN_DROPPED.get((source_blocks, kept_blocks))
        if dropped is None:
            raise ValueError(
                f"dropped_blocks=auto knows the released fl2va (47 blocks) and ref2va (49 blocks) models, not one "
                f"with {kept_blocks} of {source_blocks} blocks: type the dropped source blocks, e.g. \"6, 21, 22\""
            )
    if source_blocks - len(dropped) != kept_blocks:
        raise ValueError(
            f"{source_blocks} source blocks minus dropped {dropped} != {kept_blocks} blocks in this model"
        )
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    lora_sd, skipped = remap_blocks(lora_sd, dropped, source_blocks)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    key_map.update(factored_key_map(diffusion_model))
    loaded = comfy.lora.load_lora(lora_sd, key_map)

    patched = model.clone()
    hooks, regular, mismatched, factored = [], {}, [], 0
    for key, adapter in loaded.items():
        if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            regular[key] = adapter
            continue
        module = module_at(patched.model, key)
        expected = out_in(module)
        up, down = adapter.weights[0], adapter.weights[1]
        if expected is not None and (up.shape[0], down.shape[1]) != expected:
            mismatched.append(f"{key}: lora {(up.shape[0], down.shape[1])} vs module {expected}")
            continue
        factored += not hasattr(module, "weight")
        hooks.append(BypassForwardHook(module, adapter, multiplier=strength))
    if mismatched:
        raise ValueError("HyperFlow LoRA shape mismatch:\n" + "\n".join(mismatched[:10]))
    if regular:
        patched.add_patches(regular, strength)

    # A model that already went through this node keeps those hooks: one injection carries all of them, injected in
    # node order and ejected in reverse, so every hook puts back the forward it replaced (several LoRAs can be chained).
    earlier = [hook for inj in model.injections.get(INJECTION_KEY, []) for hook in getattr(inj, "hyperflow_hooks", [])]
    stack = earlier + hooks

    def inject_all(_patcher):
        for hook in stack:
            hook.inject()

    def eject_all(_patcher):
        for hook in reversed(stack):
            hook.eject()

    injection = PatcherInjection(inject=inject_all, eject=eject_all)
    injection.hyperflow_hooks = stack
    patched.set_injections(INJECTION_KEY, [injection])
    logging.info(
        "HyperFlow LoRA: %d bypass adapters (%d on factored linears), %d regular patches, "
        "source blocks %s left out, %d -> %d blocks",
        len(hooks), factored, len(regular), skipped, source_blocks, kept_blocks,
    )
    return patched, hooks, factored, skipped


class HyperFlowLoraBypass:
    TITLE = "HyperFlow LoRA (bypass, block remap)"
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    DESCRIPTION = (
        "Applies a LoRA trained on the full 50-block MiniMax-H3 to a pruned HyperFlow model: renumbers the LoRA "
        "blocks past the dropped ones and adds the LoRA in bypass mode, also on low-rank / split-qkv linears."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "dropped_blocks": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "Source blocks the HyperFlow model leaves out. auto: 6, 21, 22 for fl2va, 9 for "
                        "ref2va, picked by the model's block count. Or type the list.",
                    },
                ),
                "source_blocks": ("INT", {"default": 50, "min": 1, "max": 512}),
            },
        }

    def load(self, model, lora_name, strength_model, dropped_blocks="auto", source_blocks=50):
        if strength_model == 0:
            return (model,)
        lora_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        patched, *_ = apply_hyperflow_lora(model, lora_sd, strength_model, parse_dropped(dropped_blocks), source_blocks)
        return (patched,)


NODE_CLASS_MAPPINGS = {"HyperFlowLoraBypass": HyperFlowLoraBypass}
NODE_DISPLAY_NAME_MAPPINGS = {"HyperFlowLoraBypass": HyperFlowLoraBypass.TITLE}
