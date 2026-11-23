# Copyright (C) 2026 the ComfyUI-HyperFlow-INT8 authors. GPL-3.0; see LICENSE.
# load_module() and the DWD run-blocks wrapper are adapted from ComfyUI-piFlow (Copyright 2025-2026 Hansheng Chen,
# Apache-2.0); the DWD head module and its outer-sample wrapper are ComfyUI-piFlow's own, imported at run time.
"""pi-H3 Adapter for HyperFlow: a pi-H3 (Lakonik/pi-H3, LakonLab) adapter on a pruned HyperFlow H3 Flash model.

A pi-H3 adapter is not a plain LoRA. Next to rank-64 LoRA pairs on each block's attn.out_proj / mlp.fc1 / mlp.fc2 it
replaces every block's time modulation (adaln_proj, [96768, 8]) and the final layer, and a DWD adapter also replaces
the out_proj / fc1 / fc2 of the last four source blocks and adds the DWD pixel head. All of those consume stock
MiniMax H3's 8-D "curve form" time embedding (an adaln_t_table lerp). HyperFlow H3 Flash differs from stock H3 in four
ways: it drops source blocks (fl2va 6, 21, 22; ref2va 9) and renumbers the kept ones; some kept blocks store low-rank
pairs (LowRankLinear) or split q/k/v (SplitQKV); its blocks and final layer take a 16-D two-time embedding
te(t) + gate * (ee(r) - te(t)); and it ships no adaln_t_table.

On a clone of the model, with object patches, weight patches and forward injections only, this node:
* applies the LoRA pairs with the HyperFlow LoRA node's block remap and bypass hooks (same injection key, so the two
  nodes chain in either order), also on the low-rank linears;
* time path "pi-H3": turns the model's time embedding into the stock curve (table read from the stock H3 base file) and
  installs pi-H3's adaln_proj on every kept block and on the final layer (bias from the adapter, else the stock base's);
* time path "HyperFlow blocks": keeps HyperFlow's own two-time block modulation; the final layer and the DWD head get
  pi-H3's modulation on the stock curve, carried next to HyperFlow's embedding in one t_emb ([16-D | 8-D]);
* sets the final layer's norm / video_out / audio_out from the adapter;
* replaces the DWD blocks' out_proj / fc1 / fc2 with new linears holding the adapter's weights as stored;
* installs the DWD head from ComfyUI-piFlow, so its DWD Decode node reads the pixels.
On a stock 50-block H3 base (dropped blocks = none) with time path "pi-H3" it gives the same take as Load LakonLab
Adapter at the same seed (the LoRA runs in bypass mode instead of merged).
"""

from __future__ import annotations

import logging
import re
import sys
import weakref

import comfy.lora
import comfy.lora_convert
import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter
import folder_paths
import torch
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.bypass import BypassForwardHook
from safetensors import safe_open
from torch import nn

from .lora import INJECTION_KEY, KNOWN_DROPPED, factored_key_map, module_at, out_in, parse_dropped, remap_blocks

SOURCE_BLOCKS = 50
LINEARS = ("attn.out_proj", "mlp.fc1", "mlp.fc2")
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
TIME_PATHS = ["pi-H3 (stock H3 time curve)", "HyperFlow blocks + pi-H3 head"]
LORA_MODES = ["bypass (keeps the file's weights)", "merge into dense layers (faster)"]
DEFAULT_STOCK = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
HF_T_DIM = 16


# ----------------------------------------------------------------------------------------------- adapter keys

_DIFFUSERS_BLOCK = {
    "attn.to_out.0.": ("attn.out_proj.", False),
    "ff.net.0.proj.": ("mlp.fc1.", True),       # mlp.fc1 stores the two swiglu halves of ff.net.0.proj swapped
    "ff.net.2.": ("mlp.fc2.", False),
    "adaln_proj.linear.": ("adaln_proj.linear.", False),
}
_DIFFUSERS_TOP = {
    "norm_out.norm.": "final_layer.norm.",
    "norm_out.linear.": "final_layer.adaln_proj.linear.",
    "proj_out.": "final_layer.video_out.",
    "audio_proj_out.": "final_layer.audio_out.",
}


def normalize_adapter(sd: dict) -> dict:
    """Diffusers-named adapter tensors -> ComfyUI MiniMax H3 names (ComfyUI-named adapters pass through)."""
    out = {}
    for key, value in sd.items():
        if key.startswith(("dwd_", "blocks.", "final_layer.")):
            out[key] = value
            continue
        m = re.match(r"^transformer_blocks\.(\d+)\.(.*)$", key)
        if m is not None:
            rest = m.group(2)
            for src, (dst, swap) in _DIFFUSERS_BLOCK.items():
                if rest.startswith(src):
                    leaf = rest[len(src):]
                    if swap and leaf != "lora_A.weight":      # weight and lora_B rows follow the target's rows
                        value = comfy.utils.swap_scale_shift(value)
                    out[f"blocks.{m.group(1)}.{dst}{leaf}"] = value
                    break
            else:
                raise ValueError(f"pi-H3 adapter tensor not handled: {key}")
            continue
        for src, dst in _DIFFUSERS_TOP.items():
            if key.startswith(src):
                out[dst + key[len(src):]] = value
                break
        else:
            raise ValueError(f"pi-H3 adapter tensor not handled: {key}")
    return out


def split_adapter(sd: dict):
    lora, adaln, full, final, dwd = {}, {}, {}, {}, {}
    for key, value in sd.items():
        if key.startswith("dwd_"):
            dwd[key] = value
        elif key.startswith("final_layer."):
            final[key[len("final_layer."):]] = value
        elif key.endswith(LORA_SUFFIXES):
            lora[key] = value
        else:
            m = re.match(r"^blocks\.(\d+)\.(.*)$", key)
            if m is None:
                raise ValueError(f"pi-H3 adapter tensor not handled: {key}")
            block, rest = int(m.group(1)), m.group(2)
            if rest.startswith("adaln_proj.linear."):
                adaln.setdefault(block, {})[rest[len("adaln_proj.linear."):]] = value
            else:
                layer = next((name for name in LINEARS if rest.startswith(name + ".")), None)
                if layer is None:
                    raise ValueError(f"pi-H3 adapter tensor not handled: {key}")
                full.setdefault(block, {}).setdefault(layer, {})[rest[len(layer) + 1:]] = value
    return lora, adaln, full, final, dwd


# ----------------------------------------------------------------------------------------------- helpers

def _shape(weight) -> tuple[int, ...]:
    return tuple(int(s) for s in getattr(weight, "tensor_shape", weight.shape))


def load_module(module: nn.Module, sd: dict) -> None:
    """Adapter tensors into a fresh module as stored (dense dtype, or a ComfyUI-quantized weight)."""
    for name, m in module.named_modules():
        prefix = name + "." if name else ""
        if hasattr(m, "factory_kwargs") and prefix + "weight" in sd and prefix + "comfy_quant" not in sd:
            dtype = sd[prefix + "weight"].dtype
            m.factory_kwargs = dict(m.factory_kwargs, dtype=dtype)
            if getattr(m, "bias", None) is not None:
                m.bias = nn.Parameter(torch.empty_like(m.bias, dtype=dtype), requires_grad=False)
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"adapter tensors do not match the module: missing {missing}, unexpected {unexpected}")


def piflow_module(suffix: str, attr: str):
    """A module of the installed ComfyUI-piFlow pack (its DWD head, wrapper and pixel key)."""
    for name, mod in list(sys.modules.items()):
        if name.endswith(suffix) and hasattr(mod, attr):
            return mod
    raise RuntimeError("pi-H3 DWD adapters need the ComfyUI-piFlow pack (1.4.2 or newer) installed and loaded: "
                       "https://github.com/Lakonik/ComfyUI-piFlow")


def curve_lerp(table: torch.Tensor, unique_t, device) -> torch.Tensor:
    """Stock H3 curve-form time embedding: lerp of adaln_t_table rows (as MiniMaxH3Model._time_embedding)."""
    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    table = table.to(device)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


class CurveAdaln(nn.Module):
    """AdalnProj in curve form (no silu, fp32), as ComfyUI builds it for a curve-form checkpoint. The weights are a
    few MB per block; a copy on the compute device is kept so that no step pays a host-to-device transfer."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, hidden: int, expand: int, modalities: int):
        super().__init__()
        self.hidden, self.expand, self.modalities = hidden, expand, modalities
        self.register_buffer("weight", weight.float().contiguous(), persistent=False)
        self.register_buffer("bias", bias.float().contiguous(), persistent=False)
        self.__dict__["_on_device"] = None

    def forward(self, t_emb):
        w, b = self.weight, self.bias
        if w.device != t_emb.device:
            cached = self.__dict__["_on_device"]
            if cached is None or cached[0].device != t_emb.device:
                cached = (w.to(t_emb.device), b.to(t_emb.device))
                self.__dict__["_on_device"] = cached
            w, b = cached
        x = nn.functional.linear(t_emb.float(), w, b)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


class SliceT(nn.Module):
    """Feeds a slice of the combined t_emb to a time-modulated module, cast to the dtype it expects."""

    def __init__(self, inner: nn.Module, start: int, stop: int, state: dict, key: str | None = None):
        super().__init__()
        self.inner = inner
        self.start, self.stop = start, stop
        self.__dict__["state"] = state
        self.__dict__["dtype_key"] = key

    def forward(self, t_emb):
        t = t_emb[:, self.start:self.stop]
        if self.__dict__["dtype_key"] is not None:
            t = t.to(self.__dict__["state"][self.__dict__["dtype_key"]])
        return self.inner(t)


_NATIVE_ALIASES = {}


def native_name_alias(dm) -> bool:
    """Spark-H3 and ComfyUI-sol-attn accept only a model whose class is named MiniMaxH3Model. HyperFlow's loader turns
    the model into its own HyperFlowModel subclass; give it a subclass of that, named MiniMaxH3Model, with nothing
    overridden, so those nodes take it. Behaviour is HyperFlowModel's exactly."""
    cls = dm.__class__
    if cls.__name__ == "MiniMaxH3Model":
        return False
    alias = _NATIVE_ALIASES.get(cls)
    if alias is None:
        alias = type("MiniMaxH3Model", (cls,), {"__module__": cls.__module__, "__qualname__": "MiniMaxH3Model",
                                                 "__doc__": f"{cls.__name__} under the native class name"})
        _NATIVE_ALIASES[cls] = alias
    dm.__class__ = alias
    return True


# ----------------------------------------------------------------------------------------------- the patch

def apply_pih3(model, adapter_sd: dict, strength: float, time_path: str, stock_path: str, dropped: list[int] | None,
               lora_mode: str = LORA_MODES[0]):
    dm = model.model.diffusion_model
    kept = len(dm.blocks)
    if dropped is None:
        dropped = KNOWN_DROPPED.get((SOURCE_BLOCKS, kept))
        if dropped is None:
            raise ValueError(f"dropped_blocks=auto knows 47 (fl2va), 49 (ref2va) and 50 blocks, not {kept}")
    if SOURCE_BLOCKS - len(dropped) != kept:
        raise ValueError(f"{SOURCE_BLOCKS} source blocks minus dropped {dropped} != {kept} blocks in this model")
    kept_src = [s for s in range(SOURCE_BLOCKS) if s not in set(dropped)]
    to_kept = {s: j for j, s in enumerate(kept_src)}
    is_hyperflow = hasattr(dm, "endpoint_time_embedder")
    hf_path = time_path == TIME_PATHS[1]
    if hf_path and not is_hyperflow:
        raise ValueError("time path 'HyperFlow blocks' needs a HyperFlow H3 Flash model (two-time embedder)")

    lora, adaln, full, final, dwd_sd = split_adapter(normalize_adapter(adapter_sd))
    hidden = dm.hidden_size
    eps = dm.final_layer.norm.eps

    with safe_open(stock_path, "pt") as f:
        table = f.get_tensor("adaln_t_table").float()
        stock_bias = {s: f.get_tensor(f"blocks.{s}.adaln_proj.linear.bias") for s in kept_src
                      if s in adaln and "bias" not in adaln[s]}
    curve_dim = table.shape[1]

    patched = model.clone()
    state = {"dtype": torch.bfloat16}

    # --- time embedding: the pi-H3 curve alone, or [HyperFlow two-time | curve]
    orig_time = patched.get_model_object("diffusion_model._time_embedding")
    if hf_path:
        def time_embedding(unique_t, device, dtype):
            state["dtype"] = dtype
            return torch.cat([orig_time(unique_t, device, dtype).float(), curve_lerp(table, unique_t, device)], -1)
        curve_slice = (HF_T_DIM, HF_T_DIM + curve_dim)
    else:
        def time_embedding(unique_t, device, dtype):
            return curve_lerp(table, unique_t, device)
        curve_slice = None
    patched.add_object_patch("diffusion_model._time_embedding", time_embedding)

    # --- block time modulation
    n_adaln = 0
    for s in kept_src:
        j = to_kept[s]
        key = f"diffusion_model.blocks.{j}.adaln_proj"
        if hf_path:
            patched.add_object_patch(key, SliceT(patched.get_model_object(key), 0, HF_T_DIM, state, "dtype"))
            continue
        if s not in adaln or "weight" not in adaln[s]:
            raise ValueError(f"time path 'pi-H3' needs the adapter's adaln_proj for source block {s}")
        patched.add_object_patch(key, CurveAdaln(adaln[s]["weight"], adaln[s].get("bias", stock_bias.get(s)), hidden, 6, 3))
        n_adaln += 1

    # --- final layer: pi-H3's adaln on the curve, norm / heads as set patches
    if final:
        fa = CurveAdaln(final["adaln_proj.linear.weight"], final["adaln_proj.linear.bias"], hidden, 2, 1)
        if curve_slice is not None:
            fa = SliceT(fa, *curve_slice, state)
        patched.add_object_patch("diffusion_model.final_layer.adaln_proj", fa)
        sets = {}
        for leaf in ("norm.weight", "video_out.weight", "video_out.bias", "audio_out.weight", "audio_out.bias"):
            if leaf in final:
                sets["diffusion_model.final_layer." + leaf] = ("set", (final[leaf],))
        patched.add_patches(sets, 1.0)

    # --- full-replacement linears (the DWD adapter's last four source blocks)
    n_full = 0
    for s, layers in full.items():
        if s not in to_kept:
            logging.warning("pi-H3 HyperFlow: source block %d is dropped in this model; its full weights are left out", s)
            continue
        j = to_kept[s]
        for name, tensors in layers.items():
            out_f, in_f = _shape(tensors["weight"])
            old = module_at(dm.blocks[j], name)
            if out_in(old) != (out_f, in_f):
                raise ValueError(f"block {j} {name}: adapter {(out_f, in_f)} vs model {out_in(old)}")
            linear = comfy.ops.mixed_precision_ops().Linear(in_f, out_f, bias=False, device="cpu")
            load_module(linear, dict(tensors))
            patched.add_object_patch(f"diffusion_model.blocks.{j}.{name}", linear)
            n_full += 1

    # --- LoRA pairs, renumbered with the HyperFlow LoRA node's remap; bypass (or merged into dense layers)
    remapped, skipped = remap_blocks({"diffusion_model." + k: v for k, v in lora.items()}, dropped, SOURCE_BLOCKS)
    remapped = comfy.lora_convert.convert_lora(remapped)
    key_map = comfy.lora.model_lora_keys_unet(patched.model, {})
    key_map.update(factored_key_map(dm))
    loaded = comfy.lora.load_lora(remapped, key_map)
    expected = len(remapped) // 2
    merge_dense = lora_mode == LORA_MODES[1]
    hooks, merged, factored = [], {}, 0
    for key, adapter in loaded.items():
        if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            raise ValueError(f"pi-H3 HyperFlow: unexpected non-LoRA patch {key}")
        module = module_at(patched.model, key)
        up, down = adapter.weights[0], adapter.weights[1]
        if out_in(module) != (up.shape[0], down.shape[1]):
            raise ValueError(f"{key}: lora {(up.shape[0], down.shape[1])} vs module {out_in(module)}")
        dense = hasattr(module, "weight")
        factored += not dense
        if merge_dense and dense:
            merged[key] = adapter
        else:
            hooks.append(BypassForwardHook(module, adapter, multiplier=strength))
    if len(hooks) + len(merged) != expected:
        raise ValueError(f"pi-H3 HyperFlow: {len(hooks) + len(merged)} of {expected} LoRA pairs matched a layer")
    if merged:
        patched.add_patches(merged, strength)
    earlier = [h for inj in model.injections.get(INJECTION_KEY, []) for h in getattr(inj, "hyperflow_hooks", [])]
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

    # --- DWD head (ComfyUI-piFlow's module and wrapper), fed the curve part of t_emb
    if dwd_sd:
        arch = piflow_module(".modules.architectures.minimax_h3_dwd", "MiniMaxH3DWD")
        head_mod = piflow_module(".modules.architectures.dwd_head", "dwd_outer_sample_wrapper")
        dwd = arch.MiniMaxH3DWD(dwd_sd, eps, dtype=dwd_sd["dwd_norm_out.linear.weight"].dtype,
                                device=patched.offload_device, operations=comfy.ops.mixed_precision_ops())
        load_module(dwd, dwd_sd)
        head = comfy.model_patcher.CoreModelPatcher(dwd, load_device=patched.load_device,
                                                    offload_device=patched.offload_device)
        patched.set_additional_models("lakonlab_dwd", [head])
        run_blocks = patched.get_model_object("diffusion_model._run_blocks")
        dwd_ref = weakref.ref(dwd)
        pixels_key = head_mod.DWD_PIXELS_KEY

        def run_blocks_and_decode(h, t_emb, mod_segments, layout, transformer_options):
            h = run_blocks(h, t_emb, mod_segments, layout, transformer_options)
            holder = transformer_options.get(pixels_key)
            sigmas = transformer_options.get("sigmas")
            sample_sigmas = transformer_options.get("sample_sigmas")
            if holder is not None and transformer_options.get("cond_or_uncond") == [0] and torch.isclose(
                    sigmas.flatten()[0].float(), sample_sigmas[sample_sigmas > 0][-1].float().to(sigmas.device)):
                t = t_emb if curve_slice is None else t_emb[:, curve_slice[0]:curve_slice[1]]
                holder.pixels = dwd_ref().decode(h, t, mod_segments, layout)
            return h

        patched.add_object_patch("diffusion_model._run_blocks", run_blocks_and_decode)
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "lakonlab_dwd",
                                     head_mod.dwd_outer_sample_wrapper)

    aliased = native_name_alias(dm) if is_hyperflow else False
    logging.info(
        "pi-H3 HyperFlow: %s, %d -> %d blocks (dropped %s), %d LoRA pairs in bypass + %d merged (%d on low-rank "
        "linears), %d curve adaln, %d full linears, final layer %s, DWD head %s, LoRA left out for source blocks %s%s",
        "pi-H3 time path" if not hf_path else "HyperFlow block modulation", SOURCE_BLOCKS, kept, dropped, len(hooks),
        len(merged), factored, n_adaln, n_full, "yes" if final else "no", "yes" if dwd_sd else "no", skipped,
        ", model class named MiniMaxH3Model for Spark-H3 / sol-attn" if aliased else "")
    return patched


class PiH3HyperFlowAdapter:
    TITLE = "pi-H3 Adapter for HyperFlow (block remap)"
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    DESCRIPTION = ("Loads a pi-H3 (LakonLab) adapter onto a pruned HyperFlow H3 Flash model: LoRA blocks renumbered past "
                   "the dropped ones and applied in bypass mode, pi-H3's time modulation, final layer, DWD blocks and "
                   "DWD head installed. Use it where Load LakonLab Adapter sits in the pi-H3 workflows.")

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models")
        return {
            "required": {
                "model": ("MODEL",),
                "adapter_name": (folder_paths.get_filename_list("loras"),),
                "adapter_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "time_path": (TIME_PATHS, {"default": TIME_PATHS[0],
                                           "tooltip": "pi-H3: the adapter's own time modulation on the stock H3 time "
                                                      "curve, as pi-H3 was trained. HyperFlow blocks + pi-H3 head: "
                                                      "HyperFlow's two-time block modulation with pi-H3's final layer; "
                                                      "in my tests its audio came out with a loud noise bed and "
                                                      "clipping on a quiet dialogue scene."}),
                "stock_base": (models, {"default": DEFAULT_STOCK if DEFAULT_STOCK in models else None,
                                        "tooltip": "Stock MiniMax H3 base (curve form) for adaln_t_table and the adaln "
                                                   "biases the adapter does not ship. Only those tensors are read, a "
                                                   "few MB."}),
            },
            "optional": {
                "lora_mode": (LORA_MODES, {"default": LORA_MODES[0],
                                           "tooltip": "bypass: LoRA added at run time, the file's quantized weights "
                                                      "stay as they are. merge: LoRA merged into the dense (quantized) "
                                                      "layers like Load LakonLab Adapter does; about 0.7s faster, but "
                                                      "ComfyUI re-quantizes each merged layer with fresh scales, which "
                                                      "on an NVFP4 file throws away its calibrated block scales (a "
                                                      "clearly different take). Low-rank layers always use bypass."}),
                "dropped_blocks": ("STRING", {"default": "auto",
                                              "tooltip": "auto: 6, 21, 22 for 47 blocks, 9 for 49, none for 50."}),
            },
        }

    def load(self, model, adapter_name, adapter_strength, time_path, stock_base, lora_mode=LORA_MODES[0],
             dropped_blocks="auto"):
        adapter_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", adapter_name),
                                                 safe_load=True)
        stock_path = folder_paths.get_full_path_or_raise("diffusion_models", stock_base)
        text = dropped_blocks.strip().lower()
        dropped = [] if text == "none" else parse_dropped(dropped_blocks)
        return (apply_pih3(model, adapter_sd, adapter_strength, time_path, stock_path, dropped, lora_mode),)


NODE_CLASS_MAPPINGS = {"PiH3HyperFlowAdapter": PiH3HyperFlowAdapter}
NODE_DISPLAY_NAME_MAPPINGS = {"PiH3HyperFlowAdapter": PiH3HyperFlowAdapter.TITLE}
