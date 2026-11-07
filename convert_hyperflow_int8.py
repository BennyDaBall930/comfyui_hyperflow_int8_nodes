# Copyright (C) 2026 the ComfyUI-HyperFlow-INT8 authors. GPL-3.0; see LICENSE.
"""Convert a HyperFlow H3 Flash INT8 checkpoint (diffusers layout, int8_convrot) into one ComfyUI safetensors file.

The INT8 weights are copied bit for bit. HyperFlow's scheme and ComfyUI's `int8_tensorwise` + `convrot` are the same
W8A8 math: a block-diagonal regular Hadamard (kron of H4 / 2, group 256) on the input dimension, int8 codes with one
fp32 scale per output channel (ComfyUI stores it as [N, 1]), and per-token dynamic int8 activations. Nothing is
requantized.

What changes on the way:
* diffusers names -> ComfyUI MiniMax-H3 names (the names the official HyperFlow GGUF uses).
* The dropped blocks are left out and the kept blocks renumbered 0..K-1, so ComfyUI builds K blocks.
* to_q / to_k / to_v that are all INT8 (or all dense) are fused into attn.qkv_proj (row concat: exact, the scales are
  per output row and the input rotation is shared).
* ff.net.0.proj (value | gate) becomes mlp.fc1 (gate | value): the two output halves swap, rows and scales alike.
* Low-rank factors stay bf16 as `<name>.down.weight` / `<name>.up.weight`, zero-padded to a rank multiple (exact).
* hyperflow_gate and hyperflow_sigmas (the 6-step schedule) are stored as tensors, as in the official GGUF.

Usage:
  python convert_hyperflow_int8.py SRC_DIR OUT.safetensors [--rank-multiple 8] [--source-revision SHA]
SRC_DIR is a downloaded `fl2va_int8_convrot/` or `ref2va_int8_convrot/` folder of VideoRebirthAI/hyperflow-h3-flash
(config.json, the index and the shards). Needs only torch and safetensors; runs on CPU, streams one tensor at a time.

The output is a modified version of a MiniMax H3 Model Derivative: its metadata carries the MiniMax notice and a
statement that the file was modified (MiniMax H3 Community License Agreement, Section III.2 and III.4).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open

CONVERTER = "ComfyUI-HyperFlow-INT8 convert 1.0"
MINIMAX_NOTICE = (
    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. "
    "All Rights Reserved."
)
MODIFIED_NOTICE = (
    "Modified file: converted from {repo} {folder} (revision {revision}) to the ComfyUI MiniMax-H3 tensor layout by "
    "convert_hyperflow_int8.py (ComfyUI-HyperFlow-INT8). Tensors renamed, dropped blocks removed and blocks renumbered, "
    "INT8 q/k/v concatenated, feed-forward fc1 halves swapped, low-rank factors zero-padded, gate and 6-step sigmas "
    "stored as tensors; INT8 codes and scales unchanged. Not an official MiniMax or Video Rebirth release."
)
DTYPE_NAMES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int8: "I8",
    torch.uint8: "U8",
}
FROM_ST = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I8": torch.int8, "U8": torch.uint8}

# Top-level tensors: diffusers prefix -> ComfyUI prefix. Values are copied unchanged (checked against the GGUF).
TOP = {
    "proj_in.": "video_patch_proj.",
    "audio_proj_in.": "audio_patch_proj.",
    "context_embedder.": "condition_proj.",
    "time_embedder.linear_1.": "time_embedder.proj_in.",
    "time_embedder.linear_2.": "time_embedder.proj_out.",
    "endpoint_time_embedder.linear_1.": "endpoint_time_embedder.proj_in.",
    "endpoint_time_embedder.linear_2.": "endpoint_time_embedder.proj_out.",
    "norm_out.linear.": "final_layer.adaln_proj.linear.",
    "norm_out.norm.": "final_layer.norm.",
    "proj_out.": "final_layer.video_out.",
    "audio_proj_out.": "final_layer.audio_out.",
    "token_refiner.final_norm.": "token_refiner.final_norm.",
}
# Per-block small tensors.
BLOCK_COPY = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
}
QKV = (("attn.to_q", "attn.q_proj"), ("attn.to_k", "attn.k_proj"), ("attn.to_v", "attn.v_proj"))
OTHER_LINEARS = (("attn.to_out.0", "attn.out_proj"), ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2"))
SWAP_HALVES = {"ff.net.0.proj"}

# The official HyperFlow schedule (ComfyUI-HyperFlow-Flash loader_nodes.step_sigmas): raw knots through shift 12.
RAW_SIGMAS_6 = (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0)
VIDEO_SHIFT = 12.0


class Source:
    """The checkpoint's shards: header shapes and dtypes, lazy tensor reads, and which keys were consumed."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        index = json.load(open(os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")))
        self.where = index["weight_map"]
        self.info: dict[str, tuple[str, list[int]]] = {}
        for shard in sorted(set(self.where.values())):
            with open(os.path.join(folder, shard), "rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            header.pop("__metadata__", None)
            for name, entry in header.items():
                self.info[name] = (entry["dtype"], entry["shape"])
        self.handles: dict[str, object] = {}
        self.used: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def shape(self, name: str) -> list[int]:
        return list(self.info[name][1])

    def dtype(self, name: str) -> torch.dtype:
        return FROM_ST[self.info[name][0]]

    def get(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.folder, shard), "pt", device="cpu")
        self.used.add(name)
        return self.handles[shard].get_tensor(name)

    def mark(self, name: str) -> None:
        self.used.add(name)


def _swap_rows(t: torch.Tensor) -> torch.Tensor:
    n = t.shape[0] // 2
    return torch.cat([t[n:], t[:n]])


def _pad_rank(r: int, multiple: int) -> int:
    return -(-r // multiple) * multiple if multiple > 0 else r


class Plan:
    """Output tensors in file order: (name, dtype, shape, producer)."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, torch.dtype, list[int], object]] = []

    def add(self, name, dtype, shape, fn) -> None:
        self.entries.append((name, dtype, [int(s) for s in shape], fn))


def linear_kind(src: Source, name: str) -> str:
    if f"{name}.qweight" in src:
        return "int8"
    if f"{name}.down.weight" in src:
        return "lowrank"
    if f"{name}.weight" in src:
        return "dense"
    raise KeyError(f"no weights for linear {name}")


def plan_linear(plan, src, name, out_name, quant_json, swap, multiple):
    """One linear (int8, low-rank or dense bf16) under its ComfyUI name."""
    kind = linear_kind(src, name)
    if kind == "int8":
        q, s = f"{name}.qweight", f"{name}.wscale"
        n_out = src.shape(q)[0]
        plan.add(f"{out_name}.weight", torch.int8, src.shape(q),
                 lambda: (_swap_rows if swap else (lambda t: t))(src.get(q)))
        plan.add(f"{out_name}.weight_scale", torch.float32, [n_out, 1],
                 lambda: (_swap_rows if swap else (lambda t: t))(src.get(s).float()).reshape(-1, 1))
        plan.add(f"{out_name}.comfy_quant", torch.uint8, [len(quant_json)],
                 lambda: torch.tensor(list(quant_json), dtype=torch.uint8))
    elif kind == "lowrank":
        d, u = f"{name}.down.weight", f"{name}.up.weight"
        rank, n_in = src.shape(d)
        n_out = src.shape(u)[0]
        padded = _pad_rank(rank, multiple)

        def down():
            t = src.get(d)
            return torch.cat([t, t.new_zeros(padded - rank, n_in)]) if padded > rank else t

        def up():
            t = src.get(u)
            t = _swap_rows(t) if swap else t
            return torch.cat([t, t.new_zeros(n_out, padded - rank)], dim=1) if padded > rank else t

        plan.add(f"{out_name}.down.weight", src.dtype(d), [padded, n_in], down)
        plan.add(f"{out_name}.up.weight", src.dtype(u), [n_out, padded], up)
    else:
        w = f"{name}.weight"
        plan.add(f"{out_name}.weight", src.dtype(w), src.shape(w),
                 lambda: (_swap_rows if swap else (lambda t: t))(src.get(w)))
    return kind


def plan_qkv(plan, src, prefix, out_prefix, quant_json, multiple):
    names = [f"{prefix}.{a}" for a, _ in QKV]
    kinds = [linear_kind(src, n) for n in names]
    if kinds == ["int8"] * 3:
        qs = [f"{n}.qweight" for n in names]
        ss = [f"{n}.wscale" for n in names]
        n_out = sum(src.shape(q)[0] for q in qs)
        plan.add(f"{out_prefix}.attn.qkv_proj.weight", torch.int8, [n_out, src.shape(qs[0])[1]],
                 lambda: torch.cat([src.get(q) for q in qs]))
        plan.add(f"{out_prefix}.attn.qkv_proj.weight_scale", torch.float32, [n_out, 1],
                 lambda: torch.cat([src.get(s).float() for s in ss]).reshape(-1, 1))
        plan.add(f"{out_prefix}.attn.qkv_proj.comfy_quant", torch.uint8, [len(quant_json)],
                 lambda: torch.tensor(list(quant_json), dtype=torch.uint8))
        return "fused-int8"
    if kinds == ["dense"] * 3:
        ws = [f"{n}.weight" for n in names]
        n_out = sum(src.shape(w)[0] for w in ws)
        plan.add(f"{out_prefix}.attn.qkv_proj.weight", src.dtype(ws[0]), [n_out, src.shape(ws[0])[1]],
                 lambda: torch.cat([src.get(w) for w in ws]))
        return "fused-dense"
    for (a, b) in QKV:
        plan_linear(plan, src, f"{prefix}.{a}", f"{out_prefix}.{b}", quant_json, False, multiple)
    return "split:" + "/".join(kinds)


KNOWN_VARIANTS = {(6, 21, 22): "fl2va", (9,): "ref2va"}


def detect_variant(folder: str, config: dict) -> str:
    """fl2va (t2v / i2v / flf2v) or ref2va (reference to video), from the folder name or the dropped blocks."""
    name = os.path.basename(os.path.normpath(folder)).lower()
    for variant in ("ref2va", "fl2va"):
        if name.startswith(variant):
            return variant
    return KNOWN_VARIANTS.get(tuple(sorted(int(i) for i in config.get("dropped_blocks") or [])), "unknown")


def build_plan(src: Source, config: dict, multiple: int, variant: str = "unknown",
               repo: str = "VideoRebirthAI/hyperflow-h3-flash", revision: str = "unknown"):
    plan = Plan()
    quant = config.get("quantization") or {}
    if quant.get("method") != "int8_convrot":
        raise ValueError(f"expected quantization.method int8_convrot, got {quant.get('method')!r}")
    group = int(quant["group"])
    quant_json = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}).encode()
    dropped = set(int(i) for i in config.get("dropped_blocks") or [])
    kept = [i for i in range(int(config["num_layers"])) if i not in dropped]

    for a, b in TOP.items():
        for leaf in ("weight", "bias"):
            if a + leaf in src:
                key = a + leaf
                plan.add(b + leaf, src.dtype(key), src.shape(key), lambda key=key: src.get(key))

    # rope.inv_freq is a buffer in ComfyUI (diffusers computes it): 1 / theta^(2k / 2d), k = 0..d-1, in fp32 math,
    # which is bit-identical to the official GGUF and to Comfy-Org's MiniMax-H3 checkpoints.
    dim = int(config["rope_freq_dim"])
    theta = float(config["rope_theta"])
    plan.add("rope.inv_freq", torch.float32, [dim],
             lambda: 1.0 / (theta ** (torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim))))
    plan.add("hyperflow_gate", torch.float32, [1],
             lambda: torch.tensor([float(config["hyperflow_gate"])], dtype=torch.float32))
    raw = torch.tensor(RAW_SIGMAS_6, dtype=torch.float64)
    plan.add("hyperflow_sigmas", torch.float32, [len(RAW_SIGMAS_6)],
             lambda: (VIDEO_SHIFT * raw / (1 + (VIDEO_SHIFT - 1) * raw)).float())

    refiner = sorted({int(k.split(".")[2]) for k in src.where if k.startswith("token_refiner.refiner_blocks.")})
    for i in refiner:
        p, o = f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}"
        for a, b in BLOCK_COPY.items():
            if f"{p}.{a}" in src:
                key = f"{p}.{a}"
                plan.add(f"{o}.{b}", src.dtype(key), src.shape(key), lambda key=key: src.get(key))
        plan_qkv(plan, src, p, o, quant_json, multiple)
        for a, b in OTHER_LINEARS:
            plan_linear(plan, src, f"{p}.{a}", f"{o}.{b}", quant_json, a in SWAP_HALVES, multiple)

    layout = {}
    for new, old in enumerate(kept):
        p, o = f"transformer_blocks.{old}", f"blocks.{new}"
        for a, b in BLOCK_COPY.items():
            key = f"{p}.{a}"
            plan.add(f"{o}.{b}", src.dtype(key), src.shape(key), lambda key=key: src.get(key))
        kinds = [plan_qkv(plan, src, p, o, quant_json, multiple)]
        for a, b in OTHER_LINEARS:
            kinds.append(plan_linear(plan, src, f"{p}.{a}", f"{o}.{b}", quant_json, a in SWAP_HALVES, multiple))
        layout[new] = {"source_block": old, "qkv": kinds[0], "out_proj": kinds[1], "fc1": kinds[2], "fc2": kinds[3]}

    folder = os.path.basename(os.path.normpath(src.folder))
    metadata = {
        "format": "pt",
        "notice": MINIMAX_NOTICE,
        "modified": MODIFIED_NOTICE.format(repo=repo, folder=f"{folder}/", revision=revision),
        "license": "MiniMax H3 Community License Agreement (see LICENSE and NOTICE distributed with this file)",
        "hyperflow_int8": json.dumps({
            "converter": CONVERTER,
            "variant": variant,
            "source_repo": repo,
            "source_revision": revision,
            "source": folder,
            "source_class": config.get("_class_name"),
            "dropped_blocks": sorted(dropped),
            "kept_blocks": kept,
            "quantization": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group,
                             "int8_linears_source": len(quant["linears"])},
            "lowrank_rank_multiple": multiple,
            "fc1_order": "gate|value (diffusers ff.net.0.proj is value|gate)",
            "hyperflow_gate": float(config["hyperflow_gate"]),
            "layout": layout,
        }),
    }
    return plan, metadata, kept


def write(plan: Plan, metadata: dict, out_path: str) -> int:
    header = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, _ in plan.entries:
        size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        header[name] = {"dtype": DTYPE_NAMES[dtype], "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = out_path + ".partial"
    started = time.time()
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for i, (name, dtype, shape, fn) in enumerate(plan.entries):
            t = fn()
            if t.dtype != dtype or list(t.shape) != shape:
                raise RuntimeError(f"{name}: produced {t.dtype} {list(t.shape)}, planned {dtype} {shape}")
            f.write(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
            if i % 100 == 0:
                print(f"  [{i + 1}/{len(plan.entries)}] {name}  {time.time() - started:.0f}s", flush=True)
    os.replace(tmp, out_path)
    return 8 + len(blob) + offset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="HyperFlow INT8 checkpoint folder (config.json + index + shards)")
    parser.add_argument("out", help="output .safetensors path")
    parser.add_argument("--rank-multiple", type=int, default=8,
                        help="zero-pad low-rank ranks to this multiple (HyperFlow's runtime default is 8; 0 = none)")
    parser.add_argument("--variant", choices=("auto", "fl2va", "ref2va"), default="auto",
                        help="recorded in the file's metadata; auto reads the folder name or the dropped blocks")
    parser.add_argument("--source-repo", default="VideoRebirthAI/hyperflow-h3-flash",
                        help="recorded in the modification notice")
    parser.add_argument("--source-revision", default="unknown",
                        help="HF revision (commit) the source folder was downloaded at; recorded in the notice")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        print(f"output folder {out_dir} does not exist", file=sys.stderr)
        return 2
    if os.path.exists(args.out) and not args.overwrite:
        print(f"{args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    config = json.load(open(os.path.join(args.src, "config.json")))
    if config.get("_class_name") != "HyperFlowH3FlashTransformer3DModel":
        print(f"warning: config class is {config.get('_class_name')!r}, not HyperFlowH3FlashTransformer3DModel",
              file=sys.stderr)
    variant = detect_variant(args.src, config) if args.variant == "auto" else args.variant
    src = Source(args.src)
    plan, metadata, kept = build_plan(src, config, args.rank_multiple, variant, args.source_repo, args.source_revision)
    print(f"{variant}: {len(plan.entries)} tensors, {len(kept)} blocks kept (dropped {config.get('dropped_blocks')})")
    size = write(plan, metadata, args.out)
    missing = sorted(set(src.where) - src.used)
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoint tensors were not converted: {missing[:8]}")
    print(f"wrote {args.out}: {size / 1e9:.2f} GB; all {len(src.where)} checkpoint tensors consumed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
