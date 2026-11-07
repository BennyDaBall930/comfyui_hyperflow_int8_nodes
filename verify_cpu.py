# Copyright (C) 2026 the ComfyUI-HyperFlow-INT8 authors. GPL-3.0; see LICENSE.
"""CPU-only checks of a convert_hyperflow_int8.py output. Never touches the GPU (CUDA is hidden before torch loads).

  python verify_cpu.py layout   CONVERTED SRC_DIR GGUF        every tensor vs the official HyperFlow GGUF layout
  python verify_cpu.py exact    CONVERTED SRC_DIR INT8_PY     INT8 dequant + low-rank factors vs HyperFlow's int8.py
  python verify_cpu.py loader   CONVERTED SRC_DIR INT8_PY MINI  a 4-block slice through the real loader, module
                                                                forwards vs the reference math, smoke forward
  python verify_cpu.py import                                   both packs through ComfyUI's custom-node loader
Run from the ComfyUI folder with ComfyUI's python.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 8))
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


def regular_hadamard(group: int) -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64) / 2
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < group:
        h = torch.kron(h, h4)
    return h


def corr(a, b) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


class HF:
    def __init__(self, folder):
        self.folder = folder
        self.where = json.load(open(os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        self.handles = {}
        self.config = json.load(open(os.path.join(folder, "config.json")))

    def __contains__(self, k):
        return k in self.where

    def __getitem__(self, k):
        f = self.where[k]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.folder, f), "pt", device="cpu")
        return self.handles[f].get_tensor(k)


def kept_blocks(config):
    dropped = set(config.get("dropped_blocks") or [])
    return [i for i in range(config["num_layers"]) if i not in dropped]


def ours_dense(f, name, rows=None):
    """fp32 dense weight of a converted linear: INT8 (dequant + un-rotate, ComfyUI's math) or bf16."""
    w = f.get_slice(f"{name}.weight")
    w = w[rows] if rows is not None else w[:]
    keys = set(f.keys())
    if f"{name}.weight_scale" in keys:
        s = f.get_slice(f"{name}.weight_scale")
        s = s[rows] if rows is not None else s[:]
        conf = json.loads(bytes(f.get_tensor(f"{name}.comfy_quant").tolist()).decode())
        g = conf["convrot_groupsize"]
        h = regular_hadamard(g).float()  # ComfyUI: kron(H4)/sqrt(g), symmetric; W = (q * s) @ H^T per group
        w = ((w.float() * s.float()).reshape(w.shape[0], -1, g) @ h.T).reshape(w.shape)
        return w, "int8", conf
    return w.float(), "dense", None


# --------------------------------------------------------------------------------------------------- layout
def check_layout(converted, src_dir, gguf_path):
    import gguf
    import numpy as np
    from gguf.quants import dequantize

    reader = gguf.GGUFReader(gguf_path)
    gt = {t.name: t for t in reader.tensors}

    def gg(name, rows=None):
        t = gt[name]
        shape = [int(s) for s in reversed(t.shape)]
        data = t.data
        if rows is not None:
            data = data.reshape(shape[0], -1)[rows] if len(shape) == 2 else data
            shape = [data.shape[0]] + shape[1:]
        return torch.from_numpy(np.array(dequantize(np.ascontiguousarray(data), t.tensor_type), dtype=np.float32)).reshape(shape)

    f = safe_open(converted, "pt", device="cpu")
    keys = set(f.keys())
    exact_bad, small, corr_min, n_lin = [], 0, 1.0, 0
    report = []
    # Every non-linear tensor must equal the GGUF bit for bit (both store them unquantized).
    for k in sorted(keys):
        if k.endswith(("weight_scale", "comfy_quant", "down.weight", "up.weight")):
            continue
        if any(k.endswith(f"{lin}.weight") for lin in ("qkv_proj", "out_proj", "fc1", "fc2", "q_proj", "k_proj", "v_proj")):
            continue
        if k not in gt:
            exact_bad.append((k, "absent in GGUF"))
            continue
        a = f.get_tensor(k).float()
        b = gg(k)
        small += 1
        if tuple(a.shape) != tuple(b.shape) or float((a - b).abs().max()) != 0.0:
            exact_bad.append((k, float((a - b).abs().max()) if a.shape == b.shape else "shape"))
    # Every linear: same kind (dense vs factorized), same shape, and the same values up to quantization noise.
    linears = sorted({k.rsplit(".", 1)[0] for k in keys if k.endswith(("qkv_proj.weight", "out_proj.weight", "fc1.weight",
                                                                         "fc2.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight"))})
    lowrank = sorted({k[: -len(".down.weight")] for k in keys if k.endswith(".down.weight")})
    for name in linears:
        out = f.get_slice(f"{name}.weight").get_shape()[0]
        rows = slice(0, out, 11)
        a, kind, _ = ours_dense(f, name, rows)
        if f"{name}.weight" in gt:
            b = gg(f"{name}.weight", rows)
        else:  # split q/k/v part vs the GGUF's fused qkv_proj
            block = name.rsplit(".attn.", 1)[0]
            i = ("q_proj", "k_proj", "v_proj").index(name.rsplit(".", 1)[1])
            qkv = gt[f"{block}.attn.qkv_proj.weight"]
            n = int(qkv.shape[1]) // 3
            b = gg(f"{block}.attn.qkv_proj.weight", slice(i * n, (i + 1) * n, 11))
        c = corr(a, b)
        n_lin += 1
        corr_min = min(corr_min, c)
        report.append((name, kind, c))
    for name in lowrank:
        d, u = f.get_tensor(f"{name}.down.weight").float(), f.get_tensor(f"{name}.up.weight").float()
        if f"{name}.down.weight" not in gt:
            report.append((name, "lowrank: GGUF has no factors here", 0.0))
            corr_min = 0.0
            continue
        gd, gu = gg(f"{name}.down.weight"), gg(f"{name}.up.weight")
        r = min(d.shape[0], gd.shape[0])
        used = int((d.abs().amax(1) != 0).nonzero().max()) + 1
        pad_zero = float(d[used:].abs().max()) == 0.0 if used < d.shape[0] else True
        c = min(corr(d[:r], gd[:r]), corr(u[:, :r], gu[:, :r]))
        n_lin += 1
        corr_min = min(corr_min, c)
        report.append((name, f"lowrank r={d.shape[0]} (gguf {gd.shape[0]}) pad-zero {pad_zero}", c))
    gguf_blocks = {int(n.split(".")[1]) for n in gt if n.startswith("blocks.")}
    ours_blocks = {int(k.split(".")[1]) for k in keys if k.startswith("blocks.")}
    worst = sorted(report, key=lambda x: x[2])[:6]
    print(f"blocks: ours {len(ours_blocks)}, GGUF {len(gguf_blocks)}, same ids {ours_blocks == gguf_blocks}")
    print(f"tensors: ours {len(keys)}, GGUF {len(gt)}")
    print(f"unquantized tensors bit-exact vs GGUF: {small - len(exact_bad)}/{small}  mismatches: {exact_bad[:10]}")
    print(f"linears/factor pairs compared: {n_lin}, min correlation vs GGUF Q8_0 {corr_min:.6f}")
    for row in worst:
        print("   lowest:", row)
    ok = not exact_bad and corr_min > 0.9995 and ours_blocks == gguf_blocks
    print("LAYOUT", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------------------------------- remote
GGML = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 30: ("BF16", 2, 1), 8: ("Q8_0", 34, 32)}  # name, block bytes, block size


class RemoteGGUF:
    """The official GGUF read over HTTP range requests: the header, then only the byte ranges asked for."""

    def __init__(self, repo, filename, revision=None):
        import urllib.request

        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        self.urllib = urllib.request
        meta = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
        self.url, self.size, self.fetched = meta.location, meta.size, 0
        self._parse()

    def get(self, start, length):
        req = self.urllib.Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with self.urllib.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) != length:
            raise IOError(f"range {start}+{length}: got {len(data)} bytes")
        self.fetched += length
        return data

    def _parse(self):
        import struct as st

        buf = self.get(0, 8 << 20)
        pos = 0

        def take(fmt):
            nonlocal pos
            vals = st.unpack_from(fmt, buf, pos)
            pos += st.calcsize(fmt)
            return vals[0] if len(vals) == 1 else vals

        def string():
            nonlocal pos
            n = take("<Q")
            s = buf[pos:pos + n].decode()
            pos += n
            return s

        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = take("<I"), take("<Q")
                return [value(et) for _ in range(n)]
            return take(scalar[t])

        assert buf[:4] == b"GGUF", "not a GGUF file"
        pos = 4
        _version, n_tensors, n_kv = take("<I"), take("<Q"), take("<Q")
        self.kv = {}
        for _ in range(n_kv):
            key = string()
            self.kv[key] = value(take("<I"))
        self.tensors = {}
        for _ in range(n_tensors):
            name = string()
            dims = [take("<Q") for _ in range(take("<I"))]
            ttype, offset = take("<I"), take("<Q")
            shape = list(reversed(dims))
            name_t, bbytes, bsize = GGML[ttype]
            nbytes = math_prod(shape) // bsize * bbytes
            self.tensors[name.removeprefix("model.diffusion_model.")] = (shape, name_t, offset, nbytes)
        align = int(self.kv.get("general.alignment", 32))
        self.data_start = (pos + align - 1) // align * align

    def tensor(self, name):
        shape, ttype, offset, nbytes = self.tensors[name]
        raw = self.get(self.data_start + offset, nbytes)
        return decode(raw, ttype, shape)

    def rows(self, name, start, count):
        shape, ttype, offset, _ = self.tensors[name]
        _, bbytes, bsize = next(v for v in GGML.values() if v[0] == ttype)
        row = shape[1] // bsize * bbytes
        raw = self.get(self.data_start + offset + start * row, count * row)
        return decode(raw, ttype, [count, shape[1]])


def math_prod(shape):
    out = 1
    for s in shape:
        out *= int(s)
    return out


def decode(raw, ttype, shape):
    import numpy as np

    if ttype == "F32":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)
    if ttype == "F16":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float16).astype(np.float32)).reshape(shape)
    if ttype == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().reshape(shape)
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return torch.from_numpy(d * q).reshape(shape)


def check_remote(converted, repo, filename, revision=None):
    from concurrent.futures import ThreadPoolExecutor

    gg = RemoteGGUF(repo, filename, revision)
    f = safe_open(converted, "pt", device="cpu")
    keys = set(f.keys())
    meta = json.loads(f.metadata().get("hyperflow_int8", "{}"))
    print(f"remote GGUF: {len(gg.tensors)} tensors, header {gg.data_start} bytes; ours: {len(keys)} tensors, "
          f"variant {meta.get('variant')}")
    problems = []

    # Structure: blocks, every unquantized tensor's name and shape, and each linear's form.
    g_blocks = {int(n.split(".")[1]) for n in gg.tensors if n.startswith("blocks.")}
    o_blocks = {int(k.split(".")[1]) for k in keys if k.startswith("blocks.")}
    if g_blocks != o_blocks:
        problems.append(f"blocks differ: GGUF {len(g_blocks)}, ours {len(o_blocks)}")
    linear_re = (".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj.", ".out_proj.", ".fc1.", ".fc2.")
    small = []
    for name, (shape, ttype, _, nbytes) in gg.tensors.items():
        if any(s in name for s in linear_re) and ("blocks." in name):
            continue
        if name not in keys:
            problems.append(f"missing in ours: {name}")
            continue
        if list(f.get_slice(name).get_shape()) != shape:
            problems.append(f"shape {name}: GGUF {shape}, ours {f.get_slice(name).get_shape()}")
        if ttype in ("F32", "BF16", "F16"):
            small.append(name)
    forms = 0
    for b in sorted(g_blocks):
        for lin in ("attn.qkv_proj", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            name = f"blocks.{b}.{lin}"
            g_dense, g_lr = f"{name}.weight" in gg.tensors, f"{name}.down.weight" in gg.tensors
            o_dense, o_lr = f"{name}.weight" in keys, f"{name}.down.weight" in keys
            if (g_dense, g_lr) != (o_dense, o_lr):
                problems.append(f"form {name}: GGUF dense/lowrank {g_dense}/{g_lr}, ours {o_dense}/{o_lr}")
                continue
            if g_dense and gg.tensors[f"{name}.weight"][0] != list(f.get_slice(f"{name}.weight").get_shape()):
                problems.append(f"shape {name}.weight")
            if g_lr:
                g_rank = gg.tensors[f"{name}.down.weight"][0][0]
                o_rank = f.get_slice(f"{name}.down.weight").get_shape()[0]
                if -(-o_rank // 256) * 256 != g_rank:
                    problems.append(f"rank {name}: GGUF {g_rank}, ours {o_rank}")
            forms += g_dense or g_lr

    # Values: every unquantized tensor up to 6 MB, bit for bit (adaln: 3 blocks), fetched by range.
    budget = [n for n in small if gg.tensors[n][3] <= 6 << 20
              and (".adaln_proj." not in n or n.split(".")[1] in ("0", str(len(g_blocks) // 2), str(len(g_blocks) - 1)))
              or not n.startswith("blocks.") and gg.tensors[n][3] <= 6 << 20]
    budget = sorted(set(budget))
    with ThreadPoolExecutor(8) as pool:
        remote = dict(zip(budget, pool.map(gg.tensor, budget)))
    exact_bad = [n for n in budget if not torch.equal(remote[n], f.get_tensor(n).float())]

    # Quantized rows: a few rows from each half of fused / swapped / low-rank linears, vs our dequant.
    samples = []
    for b in sorted(g_blocks):
        for name, starts in ((f"blocks.{b}.attn.qkv_proj.weight", (0, 7168, 14336)), (f"blocks.{b}.mlp.fc1.weight", (0, 14336)),
                             (f"blocks.{b}.mlp.fc1.up.weight", (0, 14336)), (f"blocks.{b}.attn.q_proj.up.weight", (0,))):
            if name in gg.tensors and len([s for s in samples if s[0].endswith(name.split(".", 2)[2])]) < 2:
                samples.append((name, starts))
    samples.append(("token_refiner.blocks.0.mlp.fc1.weight", (0, 14336)))
    corr_min = 1.0
    lines = []
    for name, starts in samples:
        for start in starts:
            g = gg.rows(name, start, 16)
            base = name.rsplit(".", 1)[0]
            if name.endswith("up.weight"):
                o = f.get_slice(name)[start:start + 16].float()
                r = min(o.shape[1], g.shape[1])
                c = corr(o[:, :r], g[:, :r])
            else:
                o, _, _ = ours_dense(f, base, slice(start, start + 16))
                c = corr(o, g)
            corr_min = min(corr_min, c)
            lines.append(f"  {name} rows {start}..{start + 15}: corr {c:.6f}")
    print(f"structure: {forms} block linears with matching form/shape/rank, blocks {len(o_blocks)}=={len(g_blocks)}; "
          f"problems: {problems[:8]}")
    print(f"unquantized tensors bit-exact vs GGUF: {len(budget) - len(exact_bad)}/{len(budget)} {exact_bad[:5]}")
    for line in lines:
        print(line)
    print(f"min row correlation vs Q8_0: {corr_min:.6f}; fetched {gg.fetched / 1e6:.1f} MB of {gg.size / 1e9:.1f} GB")
    ok = not problems and not exact_bad and corr_min > 0.9995
    print("REMOTE LAYOUT", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------------------------------- exact
def load_reference_int8(int8_py):
    spec = importlib.util.spec_from_file_location("hyperflow_reference_int8", int8_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def int8_samples(hf, kept):
    """INT8 linears to sample: first and last fused q/k/v, fc1 (half swap), out_proj and fc2, preferring an INT8 that
    sits in a block with low-rank parts."""
    quant = set(hf.config["quantization"]["linears"])
    lowrank = {int(n.split(".")[1]) for n in (hf.config.get("lowrank_ranks") or {})}
    picks = {}
    for new, old in enumerate(kept):
        p = f"transformer_blocks.{old}"
        qkv = [f"{p}.attn.to_{c}" for c in "qkv"]
        cands = []
        if all(q in quant for q in qkv):
            cands.append(("qkv", f"blocks.{new}.attn.qkv_proj", qkv, False))
        for a, b, kind, sw in (("attn.to_out.0", "attn.out_proj", "out", False), ("ff.net.0.proj", "mlp.fc1", "fc1", True),
                               ("ff.net.2", "mlp.fc2", "fc2", False)):
            if f"{p}.{a}" in quant:
                cands.append((kind, f"blocks.{new}.{b}", [f"{p}.{a}"], sw))
        for kind, name, srcs, sw in cands:
            first, last, mixed = (kind, "first"), (kind, "last"), (kind, "mixed")
            picks.setdefault(first, (name, srcs, sw))
            picks[last] = (name, srcs, sw)
            if old in lowrank:
                picks.setdefault(mixed, (name, srcs, sw))
    seen, out = set(), []
    for value in picks.values():
        if value[0] not in seen:
            seen.add(value[0])
            out.append(value)
    return out


def check_exact(converted, src_dir, int8_py):
    ref = load_reference_int8(int8_py)
    hf = HF(src_dir)
    kept = kept_blocks(hf.config)
    group = hf.config["quantization"]["group"]
    R = ref.regular_hadamard(group)  # their float32 Hadamard
    f = safe_open(converted, "pt", device="cpu")
    keys = set(f.keys())
    worst = 0.0
    n = 0
    lines = []

    def ref_dense(prefix):  # HyperFlow: qweight = round(W R / wscale)  =>  W = (q * wscale) R^T
        q, s = hf[f"{prefix}.qweight"].float(), hf[f"{prefix}.wscale"].float()
        return ((q * s[:, None]).reshape(q.shape[0], -1, group) @ R.T).reshape(q.shape)

    def swap(t):
        h = t.shape[0] // 2
        return torch.cat([t[h:], t[:h]])

    # Bit-level: stored int8 codes and scales vs the source, for every INT8 linear (fused / swapped / split).
    code_bad = 0
    for new, old in enumerate(kept):
        p, o = f"transformer_blocks.{old}", f"blocks.{new}"
        qkv = [f"{p}.attn.{x}" for x in ("to_q", "to_k", "to_v")]
        if all(f"{x}.qweight" in hf for x in qkv):
            pairs = [(f"{o}.attn.qkv_proj", [hf[f"{x}.qweight"] for x in qkv], [hf[f"{x}.wscale"] for x in qkv], False)]
        else:
            pairs = [(f"{o}.attn.{b}", [hf[f"{a}.qweight"]], [hf[f"{a}.wscale"]], False)
                     for a, b in zip(qkv, ("q_proj", "k_proj", "v_proj")) if f"{a}.qweight" in hf]
        for a, b, sw in ((f"{p}.attn.to_out.0", f"{o}.attn.out_proj", False), (f"{p}.ff.net.0.proj", f"{o}.mlp.fc1", True),
                         (f"{p}.ff.net.2", f"{o}.mlp.fc2", False)):
            if f"{a}.qweight" in hf:
                pairs.append((b, [hf[f"{a}.qweight"]], [hf[f"{a}.wscale"]], sw))
        for name, qs, ss, sw in pairs:
            q, s = torch.cat(qs), torch.cat(ss)
            if sw:
                q, s = swap(q), swap(s)
            same = torch.equal(f.get_tensor(f"{name}.weight"), q) and torch.equal(f.get_tensor(f"{name}.weight_scale").flatten(), s)
            code_bad += not same
            n += 1
    lines.append(f"INT8 linears with bit-identical codes + scales: {n - code_bad}/{n}")

    # Dequantized weights: ComfyUI's dequant of our tensors vs HyperFlow's reference dequant, sampled linears.
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight

    samples = int8_samples(hf, kept)
    for name, srcs, sw in samples:
        conf = json.loads(bytes(f.get_tensor(f"{name}.comfy_quant").tolist()).decode())
        ours = dequantize_int8_convrot_weight(f.get_tensor(f"{name}.weight"), f.get_tensor(f"{name}.weight_scale"),
                                              conf["convrot_groupsize"])
        want = torch.cat([ref_dense(x) for x in srcs])
        if sw:
            want = swap(want)
        err = float((ours - want).abs().max() / want.abs().max())
        worst = max(worst, err)
        lines.append(f"  dequant {name:28s} rel max err {err:.2e}  ({conf})")

    # Low-rank factors: our padded pair computes exactly the source's up @ down.
    lr_bad = 0
    lr_n = 0
    for new, old in enumerate(kept):
        for a, b in (("attn.to_q", "attn.q_proj"), ("attn.to_k", "attn.k_proj"), ("attn.to_v", "attn.v_proj"),
                     ("attn.to_out.0", "attn.out_proj"), ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2")):
            src = f"transformer_blocks.{old}.{a}"
            if f"{src}.down.weight" not in hf:
                continue
            d, u = hf[f"{src}.down.weight"], hf[f"{src}.up.weight"]
            od, ou = f.get_tensor(f"blocks.{new}.{b}.down.weight"), f.get_tensor(f"blocks.{new}.{b}.up.weight")
            r = d.shape[0]
            if a == "ff.net.0.proj":
                u = swap(u)
            ok = (torch.equal(od[:r], d) and torch.equal(ou[:, :r], u) and not od[r:].any() and not ou[:, r:].any()
                  and od.shape[0] % 8 == 0)
            lr_bad += not ok
            lr_n += 1
    lines.append(f"low-rank pairs bit-identical (+ zero padding to x8): {lr_n - lr_bad}/{lr_n}")

    # Forward with the reference's own activation arithmetic: our stored tensors + ComfyUI's Hadamard must give
    # HyperFlow's Int8RotLinear output bit for bit (fused q/k/v and the fc1 half swap included).
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    def ours_forward(name, x):
        w, s = f.get_tensor(f"{name}.weight"), f.get_tensor(f"{name}.weight_scale")
        h = _build_hadamard(group, dtype=torch.bfloat16)
        xr = (x.reshape(x.shape[0], -1, group) @ h).reshape(x.shape).float()
        sx = xr.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
        xq = (xr / sx).round().clamp(-127, 127)
        return ((xq @ w.float().t()) * sx * s.reshape(1, -1)).to(torch.bfloat16)

    def ref_forward(prefix, x):
        q = hf[f"{prefix}.qweight"]
        m = ref.Int8RotLinear(q.shape[1], q.shape[0], group)
        m.qweight.data.copy_(q)
        m.wscale.data.copy_(hf[f"{prefix}.wscale"])
        return m(x)

    torch.manual_seed(0)
    fwd_bad = 0
    for name, srcs, sw in samples:
        k_in = f.get_slice(f"{name}.weight").get_shape()[1]
        x = (torch.randn(48, k_in) * 0.8).to(torch.bfloat16)
        want = torch.cat([ref_forward(p, x) for p in srcs], -1)
        if sw:
            want = swap(want.t()).t()
        same = torch.equal(ours_forward(name, x), want)
        fwd_bad += not same
        lines.append(f"  forward {name:28s} bit-identical to HyperFlow Int8RotLinear: {same}")
    for line in lines:
        print(line)
    ok = code_bad == 0 and lr_bad == 0 and worst < 1e-5 and fwd_bad == 0
    print("EXACT", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------------------------------- loader
def mini_blocks(converted):
    """Up to 4 converted blocks, one per distinct layout (INT8 / low-rank mixes), from the file's own metadata."""
    layout = json.loads(safe_open(converted, "pt", device="cpu").metadata()["hyperflow_int8"])["layout"]
    chosen, seen = [], set()
    for new in sorted(layout, key=int):
        entry = layout[new]
        form = (entry["qkv"], entry["out_proj"], entry["fc1"], entry["fc2"])
        if form not in seen:
            seen.add(form)
            chosen.append((int(new), entry["source_block"], form))
    return chosen[:4]


def write_mini(converted, mini_path):
    """A 4-block, refiner-free slice of the converted file (keeps RAM low): same tensors, blocks renumbered."""
    from safetensors.torch import save_file

    chosen = [new for new, _, _ in mini_blocks(converted)]
    f = safe_open(converted, "pt", device="cpu")
    out = {}
    for k in f.keys():
        if k.startswith("token_refiner.blocks."):
            continue
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            if idx not in chosen:
                continue
            out[f"blocks.{chosen.index(idx)}." + k.split(".", 2)[2]] = f.get_tensor(k)
        else:
            out[k] = f.get_tensor(k)
    save_file(out, mini_path, metadata={"format": "pt", "hyperflow_int8": f.metadata()["hyperflow_int8"]})
    return mini_path


def comfy_env():
    sys.path.insert(0, COMFY)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.model_management as mm
    import folder_paths  # noqa: F401

    # Mirror the GPU's dtype choice on CPU: bf16 compute, so non-INT8 weights stay bf16 as they would on the 5090.
    mm.should_use_bf16 = lambda *args, **kwargs: True


def check_loader(converted, src_dir, int8_py, mini_path):
    if not os.path.exists(mini_path):
        write_mini(converted, mini_path)
    blocks = mini_blocks(converted)
    print("mini slice (converted block, source block, q/k/v / out / fc1 / fc2):", blocks)
    comfy_env()
    spec = importlib.util.spec_from_file_location("hyperflow_int8_pack", os.path.join(HERE, "__init__.py"),
                                                  submodule_search_locations=[HERE])
    pack = importlib.util.module_from_spec(spec)
    sys.modules["hyperflow_int8_pack"] = pack
    spec.loader.exec_module(pack)
    loader = sys.modules["hyperflow_int8_pack.loader"]
    print("NODE_CLASS_MAPPINGS:", list(pack.NODE_CLASS_MAPPINGS), pack.NODE_DISPLAY_NAME_MAPPINGS)
    from comfy.quant_ops import QuantizedTensor

    patcher = loader.load_hyperflow_int8(mini_path, "legacy")
    model = patcher.model.diffusion_model
    patch, _ = loader.official()
    print("model class:", type(model).__name__, "| time_embedder:", type(model.time_embedder).__name__,
          "| blocks:", len(model.blocks), "| sigmas:", [round(float(s), 4) for s in model.hyperflow_sigmas])

    # Structure: no placeholder survived, INT8 where expected, fp32 islands.
    placeholders, int8, bf16, fp32 = [], 0, 0, []
    for name, p in model.named_parameters():
        if isinstance(p.data, QuantizedTensor) or isinstance(p, QuantizedTensor):
            int8 += 1
            continue
        if p.numel() > 1 and 0 in p.stride():
            placeholders.append(name)
        if p.dtype == torch.float32:
            fp32.append(name)
        elif p.dtype == torch.bfloat16:
            bf16 += 1
    print(f"params: INT8 QuantizedTensor {int8}, bf16 {bf16}, fp32 {len(fp32)}, placeholders left {placeholders}")
    print("fp32:", sorted({n.rsplit('.', 1)[0] for n in fp32}))

    def stale_dtypes():  # archived model dtype (what the patchers stage in) vs the parameter's real dtype
        out = []
        for n, m in model.named_modules():
            for leaf in ("weight", "bias"):
                p, want = getattr(m, leaf, None), getattr(m, f"{leaf}_comfy_model_dtype", None)
                if isinstance(p, torch.Tensor) and not isinstance(p, QuantizedTensor) and want is not None and want != p.dtype:
                    out.append(f"{n}.{leaf}")
        return out

    print("archived-dtype mismatches:", stale_dtypes())
    import comfy.model_management as mm

    mm.load_models_gpu([patcher])  # CPU here: runs the legacy patcher's load path over the swapped modules
    after = sorted({n.rsplit('.', 1)[0] for n, p in model.named_parameters() if p.dtype == torch.float32})
    print("after patcher load, fp32 modules:", len(after), "| still fp32:", after == sorted({n.rsplit('.', 1)[0] for n in fp32}))

    ref = load_reference_int8(int8_py)
    hf = HF(src_dir)
    group = hf.config["quantization"]["group"]
    torch.manual_seed(0)
    x = (torch.randn(1, 40, 5376) * 1.5).to(torch.bfloat16)

    def ref_linear(prefix):
        if f"{prefix}.qweight" in hf:
            q = hf[f"{prefix}.qweight"]
            m = ref.Int8RotLinear(q.shape[1], q.shape[0], group)
            m.qweight.data.copy_(q)
            m.wscale.data.copy_(hf[f"{prefix}.wscale"])
            return m.forward  # CPU: their plain-torch implementation of the same scheme
        d, u = hf[f"{prefix}.down.weight"], hf[f"{prefix}.up.weight"]
        return lambda t: torch.nn.functional.linear(torch.nn.functional.linear(t, d), u)

    def ref_dense_bf16(prefix):  # unquantized-activation baseline: x @ W_deq^T in fp32
        if f"{prefix}.qweight" in hf:
            q, s = hf[f"{prefix}.qweight"].float(), hf[f"{prefix}.wscale"].float()
            w = ((q * s[:, None]).reshape(q.shape[0], -1, group) @ ref.regular_hadamard(group).T).reshape(q.shape)
        else:
            w = hf[f"{prefix}.up.weight"].float() @ hf[f"{prefix}.down.weight"].float()
        return lambda t: t.float() @ w.T

    def stats(a, b):
        a, b = a.float(), b.float()
        return f"rel-L2 {float((a - b).norm() / b.norm()):.2e}  corr {corr(a, b):.6f}"

    rows = []
    for mini, (_, src, _) in enumerate(blocks):
        block = model.blocks[mini]
        p = f"transformer_blocks.{src}"
        with torch.no_grad():
            got = block.attn.qkv_proj(x[0])
            want = torch.cat([ref_linear(f"{p}.attn.to_{c}")(x[0]) for c in "qkv"], -1)
            base = torch.cat([ref_dense_bf16(f"{p}.attn.to_{c}")(x[0]) for c in "qkv"], -1)
            rows.append((f"block {src} qkv [{type(block.attn.qkv_proj).__name__}]", stats(got, want), stats(want, base)))
            h = (torch.randn(40, 7168) * 0.5).to(torch.bfloat16)
            got = block.attn.out_proj(h)
            want = ref_linear(f"{p}.attn.to_out.0")(h)
            rows.append((f"block {src} out_proj [{type(block.attn.out_proj).__name__}]", stats(got, want),
                         stats(want, ref_dense_bf16(f"{p}.attn.to_out.0")(h))))
            got = block.mlp(x[0])
            fc1 = ref_linear(f"{p}.ff.net.0.proj")(x[0])
            value, gate = fc1.chunk(2, -1)  # diffusers SwiGLU: value | gate
            want = ref_linear(f"{p}.ff.net.2")(value * torch.nn.functional.silu(gate))
            rows.append((f"block {src} mlp [{type(block.mlp).__name__}: fc1 {type(block.mlp.fc1).__name__}, "
                         f"fc2 {type(block.mlp.fc2).__name__}]", stats(got, want), ""))
    print("module forwards, ComfyUI CPU path vs HyperFlow reference (and reference INT8 vs its dequantized fp32 weights):")
    for name, s1, s2 in rows:
        print(f"  {name:70s} {s1}   | ref int8 vs fp32: {s2}")

    # Two-time embedding in fp32: temb = te(t) + gate * (ee(r) - te(t)), checkpoint fp32 weights.
    t = torch.tensor([0.0, 0.25, 0.999], dtype=torch.float32)
    r = torch.tensor([0.3, 0.6, 0.999], dtype=torch.float32)
    state = model.time_embedder.__dict__["endpoint_state"]
    state.set_endpoints(dict(zip(t.tolist(), r.tolist())))
    with torch.no_grad():
        got = model.time_embedder(t)

        def emb(v, a, b):
            half = 128
            freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, dtype=torch.float32) / half)
            args = v[:, None] * freqs[None]
            e = torch.cat([torch.cos(args), torch.sin(args)], -1)
            return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(e, hf[f"{a}.weight"], hf[f"{a}.bias"])),
                                              hf[f"{b}.weight"], hf[f"{b}.bias"])

        te = emb(t, "time_embedder.linear_1", "time_embedder.linear_2")
        ee = emb(r, "endpoint_time_embedder.linear_1", "endpoint_time_embedder.linear_2")
        want = te + hf.config["hyperflow_gate"] * (ee - te)
    print(f"two-time embedding vs fp32 reference: max abs err {float((got - want).abs().max()):.2e} "
          f"(r = t would differ by {float((te - want).abs().max()):.2e})")

    # Smoke forward of the 4-block HyperFlowModel: (t, r) from sample_sigmas vs r = t.
    sigmas = model.hyperflow_sigmas
    video = torch.randn(1, 24, 2, 8, 8)
    audio = torch.randn(1, 32, 2, 4)
    context = (torch.randn(1, 12, 5120) * 0.5).to(torch.bfloat16)
    timestep = (sigmas[2:3] * 1000.0).float()
    with torch.no_grad():
        out_tr = model([video, audio], timestep, context, transformer_options={"sample_sigmas": sigmas})
        pairs = dict(state.endpoints)
        out_tt = model([video, audio], timestep, context, transformer_options={})
    finite = all(bool(torch.isfinite(o).all()) for o in out_tr)
    diff = float((out_tr[0].float() - out_tt[0].float()).norm() / out_tt[0].float().norm())
    print(f"smoke forward at sigma {float(sigmas[2]):.4f}: shapes {[tuple(o.shape) for o in out_tr]}, finite {finite}, "
          f"(t -> r) pairs {pairs}, (t,r) vs r=t video rel diff {diff:.3e}")
    return finite and not placeholders and int8 > 0


def check_import():
    """Load the official pack and this pack with ComfyUI's own custom-node loader (no server), then resolve
    the official modules the way the node does at load time."""
    import asyncio

    comfy_env()
    import nodes

    custom = os.path.dirname(HERE)
    for pack in ("ComfyUI-HyperFlow-Flash", os.path.basename(HERE)):
        ok = asyncio.run(nodes.load_custom_node(os.path.join(custom, pack), set(), module_parent="custom_nodes"))
        print(f"load_custom_node({pack}): {ok}")
    names = [n for n in nodes.NODE_CLASS_MAPPINGS if n.startswith("HyperFlow")]
    print("HyperFlow nodes registered:", names)
    loader = sys.modules[nodes.NODE_CLASS_MAPPINGS["HyperFlowINT8Loader"].__module__]
    patch, loader_nodes = loader.official()
    print("official modules resolved:", patch.__name__, loader_nodes.__name__)
    inputs = nodes.NODE_CLASS_MAPPINGS["HyperFlowINT8Loader"].INPUT_TYPES()
    files = inputs["required"]["unet_name"][0]
    print("unet_name choices with 'hyperflow':", [n for n in files if "hyperflow" in n.lower()])
    print("optional:", {k: v[0] for k, v in inputs["optional"].items()})
    return "HyperFlowINT8Loader" in names and "HyperFlowSigmas" in names and "hyperflow_flash_official" not in patch.__name__


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "layout":
        ok = check_layout(*sys.argv[2:5])
    elif mode == "exact":
        ok = check_exact(*sys.argv[2:5])
    elif mode == "loader":
        ok = check_loader(*sys.argv[2:6])
    elif mode == "import":
        ok = check_import()
    elif mode == "remote":
        ok = check_remote(*sys.argv[2:6])
    else:
        raise SystemExit(__doc__)
    sys.exit(0 if ok else 1)
