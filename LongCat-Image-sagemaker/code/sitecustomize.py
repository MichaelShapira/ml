import pathlib

def _patch():
    import diffusers
    target = pathlib.Path(diffusers.__file__).parent / "models" / "attention_dispatch.py"
    src = target.read_text()

    # Check if already patched
    if "patched_by_sitecustomize" in src:
        print("[sitecustomize] already patched, skipping")
        return

    fixes = [
        ("q: 'torch.Tensor'",                      "q: torch.Tensor"),
        ("k: 'torch.Tensor'",                      "k: torch.Tensor"),
        ("v: 'torch.Tensor'",                      "v: torch.Tensor"),
        ("qv: 'torch.Tensor | None'",              "qv: Optional[torch.Tensor]"),
        ("q_descale: 'torch.Tensor | None'",       "q_descale: Optional[torch.Tensor]"),
        ("k_descale: 'torch.Tensor | None'",       "k_descale: Optional[torch.Tensor]"),
        ("v_descale: 'torch.Tensor | None'",       "v_descale: Optional[torch.Tensor]"),
        ("softmax_scale: 'float | None'",          "softmax_scale: Optional[float]"),
        ("pack_gqa: 'bool | None'",                "pack_gqa: Optional[bool]"),
        ("-> 'tuple[torch.Tensor, torch.Tensor]'", "-> Tuple[torch.Tensor, torch.Tensor]"),
    ]

    for old, new in fixes:
        src = src.replace(old, new)

    # Insert typing import AFTER any __future__ imports, not at top
    if "from typing import Optional, Tuple" not in src:
        src = src.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\nfrom typing import Optional, Tuple  # patched_by_sitecustomize"
        )
    
    target.write_text(src)
    print("[sitecustomize] attention_dispatch.py patched OK")

try:
    _patch()
except Exception as e:
    print(f"[sitecustomize] patch skipped: {e}")