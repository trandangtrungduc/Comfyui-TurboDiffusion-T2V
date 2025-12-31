"""
ComfyUI-native offload integration.

This module provides a callable wrapper that integrates a large torch.nn.Module
with ComfyUI's native model management / async weight offloading system.

Why this exists:
- ComfyUI prints "Using async weight offloading with 2 streams" when its native
  model-management/offload path is active.
- Our custom layerwise wrapper works, but bypasses ComfyUI's async offloader.

We keep this implementation defensive because ComfyUI APIs can vary by version.
"""

from __future__ import annotations

from typing import Optional, Any

import torch


def _estimate_model_size_bytes(model: torch.nn.Module) -> int:
    total = 0
    try:
        for p in model.parameters():
            total += p.numel() * p.element_size()
    except Exception:
        pass
    try:
        for b in model.buffers():
            total += b.numel() * b.element_size()
    except Exception:
        pass
    return int(total)


class ComfyNativeOffloadCallable:
    """
    Callable wrapper around a ComfyUI ModelPatcher (or equivalent) that ensures the
    model is loaded via ComfyUI's model_management before each forward.
    """

    def __init__(self, model: torch.nn.Module, *, load_device: Optional[torch.device] = None):
        try:
            import comfy.model_management as mm  # type: ignore
            import comfy.model_patcher as mp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ComfyUI is not available in this Python environment; cannot use comfy_native offload mode."
            ) from e

        self._mm = mm
        self._mp = mp

        self._load_device = load_device if load_device is not None else mm.get_torch_device()
        self._offload_device = mm.unet_offload_device() if hasattr(mm, "unet_offload_device") else torch.device("cpu")

        # Keep the raw model on CPU initially; ComfyUI will manage transfers/offload.
        model = model.to("cpu").eval()

        size_bytes = _estimate_model_size_bytes(model)

        # ModelPatcher constructor differs across ComfyUI versions; try a few forms.
        patcher = None
        for ctor in (
            lambda: mp.ModelPatcher(model, self._load_device, self._offload_device),
            lambda: mp.ModelPatcher(model, self._load_device, self._offload_device, size=size_bytes),
            lambda: mp.ModelPatcher(model, self._load_device, self._offload_device, size=size_bytes, current_device=self._offload_device),
        ):
            try:
                patcher = ctor()
                break
            except TypeError:
                continue

        if patcher is None:
            raise RuntimeError("Failed to construct ComfyUI ModelPatcher for this ComfyUI version.")

        self.patcher = patcher

    def _ensure_loaded(self):
        mm = self._mm
        # Prefer the most specific APIs if present.
        if hasattr(mm, "load_model_gpu"):
            mm.load_model_gpu(self.patcher)
            return
        if hasattr(mm, "load_models_gpu"):
            mm.load_models_gpu([self.patcher])
            return
        # Fallback: try moving the patched model.
        m = getattr(self.patcher, "model", None)
        if isinstance(m, torch.nn.Module):
            m.to(self._load_device)

    def to(self, device: Any, **_):
        # Update target load device; actual movement handled by ComfyUI.
        try:
            self._load_device = torch.device(device)
        except Exception:
            pass
        return self

    def cpu(self):
        # Best-effort offload.
        if hasattr(self._mm, "unload_model"):
            try:
                self._mm.unload_model(self.patcher)
            except Exception:
                pass
        return self

    def cuda(self, device: Optional[int] = None):
        idx = 0 if device is None else int(device)
        self._load_device = torch.device(f"cuda:{idx}")
        return self

    def __call__(self, *args, **kwargs):
        self._ensure_loaded()
        # Align floating-point inputs with the model's parameter dtype to avoid
        # Linear matmul dtype mismatches (e.g., fp32 inputs vs bf16 weights).
        target_dtype = None
        model_for_dtype = getattr(self.patcher, "model", None)
        if isinstance(model_for_dtype, torch.nn.Module):
            try:
                p = next(model_for_dtype.parameters(), None)
                target_dtype = p.dtype if p is not None else None
            except Exception:
                target_dtype = None

        def convert(obj):
            if isinstance(obj, torch.Tensor):
                if target_dtype is not None and obj.is_floating_point() and obj.dtype != target_dtype:
                    return obj.to(dtype=target_dtype)
                return obj
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, tuple):
                if type(obj) is tuple:
                    return tuple(convert(x) for x in obj)
                if hasattr(obj, "_fields"):
                    return type(obj)(*(convert(x) for x in obj))
                return obj
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        args = convert(args)
        kwargs = convert(kwargs)

        # ModelPatcher generally exposes the underlying module at `.model`.
        model = getattr(self.patcher, "model", None)
        if callable(model):
            return model(*args, **kwargs)
        # Some variants need `.patch_model()` to get the callable module.
        if hasattr(self.patcher, "patch_model"):
            m = self.patcher.patch_model()
            return m(*args, **kwargs)
        raise RuntimeError("ComfyUI ModelPatcher did not expose a callable model.")


