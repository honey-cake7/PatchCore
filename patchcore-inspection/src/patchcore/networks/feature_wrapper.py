"""Generic adapter to make a multi-scale backbone hookable by PatchCore.

PatchCore's `NetworkFeatureAggregator` registers a forward hook on a single named
submodule and expects that module's output to be a feature map. Some backbones only
expose their pyramid features via a custom method (SegFormer's `output_hidden_states`,
MambaVision's `forward_features`) or return tuples from their stage modules, neither of
which the aggregator can hook directly.

`MultiScaleWrapper` runs such a backbone's native multi-scale method and routes each of
the resulting `[B, C, H, W]` maps through a top-level `Identity` (`stages.0..N`). PatchCore
can then hook them by the single-dot names `stages.0`, `stages.1`, ... and the captured
tensors are already 4-D, so `ForwardHook` passes them through unchanged.
"""
import torch.nn as nn


class MultiScaleWrapper(nn.Module):
    def __init__(self, model, extract_fn, num_stages=4):
        """
        Args:
            model: the wrapped backbone (registered as a submodule, so it moves with .to()).
            extract_fn: callable(model, x) -> list/tuple of N spatial [B, C, H, W] maps.
            num_stages: number of feature maps extract_fn returns (number of hook points).
        """
        super().__init__()
        self.model = model
        self._extract_fn = extract_fn
        self.stages = nn.ModuleList([nn.Identity() for _ in range(num_stages)])

    def forward(self, x):
        feats = self._extract_fn(self.model, x)
        return [stage(f) for stage, f in zip(self.stages, feats)]
