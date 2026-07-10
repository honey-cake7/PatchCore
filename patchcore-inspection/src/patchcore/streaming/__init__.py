"""Streaming, drift-aware memory-bank maintenance for PatchCore.

This sub-package reformulates PatchCore memory-bank maintenance as a budgeted
sequential decision problem: as a stream of *normal* images arrives under a
fixed memory budget, a policy (hand-designed baselines or a learned PPO agent)
decides which incoming patch embeddings to admit and which existing entries to
evict, so the bank stays calibrated as the appearance of normal tissue drifts.

Everything downstream of the frozen PatchCore feature extractor operates on
cached numpy patch embeddings (see :mod:`patchcore.streaming.cache`), never on
images, so environment steps are cheap and every run is reproducible.
"""
