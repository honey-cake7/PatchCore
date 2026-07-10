"""Build ordered normal streams and per-stage labeled test manifests.

Operates directly on the MVTec-style on-disk layout used by every Kvasir-family
dataset in this fork::

    <source>/<classname>/train/good/*
    <source>/<classname>/test/{good,<anomaly_type>}/*
    <source>/<classname>/ground_truth/<anomaly_type>/*

Synthetic staged drift shuffles the normal pool once and assigns stages by
stream position; metadata-derived real drift orders the stream by acquisition
provenance recovered from filename prefixes (``<video_id>_frame_<n>`` for
Kvasir-Capsule, ``<anatomical-class>_...`` for HyperKvasir).
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_PROVENANCE_RE = re.compile(r"^(?P<group>[A-Za-z0-9\-]+?)[_-](?P<frame>\d+)")


@dataclass
class StreamEntry:
    image_path: str
    stream_pos: int
    stage_id: int
    group: Optional[str] = None       # video_id / anatomical class
    frame_idx: Optional[int] = None


@dataclass
class TestItem:
    image_path: str
    label: int                        # 0 normal, 1 anomaly
    mask_path: Optional[str]


def _list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(IMG_EXTS)
    ]


def _parse_provenance(path: str) -> Tuple[Optional[str], Optional[int]]:
    m = _PROVENANCE_RE.match(os.path.basename(path))
    if not m:
        return None, None
    return m.group("group"), int(m.group("frame"))


class StreamBuilder:
    def __init__(
        self, source: str, classname: str, resize: int = 256, imagesize: int = 224
    ) -> None:
        self.source = source
        self.classname = classname
        self.resize = resize
        self.imagesize = imagesize
        self.class_root = os.path.join(source, classname)

    # ---- normal streams --------------------------------------------------
    def _normal_pool(self) -> List[str]:
        return _list_images(os.path.join(self.class_root, "train", "good"))

    def synthetic_drift_stream(self, schedule, seed: int = 0) -> List[StreamEntry]:
        """Normals shuffled once, stage assigned by stream position."""
        import numpy as np

        pool = self._normal_pool()
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pool))
        n = len(pool)
        entries = []
        for pos, idx in enumerate(order):
            stage = schedule.stage_at(pos, n)
            g, fr = _parse_provenance(pool[idx])
            entries.append(StreamEntry(pool[idx], pos, stage, g, fr))
        return entries

    def real_drift_stream(self) -> List[StreamEntry]:
        """Order normals by acquisition provenance; stage = provenance group."""
        pool = self._normal_pool()
        parsed = [(p, *_parse_provenance(p)) for p in pool]
        # Sort by (group, frame). Entries without provenance fall back to name.
        parsed.sort(key=lambda t: (t[1] or os.path.basename(t[0]), t[2] or 0))
        groups = {}
        for _, g, _f in parsed:
            key = g or "unknown"
            if key not in groups:
                groups[key] = len(groups)
        return [
            StreamEntry(p, pos, groups.get(g or "unknown", 0), g, f)
            for pos, (p, g, f) in enumerate(parsed)
        ]

    # ---- test manifests --------------------------------------------------
    def _test_items(self) -> List[TestItem]:
        test_root = os.path.join(self.class_root, "test")
        gt_root = os.path.join(self.class_root, "ground_truth")
        items: List[TestItem] = []
        if not os.path.isdir(test_root):
            return items
        for anomaly in sorted(os.listdir(test_root)):
            adir = os.path.join(test_root, anomaly)
            if not os.path.isdir(adir):
                continue
            imgs = _list_images(adir)
            if anomaly == "good":
                items += [TestItem(p, 0, None) for p in imgs]
            else:
                mask_dir = os.path.join(gt_root, anomaly)
                masks = {os.path.basename(m): m for m in _list_images(mask_dir)}
                for p in imgs:
                    base = os.path.basename(p)
                    # masks may share the exact name or a "<stem>_mask" variant
                    mp = masks.get(base)
                    if mp is None:
                        stem = os.path.splitext(base)[0]
                        for cand in masks:
                            if cand.startswith(stem):
                                mp = masks[cand]
                                break
                    items.append(TestItem(p, 1, mp))
        return items

    def per_stage_test_manifests(self, schedule) -> Dict[int, List[TestItem]]:
        """One copy of the full labeled test split per stage (drift baked in at cache time)."""
        items = self._test_items()
        return {stage: items for stage in range(schedule.n_stages)}
