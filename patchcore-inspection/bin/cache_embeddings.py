"""Run the frozen PatchCore backbone once and cache patch embeddings.

This is the only GPU-bound step of the streaming pipeline. It writes a drift-
ordered normal stream cache plus one labeled test-set cache per drift stage,
after which the entire RL/benchmark stack operates on numpy memmaps and never
touches images or the backbone again.

Example (cluster)::

    python bin/cache_embeddings.py \
        --backbone_name wideresnet50 -le layer2 -le layer3 \
        --data_path /home/user1/aniket/Patchcore/dataset/kvasir_patchcore \
        --classname kvasir_patchcore --dataset_kind mvtec \
        --drift staged_abrupt_4 --seed 0 \
        --out_dir cache/kvasir/wrn50/staged_abrupt_4
"""
import logging
import os
import subprocess
import sys

import click
import numpy as np
import PIL.Image
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.streaming import drift as drift_mod
from patchcore.streaming.cache import EmbeddingCacheWriter
from patchcore.streaming.stream import StreamBuilder

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def build_patchcore(
    backbone_name, layers, device, imagesize, pre_dim, target_dim, patchsize=3
):
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, None
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=list(layers),
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=pre_dim,
        target_embed_dimension=target_dim,
        patchsize=patchsize,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    return model


def _base_transforms(resize, imagesize):
    return transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(imagesize)])


def _to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def _embed_image(model, pil_img, to_tensor, device):
    tensor = to_tensor(pil_img).unsqueeze(0).to(torch.float).to(device)
    with torch.no_grad():
        feats = model._embed(tensor)  # [P, D] numpy
    return np.asarray(feats, dtype=np.float32)


def _probe_dims(model, resize, imagesize, to_tensor, device):
    dummy = PIL.Image.new("RGB", (resize, resize))
    dummy = _base_transforms(resize, imagesize)(dummy)
    feats = _embed_image(model, dummy, to_tensor, device)
    p, d = feats.shape
    # patch grid is square for the WRN/ViT configs used here
    side = int(round(p ** 0.5))
    return p, d, (side, side)


@click.command()
@click.option("--backbone_name", required=True)
@click.option("--layers_to_extract_from", "-le", multiple=True, required=True)
@click.option("--data_path", required=True)
@click.option("--classname", required=True)
@click.option("--dataset_kind", default="mvtec")
@click.option("--drift", "drift_name", default="staged_abrupt_4")
@click.option("--drift_mode", type=click.Choice(["synthetic", "real"]), default="synthetic")
@click.option("--seed", type=int, default=0)
@click.option("--resize", type=int, default=256)
@click.option("--imagesize", type=int, default=224)
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--gpu", type=int, default=0)
@click.option("--out_dir", required=True)
def main(
    backbone_name,
    layers_to_extract_from,
    data_path,
    classname,
    dataset_kind,
    drift_name,
    drift_mode,
    seed,
    resize,
    imagesize,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    gpu,
    out_dir,
):
    logging.basicConfig(level=logging.INFO)
    device = patchcore.utils.set_torch_device([gpu] if torch.cuda.is_available() else [])
    model = build_patchcore(
        backbone_name, layers_to_extract_from, device, imagesize,
        pretrain_embed_dimension, target_embed_dimension, patchsize,
    )
    base_tf = _base_transforms(resize, imagesize)
    to_tensor = _to_tensor()
    P, D, patch_shape = _probe_dims(model, resize, imagesize, to_tensor, device)
    LOGGER.info("patches/image=%d dim=%d patch_shape=%s", P, D, patch_shape)

    schedule = drift_mod.SCHEDULES[drift_name]
    builder = StreamBuilder(data_path, classname, resize=resize, imagesize=imagesize)

    meta = {
        "backbone": backbone_name,
        "layers": list(layers_to_extract_from),
        "target_embed_dimension": target_embed_dimension,
        "patchsize": patchsize,
        "patch_shape": list(patch_shape),
        "imagesize": imagesize,
        "drift": drift_name,
        "drift_mode": drift_mode,
        "seed": seed,
        "git_sha": _git_sha(),
    }

    # ---- stream ----
    if drift_mode == "synthetic":
        entries = builder.synthetic_drift_stream(schedule, seed=seed)
    else:
        entries = builder.real_drift_stream()
    stream_dir = os.path.join(out_dir, "stream")
    writer = EmbeddingCacheWriter(
        stream_dir, len(entries), P, D,
        meta={**meta, "stage_ids": [e.stage_id for e in entries]},
    )
    for i, e in enumerate(entries):
        img = PIL.Image.open(e.image_path).convert("RGB")
        img = base_tf(img)
        if drift_mode == "synthetic":
            tf = schedule.transform_at(e.stream_pos, len(entries), base_seed=seed)
            img = tf(img, image_key=os.path.basename(e.image_path))
        writer.write(i, _embed_image(model, img, to_tensor, device))
        if (i + 1) % 200 == 0:
            LOGGER.info("stream %d/%d", i + 1, len(entries))
    writer.finalize()
    LOGGER.info("wrote stream cache -> %s", stream_dir)

    # ---- per-stage test sets ----
    manifests = builder.per_stage_test_manifests(schedule)
    for stage, items in manifests.items():
        tdir = os.path.join(out_dir, "test", f"stage_{stage}")
        tw = EmbeddingCacheWriter(
            tdir, len(items), P, D,
            meta={**meta, "stage": stage, "stage_ids": [stage] * len(items)},
        )
        labels, masks = [], []
        tf = schedule.stage_transform(stage, base_seed=seed)
        for j, item in enumerate(items):
            img = base_tf(PIL.Image.open(item.image_path).convert("RGB"))
            if drift_mode == "synthetic":
                img = tf(img, image_key=os.path.basename(item.image_path))
            tw.write(j, _embed_image(model, img, to_tensor, device))
            labels.append(item.label)
            masks.append(_load_mask(item.mask_path, resize, imagesize))
        tw.write_labels(labels)
        tw.write_masks(np.stack(masks))
        tw.finalize()
        LOGGER.info("wrote test stage %d (%d imgs) -> %s", stage, len(items), tdir)


def _load_mask(mask_path, resize, imagesize):
    if mask_path is None or not os.path.exists(mask_path):
        return np.zeros((imagesize, imagesize), dtype=np.uint8)
    m = PIL.Image.open(mask_path).convert("L")
    m = transforms.CenterCrop(imagesize)(transforms.Resize(resize)(m))
    return (np.asarray(m) > 0).astype(np.uint8)


if __name__ == "__main__":
    main()
