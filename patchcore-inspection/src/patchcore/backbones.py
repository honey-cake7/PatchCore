import os

import timm  # noqa
import torchvision.models as models  # noqa
import torch

from patchcore.networks.pvtv2 import pvt_v2_b2


def load_gastronet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    
    weights_path = "/home/user1/aniket/Patchcore/PatchCore/models/gastronet.pth"
    state_dict = torch.load(weights_path, map_location=device)  # load directly to target device
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    
    state_dict = {
        k.replace("backbone.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()          # important — disables dropout/batchnorm training behavior
    return model




def _load_pvtv2_b2(weights_path):
    """Load a PVTv2-B2 backbone, auto-detecting the checkpoint's key prefix.

    Handles plain ImageNet weights (no prefix), Polyp-PVT weights ("backbone." prefix),
    and DataParallel weights ("module." prefix). Fails loudly if the checkpoint doesn't
    match the architecture, so we never silently load the wrong weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pvt_v2_b2()

    save_model = torch.load(weights_path, map_location=device)
    for key in ("model", "state_dict"):  # unwrap common checkpoint containers
        if isinstance(save_model, dict) and key in save_model:
            save_model = save_model[key]
            break

    model_dict = model.state_dict()
    # Pick the prefix-stripping that matches the most target keys.
    candidates = {
        "": save_model,
        "backbone.": {
            k[len("backbone."):]: v for k, v in save_model.items() if k.startswith("backbone.")
        },
        "module.": {
            k[len("module."):]: v for k, v in save_model.items() if k.startswith("module.")
        },
    }
    best = max(candidates.values(), key=lambda sd: sum(k in model_dict for k in sd))
    state_dict = {k: v for k, v in best.items() if k in model_dict}

    n_matched, n_total = len(state_dict), len(model_dict)
    print(f"[pvtv2] {weights_path}: matched {n_matched}/{n_total} backbone keys")
    assert n_matched > n_total * 0.8, (
        f"only {n_matched}/{n_total} keys matched — wrong checkpoint for pvt_v2_b2?"
    )

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model.to(device).eval()


def load_polyp_pvt():
    return _load_pvtv2_b2(os.environ.get("POLYP_PVT_WEIGHTS", "models/PolypPVT.pth"))


def load_pvtv2_b2():
    return _load_pvtv2_b2(os.environ.get("PVTV2_B2_WEIGHTS", "models/pvt_v2_b2.pth"))


def load_segformer_mit_b3():
    """SegFormer MiT-b3 (ImageNet) encoder, wrapped to expose its 4 pyramid maps.

    Channels [64, 128, 320, 512] at strides /4, /8, /16, /32 -> hook stages.1 / stages.2.
    """
    from transformers import SegformerModel  # lazy: optional dependency

    from patchcore.networks.feature_wrapper import MultiScaleWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerModel.from_pretrained("nvidia/mit-b3")

    def extract(m, x):
        return list(m(x, output_hidden_states=True).hidden_states)  # 4 x [B, C, H, W]

    return MultiScaleWrapper(model, extract).to(device).eval()


def load_mambavision_t():
    """NVIDIA MambaVision-T (ImageNet), wrapped to expose its 4 pyramid maps.

    Channels [80, 160, 320, 640] at strides /4, /8, /16, /32 -> hook stages.1 / stages.2.
    Requires CUDA (mamba_ssm / causal_conv1d) + einops; cannot run CPU-only.
    """
    from transformers import AutoModel  # lazy: optional dependency

    from patchcore.networks.feature_wrapper import MultiScaleWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)

    def extract(m, x):
        # The HF wrapper's forward returns (out_avg_pool, features) where features is the
        # list of 4 stage maps [B,C,H,W]; it has no `forward_features` method.
        return list(m(x)[1])

    return MultiScaleWrapper(model, extract).to(device).eval()


_BACKBONES = {
    "gastronet": "load_gastronet()",
    "polyp-pvt": "load_polyp_pvt()",
    "pvtv2_b2": "load_pvtv2_b2()",
    "segformer_mit_b3": "load_segformer_mit_b3()",
    "mambavision_t": "load_mambavision_t()",
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}

def load(name):
    return eval(_BACKBONES[name])

