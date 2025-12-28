import os
from dataclasses import dataclass

import torch
import json
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_sft

from optimum.quanto import requantize

from .model import Flux, FluxParams
from .controlnet import ControlNetFlux
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder
from .annotator.dwpose import DWposeDetector
from .annotator.mlsd import MLSDdetector
from .annotator.canny import CannyDetector
from .annotator.midas import MidasDetector
from .annotator.hed import HEDdetector
from .annotator.tile import TileDetector
from .annotator.zoe import ZoeDetector
from .annotator.midas import DPTDetector
import torch.nn as nn

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path, repo_id, name):
    print(local_path)
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    x=x.astype(np.uint8)
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
#Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

class Annotator:
    def __init__(self, name: str, device: str):
        if name == "canny":
            processor = CannyDetector()
        elif name == "openpose":
            processor = DWposeDetector(device)
        elif name == "depth":
            processor = MidasDetector()
            # processor = DPTDetector()
        elif name == "hed":
            processor = HEDdetector()
        elif name == "hough":
            processor = MLSDdetector()
        elif name == "tile":
            processor = TileDetector()
        elif name == "zoe":
            processor = ZoeDetector()
        elif name == 'mixed':
            processor = DPTDetector()
        self.name = name
        self.processor = processor

    def __call__(self, image: Image, width: int, height: int):
        image = np.array(image)
        detect_resolution = max(width, height)
        image, remove_pad = resize_image_with_pad(image, detect_resolution)

        image = np.array(image)
        if self.name == "canny":
            result = self.processor(image, low_threshold=100, high_threshold=200)
        elif self.name == "hough":
            result = self.processor(image, thr_v=0.05, thr_d=5)
        elif self.name == "depth":
            result = self.processor(image)
            result, _ = result
        else:
            result = self.processor(image)
        result = HWC3(remove_pad(result))
        result = cv2.resize(result, (width, height))
        return result


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path='/share/project/cwm/mingju.gao/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/flux1-dev.safetensors',
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "hoi": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=320,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="XLabs-AI/flux-dev-fp8",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux-dev-fp8.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model

def load_condition_flow(path,device):
    # Loading Flux
    print("Init model")
    model = Flux(configs['hoi'].params).to(torch.bfloat16)
    sd = torch.load(path)
    model.load_state_dict(sd, strict=False, assign=True)
    return model

def load_flow_model2(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):  
        print("Try to download flow model")
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))
    print(f"Load flow from {ckpt_path}")
    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model

def extend_image_in(model):
    old_layer:nn.Linear = model.img_in
    layer_in=old_layer.in_features
    layer_out=old_layer.out_features
    new_layer = nn.Linear(in_features=5*layer_in,out_features=layer_out)
    # 复制原始权重
    with torch.no_grad():
        new_layer.weight.zero_()  # 先将新层的权重设为 0
        new_layer.weight[:, :layer_in] = old_layer.weight
        new_layer.bias.copy_(old_layer.bias)  # 复制偏置项
    del old_layer
    model.img_in=new_layer
def load_flow_model_extend(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)
    
    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    extend_image_in(model)
    return model

def load_flow_model_quintized(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    json_path = hf_hub_download(configs[name].repo_id, 'flux_dev_quantization_map.json')


    model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device='cpu')
    with open(json_path, "r") as f:
        quantization_map = json.load(f)
    print("Start a quantization process...")
    requantize(model, sd, quantization_map, device=device)
    print("Model is quantized!")
    return model

def load_controlnet(name, device, transformer=None,depth:int=2):
    with torch.device(device):
        controlnet = ControlNetFlux(configs[name].params,controlnet_depth=depth)
        print("ControlNetFlux loaded")
        print(configs[name].params)
    if transformer is not None:
        controlnet.load_state_dict(transformer.state_dict(), strict=False)
    return controlnet
def expand_first_conv(model, extra_channels):
    """
    将 model.input_hint_block 中的第一个 Conv2d 层的输入通道数扩展，并用 0 填充新通道的权重。
    
    参数：
    - model: ControlNetFlux 模型
    - extra_channels: 需要增加的输入通道数
    """
    # 获取 input_hint_block 的第一个 Conv2d 层
    old_conv = model.input_hint_block[0]
    
    # 计算新的输入通道数
    new_in_channels = old_conv.in_channels + extra_channels
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    bias = old_conv.bias is not None  # 检查是否有 bias
    
    # 创建新的 Conv2d 层，增加输入通道
    new_conv = nn.Conv2d(new_in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    
    # 复制原始权重
    with torch.no_grad():
        new_conv.weight.zero_()  # 先将新层的权重设为 0
        new_conv.weight[:, :old_conv.in_channels, :, :] = old_conv.weight  # 复制原来的权重
        
        if bias:
            new_conv.bias.copy_(old_conv.bias)  # 复制偏置项
    # 替换原始 Sequential 中的第一个卷积层
    model.input_hint_block[0] = new_conv
def expand_first_conv_average(model, extra_channels):
    """
    将 model.input_hint_block 中的第一个 Conv2d 层的输入通道数扩展，并用 0 填充新通道的权重。
    
    参数：
    - model: ControlNetFlux 模型
    - extra_channels: 需要增加的输入通道数
    """
    # 获取 input_hint_block 的第一个 Conv2d 层
    old_conv = model.input_hint_block[0]
    
    # 计算新的输入通道数
    new_in_channels = old_conv.in_channels + extra_channels
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    bias = old_conv.bias is not None  # 检查是否有 bias
    
    # 创建新的 Conv2d 层，增加输入通道
    new_conv = nn.Conv2d(new_in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    # 复制原始权重
    with torch.no_grad():
        new_conv.weight.zero_()  # 先将新层的权重设为 0
        for k in range(new_in_channels//old_conv.in_channels):
            new_conv.weight[:, k*old_conv.in_channels:(k+1)*old_conv.in_channels, :, :] = old_conv.weight*(old_conv.in_channels/new_in_channels)  
        if bias:
            new_conv.bias.copy_(old_conv.bias)  # 复制偏置项
    # 替换原始 Sequential 中的第一个卷积层
    model.input_hint_block[0] = new_conv
    
def load_controlnet_extend(name, device, transformer=None, condition_in_channels: int = 3, depth=2):
    """
    加载 ControlNetFlux, 并对需要调整通道数的线性层进行零扩展。
    
    Args:
        name (str): 模型名称，用于从 configs 获取参数。
        device: 加载模型的设备。
        transformer (nn.Module, optional): 预训练的 Transformer 模型，用于加载权重。
        new_in_channels (int, optional): 新的输入通道数。如果未提供，则保持默认。
    
    Returns:
        ControlNetFlux: 修改后的 ControlNet 模型。
    """
    # 获取原始配置参数
    params = configs[name].params
    
    # 在指定设备上初始化模型
    with torch.device(device):
        controlnet = ControlNetFlux(params, controlnet_depth=depth, condition_in_channels=condition_in_channels)
    
    # 如果提供了预训练的 transformer 模型，加载权重并进行通道扩展
    if transformer is not None:
        # 获取预训练模型的状态字典
        if isinstance(transformer, dict):
            pretrained_state_dict = transformer
        else:
            pretrained_state_dict = transformer.state_dict()
        # 加载修改后的状态字典
        controlnet.load_state_dict(pretrained_state_dict, strict=False)
    # expand_first_conv_average(controlnet, condition_in_channels - 3)
        
    return controlnet

def load_controlnet_trained(name, device, transformer=None, condition_in_channels: int = 3,depth=2):
    """
    加载 ControlNetFlux，并对需要调整通道数的线性层进行零扩展。
    
    Args:
        name (str): 模型名称，用于从 configs 获取参数。
        device: 加载模型的设备。
        transformer (nn.Module, optional): 预训练的 Transformer 模型，用于加载权重。
        new_in_channels (int, optional): 新的输入通道数。如果未提供，则保持默认。
    
    Returns:
        ControlNetFlux: 修改后的 ControlNet 模型。
    """
    # 获取原始配置参数
    params = configs[name].params
    
    # 在指定设备上初始化模型
    with torch.device(device):
        controlnet = ControlNetFlux(params,controlnet_depth=depth,condition_in_channels=condition_in_channels)
    
    # 如果提供了预训练的 transformer 模型，加载权重并进行通道扩展
    if transformer is not None:
        # 获取预训练模型的状态字典
        if isinstance(transformer, dict):
            pretrained_state_dict = transformer
        else:
            pretrained_state_dict = transformer.state_dict()
        # 加载修改后的状态字典
        controlnet.load_state_dict(pretrained_state_dict, strict=False)
    return controlnet
def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    # 支持通过环境变量控制 Hugging Face 下载与缓存行为：
    #   HF_TOKEN - 如果模型是私有或需要 token，可设置该环境变量
    #   HF_CACHE - 指定 huggingface 缓存目录（会传给 `cache_dir`）
    # Prepare hf kwargs: only support HF_CACHE; explicitly disable HF_TOKEN and HF_EXTRA_ARGS
    # 强制使用指定的 HF_HOME 路径
    os.environ["HF_HOME"] = "/share/project/cwm/mingju.gao/.cache/huggingface"
    os.environ['HF_CACHE'] = "/share/project/cwm/mingju.gao/.cache/huggingface/hub"
    print("Loading T5 text encoder...")
    return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    # 同样支持通过 HF_TOKEN/HF_CACHE/HF_EXTRA_ARGS 控制下载和缓存
    # Only support HF_CACHE; disable HF_TOKEN and HF_EXTRA_ARGS
    os.environ["HF_HOME"] = "/share/project/cwm/mingju.gao/.cache/huggingface"
    os.environ['HF_CACHE'] = "/share/project/cwm/mingju.gao/.cache/huggingface/hub"
    print("Loading CLIP image encoder...")
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        # ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)
        ckpt_path = '/share/project/cwm/mingju.gao/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/ae.safetensors'

    # Loading the autoencoder
    print(f"Init AE from {ckpt_path}")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# A fixed 48-bit message that was choosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
