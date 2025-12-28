from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor
import cv2
from einops import rearrange
import uuid
import os

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, prepare_single, unpack,denoise_controlnet_mix,denoise_full_control,denoise_double_control,denoise_single_control
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_condition_flow,
    load_controlnet_extend,
    load_controlnet_trained,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)
from src.dataset import get_transform,get_resize

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from PIL import Image

def debug_save_image(image, filename):
    image = (image + 1.0) * 127.5
    image = image.clamp(0, 255).to(torch.uint8)
    image = image[0]
    image = image.cpu().permute(1, 2, 0).numpy()
    image = Image.fromarray(image)
    image.save(filename)


class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False, model_path:str=None):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type
        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if not model_path:
            if "fp8" in model_type:
                self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
            else:
                self.model = load_flow_model(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_condition_flow(path=model_path,device=device)
        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.transform= get_transform()
        self.ip_loaded = False

    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)

        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)
        print("Loading controlnet from:", local_path, repo_id, name)
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type
    
    def set_controlnet_extend(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None,condition_in_channels :int = 3):
        """用于加载修改了输入维度的controlnet"""
        self.model.to(self.device)
        self.controlnet = load_controlnet_extend(self.model_type, self.device,condition_in_channels=condition_in_channels).to(torch.bfloat16)
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type

    def set_controlnet_by_offer(self, control_type: str, controlnet: torch.nn.Module):
        """用于直接指定controlnet"""
        self.model.to(self.device)
        self.controlnet = controlnet.to(torch.bfloat16)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type
    
    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values
        image_prompt = image_prompt.to(self.image_encoder.device)
        image_prompt_embeds = self.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=self.device, dtype=torch.bfloat16,
        )
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj

    def __call__(self,
                 prompt: str,
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        if not (image_prompt is None and neg_image_prompt is None) :
            assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

            if image_prompt is None:
                image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

            image_proj = self.get_image_proj(image_prompt)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.controlnet_loaded:
            if isinstance(controlnet_image, Image.Image):
                depth = self.transform(controlnet_image).unsqueeze(0).to(self.device).to(torch.bfloat16)
                controlnet_image = depth
            if isinstance(controlnet_image,list):
                self.transform = get_resize(height,width)
                depth,normal,hand=controlnet_image
                depth=self.transform(depth).unsqueeze(0).to(self.device)
                normal=self.transform(normal).unsqueeze(0).to(self.device)
                hand=self.transform(hand).unsqueeze(0).to(self.device)
                controlnet_image = [depth,normal,hand]
            

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
        )
    
    # @torch.inference_mode()
    # def infer_data(self,data:dict,seed=12345,dtype=torch.float32):
    #     with torch.autocast(device_type='cuda',dtype=dtype):
    #         prompt=data['prompt']+"\nGenerate a static image, ensuring the hands are clear and complete\n"
    #         width=data['video_metadata']['width']
    #         height=data['video_metadata']['height']
    #         depth=data['depth_map'].clone().to(self.device).unsqueeze(0)
    #         normal=data['normal_map'].clone().to(self.device).unsqueeze(0)
    #         hand=data['hand_keypoints'].clone().to(self.device).unsqueeze(0)
    #         seg=data['seg_mask'].clone().to(self.device).unsqueeze(0)
    #         guidance=4.0
    #         num_steps=28
    #         # seed=12345
    #         timesteps=get_schedule(num_steps,(width//8)*(height//8)//(16*16),shift=True)
    #         x=get_noise(1,height,width,device=self.device,dtype=torch.float32,seed=seed)

    #         with torch.no_grad():
    #             inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
    #         x0 = denoise_controlnet_mix(
    #                         self.model,
    #                         **inp_cond,
    #                         controlnet=self.controlnet,
    #                         timesteps=timesteps,
    #                         guidance=guidance,
    #                         controlnet_cond=(depth,normal,hand,seg),
    #                     )
    #         x0 = unpack(x0.float(), height, width)
    #         x0 = self.ae.decode(x0)
    #         x1 = x0.clamp(-1, 1)

    #         x1 = rearrange(x1[-1], "c h w -> h w c")
    #         x1 = (127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)

    #         depth=((data['depth_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
    #         normal=((data['normal_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
    #         hand=((data['hand_keypoints'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
    #         seg=((data['seg_mask'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
    #         img=((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)

    #         alpha = 0.5  # 设定混合比例
    #         depth_blended = cv2.addWeighted(img, alpha, depth, 1 - alpha, 0)
    #         normal_blended =cv2.addWeighted(img, alpha, normal, 1 - alpha, 0)
    #         hand_blended =cv2.addWeighted(img, alpha, hand, 1 - alpha, 0)
    #         seg_blended =cv2.addWeighted(img, alpha, seg, 1 - alpha, 0)
    #         cat_img=np.concatenate([x1,img,depth_blended,normal_blended,hand_blended,seg_blended],axis=1)
    #     return cat_img
    
    def infer_data(self, data:dict, seed=12345, dtype=torch.float32):
        with torch.autocast(device_type='cuda',dtype=dtype):
            prompt = data['prompt']
            width = data['video_metadata']['width']
            height = data['video_metadata']['height']
            seg = data['seg_mask'].clone().to(self.device).unsqueeze(0)
            depth = data["depth_map"].clone().to(self.device).unsqueeze(0)
            normal = data["normal_map"].clone().to(self.device).unsqueeze(0)
            hand = data["hand_keypoints"].clone().to(self.device).unsqueeze(0)
            guidance = 4.0
            num_steps = 28
            timesteps = get_schedule(num_steps, (width // 8) * (height // 8) // (16 * 16), shift=True)
            x = get_noise(1, height, width, device=self.device, dtype=torch.float32, seed=seed)
            
            with torch.no_grad():
                inp_cond = prepare_single(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

            x0 = denoise_controlnet_mix(
                            self.model,
                            **inp_cond,
                            controlnet=self.controlnet,
                            timesteps=timesteps,
                            guidance=guidance,
                            controlnet_cond=(depth,normal,hand,seg),
                        )
            
            x0 = unpack(x0.float(), height, width)
            x0 = self.ae.decode(x0)
            x1 = x0.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            x1 = (127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)

            depth = ((data['depth_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            normal = ((data['normal_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            hand = ((data['hand_keypoints'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            seg = ((data['seg_mask'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            img = ((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)

            alpha = 0.5  # 设定混合比例
            depth_blended = cv2.addWeighted(x1, alpha, depth, 1 - alpha, 0)
            normal_blended =cv2.addWeighted(x1, alpha, normal, 1 - alpha, 0)
            hand_blended = cv2.addWeighted(x1, alpha, hand, 1 - alpha, 0)
            seg_blended = cv2.addWeighted(x1, alpha, seg, 1 - alpha, 0)

            cat_img = np.concatenate([x1, img, seg_blended, depth_blended, 
                                      normal_blended, hand_blended], axis=1)
            
        return cat_img
    
    def infer_data_naive(self,data:dict,seed=12345,dtype=torch.float32):
        with torch.autocast(device_type='cuda',dtype=dtype):
            self.ae.to(self.device)
            self.model.to(self.device)
            self.t5.to(self.device)
            self.clip.to(self.device)
            
            prompt=data['prompt']
            width=data['video_metadata']['width']
            height=data['video_metadata']['height']
            depth=data['depth_map'].clone().to(self.device).unsqueeze(0)
            normal=data['normal_map'].clone().to(self.device).unsqueeze(0)
            hand=data['hand_keypoints'].clone().to(self.device).unsqueeze(0)
            seg=data['seg_mask'].clone().to(self.device).unsqueeze(0)
            guidance=4.0
            num_steps=28
            # seed=12345
            timesteps=get_schedule(num_steps,(width//8)*(height//8)//(16*16),shift=True)
            x= get_noise(1,height,width,device=self.device,dtype=torch.float32,seed=seed)
            with torch.no_grad():
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
                
            def encode2lat(x):
                with torch.no_grad():
                    x = self.ae.encode(x.to(self.device).to(dtype))
                    x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                    return x
                    
            controlnet_cond=(encode2lat(depth),encode2lat(normal),encode2lat(hand),encode2lat(seg))
            depth = rearrange(depth, "b (h w) (c ph pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            
            x0 = denoise_full_control(
                            self.model,
                            **inp_cond,
                            timesteps=timesteps,
                            guidance=guidance,
                            controlnet_cond=controlnet_cond,
                        )
            x0 = unpack(x0.float(), height, width)
            x0=self.ae.decode(x0)
            x1 = x0.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            x1=(127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)
            depth=((data['depth_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            normal=((data['normal_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            hand=((data['hand_keypoints'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            seg=((data['seg_mask'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            img=((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            alpha = 0.5  # 设定混合比例
            depth_blended = cv2.addWeighted(img, alpha, depth, 1 - alpha, 0)
            normal_blended =cv2.addWeighted(img, alpha, normal, 1 - alpha, 0)
            hand_blended =cv2.addWeighted(img, alpha, hand, 1 - alpha, 0)
            seg_blended =cv2.addWeighted(img, alpha, seg, 1 - alpha, 0)
            cat_img=np.concatenate([x1,img,depth_blended,normal_blended,hand_blended,seg_blended],axis=1)
        return cat_img
    
    def infer_data_save(self,data:dict):
        self.ae.to(dtype=torch.bfloat16,device=self.device)
        self.model.to(dtype=torch.bfloat16,device=self.device)
        self.controlnet.to(dtype=torch.bfloat16,device=self.device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            prompt=data['prompt']+"\nGenerate a static image, ensuring the hands are clear and complete\n"
            width=data['video_metadata']['width']
            height=data['video_metadata']['height']
            depth=data['depth_map'].clone().to(self.device).unsqueeze(0)
            normal=data['normal_map'].clone().to(self.device).unsqueeze(0)
            hand=data['hand_keypoints'].clone().to(self.device).unsqueeze(0)
            seg=data['seg_mask'].clone().to(self.device).unsqueeze(0)
            
            guidance=4.0
            num_steps=14
            seed=12345
            timesteps=get_schedule(num_steps,(width//8)*(height//8)//(16*16),shift=True)
            x=get_noise(1,height,width,device=self.device,dtype=torch.float32,seed=seed)
            with torch.no_grad():
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            x0 = denoise_controlnet_mix(
                            self.model,
                            **inp_cond,
                            controlnet=self.controlnet,
                            timesteps=timesteps,
                            guidance=guidance,
                            controlnet_cond=(depth,normal,hand,seg),
                        )
            x0 = unpack(x0.float(), height, width)
            x0=self.ae.decode(x0)
            x1 = x0.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            x1=(127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)
            img=((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            return x1,img
    
    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)
        
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            if self.controlnet_loaded:
                if isinstance(controlnet_image,list):
                    x = denoise_controlnet_mix(
                        self.model,
                        **inp_cond,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_image,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt'],
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec'],
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        image_proj=image_proj,
                        neg_image_proj=neg_image_proj,
                        ip_scale=ip_scale,
                        neg_ip_scale=neg_ip_scale,
                    )
                else:
                    x = denoise_controlnet(
                        self.model,
                        **inp_cond,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_image,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt'],
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec'],
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        image_proj=image_proj,
                        neg_image_proj=neg_image_proj,
                        ip_scale=ip_scale,
                        neg_ip_scale=neg_ip_scale,
                    )
            else:
                x = denoise(
                    self.model,
                    **inp_cond,
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


class XFluxSampler(XFluxPipeline):
    def __init__(self, clip, t5, ae, model, control_net, device):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.controlnet=control_net
        if control_net:
            self.controlnet_loaded = True
        self.ip_loaded = False
        self.offload = False



class DoubleControlSampler(XFluxPipeline):
    def __init__(self, clip, t5, ae, model, control_net1,control_net2,device):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.controlnet1 = control_net1
        self.controlnet2 = control_net2
        self.ip_loaded = False
        self.offload = False
    
    def infer_data(self, data:dict, seed=12345, dtype=torch.float32, control_gs:tuple[float]=(1,1,0,0), is_reg=False):
        with torch.autocast(device_type='cuda',dtype=dtype),torch.no_grad():
            prompt = data['prompt']
            width = data['video_metadata']['width']
            height = data['video_metadata']['height']
            depth = data['depth_map'].clone().to(self.device).unsqueeze(0)
            normal = data['normal_map'].clone().to(self.device).unsqueeze(0)
            hand = data['hand_keypoints'].clone().to(self.device).unsqueeze(0)
            seg = data['seg_mask'].clone().to(self.device).unsqueeze(0)
            
            if is_reg:
                prompt = data['reg_prompt']
                depth = -torch.ones_like(depth).to(self.device)
                normal = -torch.ones_like(normal).to(self.device)
                hand = -torch.ones_like(hand).to(self.device)
                seg = -torch.ones_like(seg).to(self.device)
            
            guidance = 4.0
            num_steps = 28
            # seed=12345
            timesteps = get_schedule(num_steps,(width//8)*(height//8)//(16*16),shift=True)
            x = get_noise(1,height,width,device=self.device,dtype=torch.float32,seed=seed)
            with torch.no_grad():
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            with torch.no_grad():
                inp_cond_default= prepare(t5=self.t5, clip=self.clip, img=x, prompt="In the center of the image, a man stands confidently, wearing a black t-shirt and a blue surgical mask. He gazes towards a table in front of him. The background features a high-tech laboratory filled with scientific equipment, computers, and glowing screens, suggesting a futuristic and sterile environment.")
            x0 = denoise_double_control(
                            self.model,
                            inp_condition_true=inp_cond,
                            inp_condition_default=inp_cond_default,
                            controlnet1=self.controlnet1,
                            controlnet2=self.controlnet2,
                            timesteps=timesteps,
                            guidance=guidance,
                            controlnet_cond=(depth,normal,hand,seg),
                            control_gs=control_gs
                        )
            x0 = unpack(x0.float(), height, width)
            x0=self.ae.decode(x0)
            x1 = x0.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            x1=(127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)
            depth=((data['depth_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            normal=((data['normal_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            hand=((data['hand_keypoints'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            seg=((data['seg_mask'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            img=((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            alpha = 0.5  # 设定混合比例
            depth_blended = cv2.addWeighted(x1, alpha, depth, 1 - alpha, 0)
            normal_blended =cv2.addWeighted(x1, alpha, normal, 1 - alpha, 0)
            hand_blended =cv2.addWeighted(x1, alpha, hand, 1 - alpha, 0)
            seg_blended =cv2.addWeighted(x1, alpha, seg, 1 - alpha, 0)
            cat_img=np.concatenate([x1,img,depth_blended,normal_blended,hand_blended,seg_blended],axis=1)
            
        return cat_img

        
class DoubleControlPipeline(XFluxPipeline):
    def __init__(self, control_net,**kwargs):
        super().__init__(**kwargs)
        contronet_model=load_checkpoint(control_net,None,None)
        control1={k.replace("controlnet1.",""):v for k,v in contronet_model.items() if k.startswith("controlnet1.")}
        control2={k.replace("controlnet2.",""):v for k,v in contronet_model.items() if k.startswith("controlnet2.")}
        self.controlnet1 = load_controlnet_trained('flux-dev',self.device, control1, 6, 2)
        self.controlnet2 = load_controlnet_trained('flux-dev',self.device, control2, 6, 2)
        self.controlnet_loaded = True
        self.ip_loaded = False
        self.offload = False
    
    def infer_data(self,data:dict, seed=12345, dtype=torch.float32, control_gs=(1.0,1.0,0.0,0.0), device='cuda'):
        with torch.autocast(device_type=device, dtype=dtype),torch.no_grad():
            self.ae.to(self.device)
            self.model.to(self.device)
            self.controlnet1.to(self.device)
            self.controlnet2.to(self.device)
            self.t5.to(self.device)
            self.clip.to(self.device)
            prompt=data['prompt']
            width=data['video_metadata']['width']
            height=data['video_metadata']['height']
            depth=data['depth_map'].clone().to(self.device).unsqueeze(0)
            normal=data['normal_map'].clone().to(self.device).unsqueeze(0)
            hand=data['hand_keypoints'].clone().to(self.device).unsqueeze(0)
            seg=data['seg_mask'].clone().to(self.device).unsqueeze(0)
            guidance=4.0
            num_steps=28
            # seed=12345
            timesteps = get_schedule(num_steps, (width//8)*(height//8)//(16*16), shift=True)
            x = get_noise(1, height,width, device=self.device, dtype=torch.float32, seed=seed)
            with torch.no_grad():
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            with torch.no_grad():
                inp_cond_default = prepare(t5=self.t5,clip=self.clip,img=x,prompt="In the center of the image, a man stands confidently, wearing a black t-shirt and a blue surgical mask. He gazes towards a table in front of him. The background features a high-tech laboratory filled with scientific equipment, computers, and glowing screens, suggesting a futuristic and sterile environment.")
            x0 = denoise_double_control(
                            self.model,
                            inp_condition_true=inp_cond,
                            inp_condition_default=inp_cond_default,
                            controlnet1=self.controlnet1,
                            controlnet2=self.controlnet2,
                            timesteps=timesteps,
                            guidance=guidance,
                            controlnet_cond=(depth,normal,hand,seg),
                            control_gs=control_gs
                        )
            x0 = unpack(x0.float(), height, width)
            x0=self.ae.decode(x0)
            x1 = x0.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            x1=(127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)
            
            if len(data['depth_map'].shape)==4:
                data['depth_map']=data['depth_map'].squeeze(0)
            if len(data['normal_map'].shape)==4:
                data['normal_map']=data['normal_map'].squeeze(0)
            if len(data['hand_keypoints'].shape)==4:
                data['hand_keypoints']=data['hand_keypoints'].squeeze(0)
            if len(data['seg_mask'].shape)==4:
                data['seg_mask'] = data['seg_mask'].squeeze(0)
            if len(data['video'].shape)==4:
                data['video'] = data['video'].squeeze(0) 
            
            depth=((data['depth_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            normal=((data['normal_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            hand=((data['hand_keypoints'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            seg=((data['seg_mask'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            img=((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            alpha = 0.5  # 设定混合比例
            depth_blended = cv2.addWeighted(x1, alpha, depth, 1 - alpha, 0)
            normal_blended =cv2.addWeighted(x1, alpha, normal, 1 - alpha, 0)
            hand_blended =cv2.addWeighted(x1, alpha, hand, 1 - alpha, 0)
            seg_blended =cv2.addWeighted(x1, alpha, seg, 1 - alpha, 0)
            cat_img=np.concatenate([x1,img,depth_blended,normal_blended,hand_blended,seg_blended],axis=1)
        return cat_img
    
class SingleControlPipeline(XFluxPipeline):
    def __init__(self, control_net,**kwargs):
        super().__init__(**kwargs)
        controlnet_model= load_checkpoint(control_net,None,None)
        self.controlnet = load_controlnet_trained('flux-dev','cuda', controlnet_model, condition_in_channels=12)
        self.controlnet_loaded = True
        self.ip_loaded = False
        self.offload = False
    
    def infer_data(self,data:dict,seed=12345,dtype=torch.float32, **kwargs):
        with torch.autocast(device_type='cuda',dtype=dtype),torch.no_grad():
            self.ae.to('cuda')
            self.model.to('cuda')
            self.controlnet.to('cuda')
            self.t5.to('cuda')
            self.clip.to('cuda')
            prompt=data['prompt']
            width=data['video_metadata']['width']
            height=data['video_metadata']['height']
            depth=data['depth_map'].clone().to(self.device).unsqueeze(0)
            normal=data['normal_map'].clone().to(self.device).unsqueeze(0)
            hand=data['hand_keypoints'].clone().to(self.device).unsqueeze(0)
            seg=data['seg_mask'].clone().to(self.device).unsqueeze(0)
            guidance=4.0
            num_steps=28
            # seed=12345
            timesteps=get_schedule(num_steps,(width//8)*(height//8)//(16*16),shift=True)
            x=get_noise(1,height,width,device=self.device,dtype=torch.float32,seed=seed)
            with torch.no_grad():
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            x0 = denoise_single_control(
                            self.model,
                            **inp_cond,
                            controlnet=self.controlnet,
                            timesteps=timesteps,
                            guidance=guidance,
                            controlnet_cond=(depth,normal,hand,seg),
                            **kwargs
                        )
            x0 = unpack(x0.float(), height, width)
            x0=self.ae.decode(x0)
            x1 = x0.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            x1=(127.5 * (x1 + 1.0)).cpu().byte().numpy().astype(np.uint8)
            depth=((data['depth_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            normal=((data['normal_map'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            hand=((data['hand_keypoints'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            seg=((data['seg_mask'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            img=((data['video'].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
            alpha = 0.5  # 设定混合比例
            depth_blended = cv2.addWeighted(x1, alpha, depth, 1 - alpha, 0)
            normal_blended =cv2.addWeighted(x1, alpha, normal, 1 - alpha, 0)
            hand_blended =cv2.addWeighted(x1, alpha, hand, 1 - alpha, 0)
            seg_blended =cv2.addWeighted(x1, alpha, seg, 1 - alpha, 0)
            cat_img=np.concatenate([x1,img,depth_blended,normal,hand_blended,seg_blended],axis=1)
        return cat_img