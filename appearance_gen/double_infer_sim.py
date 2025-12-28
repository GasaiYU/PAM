import argparse
import os
import torch
from src.flux.xflux_pipeline import DoubleControlPipeline
import numpy as np
import cv2
from tqdm import tqdm

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    parser.add_argument(
        "--data_root", type=str, default=None, help="Path to wild data root"
    )
    return parser

    
def get_min_index(data_dir):
    index_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.png') and "rgb" in file:
            index_list.append(int(file.split('.')[0].split('_')[-1]))
    return min(index_list)


def get_valid_set(data_root, size=(480, 640)):
    data_dirs = os.listdir(data_root)
    data_dirs.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1]))
    for data_dir in data_dirs:
        data_dir = os.path.join(data_root, data_dir)

        start_index = get_min_index(data_dir)
        rgb_path = os.path.join(data_dir, f"rendered_rgb_{start_index}.png")
        normal_path = os.path.join(data_dir, f"rendered_normal_{start_index}.png")
        depth_path = os.path.join(data_dir, f"rendered_depth_{start_index}.png")
        seg_path = os.path.join(data_dir, f"rendered_seg_mask_{start_index}.png")
        hand_keypoints_path = os.path.join(data_dir, f"rendered_hand_keypoints_{start_index}.png")

        rgb_image = cv2.imread(rgb_path)
        normal_image = cv2.imread(normal_path)
        depth_image = cv2.imread(depth_path)
        seg_image = cv2.imread(seg_path)
        hand_keypoints_image = cv2.imread(hand_keypoints_path)

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
        hand_keypoints_image = cv2.cvtColor(hand_keypoints_image, cv2.COLOR_BGR2RGB)

        rgb_image = rgb_image.astype(np.float32) / 127.5 - 1.0
        normal_image = normal_image.astype(np.float32) / 127.5 - 1.0
        depth_image = depth_image.astype(np.float32) / 127.5 - 1.0
        seg_image = seg_image.astype(np.float32) / 127.5 - 1.0
        hand_keypoints_image = hand_keypoints_image.astype(np.float32) / 127.5 - 1.0

        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        normal_image = np.transpose(normal_image, (2, 0, 1))
        depth_image = np.transpose(depth_image, (2, 0, 1))
        seg_image = np.transpose(seg_image, (2, 0, 1))
        hand_keypoints_image = np.transpose(hand_keypoints_image, (2, 0, 1))

        rgb_image = torch.from_numpy(rgb_image).float()
        normal_image = torch.from_numpy(normal_image).float()
        depth_image = torch.from_numpy(depth_image).float()
        seg_image = torch.from_numpy(seg_image).float()
        hand_keypoints_image = torch.from_numpy(hand_keypoints_image).float()

        res = {}
        res['video'] = rgb_image
        res['normal_map'] = normal_image
        res['depth_map'] = depth_image
        res['seg_mask'] = seg_image
        res['hand_keypoints'] = hand_keypoints_image
        res['prompt'] = 'A hand is trying to grasp something'
        res['video_metadata'] = {
            'height': 480,
            'width': 640
        }
        yield res


def main(args):
    dataset = get_valid_set(data_root=args.data_root)
    with torch.no_grad():
        dataset = list(dataset)

        xflux_pipeline = DoubleControlPipeline(model_type='flux-dev', device='cuda', offload=True, control_net=args.local_path)
        os.makedirs(args.save_path, exist_ok=True)

        for i, data in tqdm(enumerate(dataset)):
            for j in range(30):
                print(f'Processing image {i}, seed {j}',f'save to {args.save_path}/test_result_{i}_{j}.png')
                result = xflux_pipeline.infer_data(data=data, seed=42+j, control_gs=(1, 1, 0, 0))
                cv2.imwrite(f'{args.save_path}/test_result_{i}_{j}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
