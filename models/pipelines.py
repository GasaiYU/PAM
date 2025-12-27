import os
import sys
import math
from tqdm import tqdm
from PIL import Image, ImageDraw
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.append(os.path.join(project_root, "submodules/MoGe"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print("Warning: MoGe not found, motion transfer will not be applied")
    
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import FluxControlPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking, CogVideoXTransformer3DModelTracking

from submodules.MoGe.moge.model import MoGeModel
from image_gen_aux import DepthPreprocessor
from moviepy.editor import ImageSequenceClip

class DiffusionAsShaderPipeline:
    def __init__(self, model_path, transformer_path, gpu_id=0, output_dir='outputs', dtype: torch.dtype = torch.bfloat16):
        """Initialize MotionTransfer class
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        # video parameters
        self.max_depth = 65.0
        self.fps = 8

        # camera parameters
        self.camera_motion=None
        self.fov=55

        # device
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        # files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])

        transformer = CogVideoXTransformer3DModelTracking.from_pretrained(transformer_path, torch_dtype=dtype)
        self.pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(model_path, torch_dtype=dtype, transformer=transformer)

    @torch.no_grad()
    def _infer(
        self, 
        prompt: str,
        tracking_tensor: torch.Tensor = None,
        image_tensor: torch.Tensor = None,  # [C,H,W] in range [0,1]
        output_path: str = "./output.mp4",
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        fps: int = 24,
        seed: int = 42,
    ):
        """
        Generates a video based on the given prompt and saves it to the specified path.

        Parameters:
        - prompt (str): The description of the video to be generated.
        - model_path (str): The path of the pre-trained model to be used.
        - tracking_tensor (torch.Tensor): Tracking video tensor [T, C, H, W] in range [0,1]
        - image_tensor (torch.Tensor): Input image tensor [C, H, W] in range [0,1]
        - output_path (str): The path where the generated video will be saved.
        - num_inference_steps (int): Number of steps for the inference process.
        - guidance_scale (float): The scale for classifier-free guidance.
        - num_videos_per_prompt (int): Number of videos to generate per prompt.
        - dtype (torch.dtype): The data type for computation.
        - seed (int): The seed for reproducibility.
        """
        pipe = self.pipe
        
        # Convert tensor to PIL Image
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        height, width = image.height, image.width

        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        # Process tracking tensor
        tracking_maps = tracking_tensor.float() # [T, C, H, W]
        tracking_maps = tracking_maps.to(device=self.device, dtype=dtype)
        tracking_first_frame = tracking_maps[0:1]  # Get first frame as [1, C, H, W]
        height, width = tracking_first_frame.shape[2], tracking_first_frame.shape[3]

        # 2. Set Scheduler.
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        pipe.to(self.device, dtype=dtype)
        # pipe.enable_sequential_cpu_offload()

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        pipe.transformer.gradient_checkpointing = False
        
        print("Encoding tracking maps")
        tracking_maps = tracking_maps.unsqueeze(0) # [B, T, C, H, W]
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
        tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        # 4. Generate the video frames based on the prompt.
        video_generate = pipe(
            prompt=prompt,
            negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            tracking_maps=tracking_maps,
            tracking_image=tracking_first_frame,
            height=height,
            width=width,
        ).frames[0]
        
        # 5. Export the generated frames to a video file. fps must be 8 for original video.
        output_path = output_path if output_path else f"result.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_to_video(video_generate, output_path, fps=fps)
        
    #========== camera parameters ==========#

    def _set_camera_motion(self, camera_motion):
        self.camera_motion = camera_motion

    def _get_intr(self, fov, H=480, W=720):
        fov_rad = math.radians(fov)
        focal_length = (W / 2) / math.tan(fov_rad / 2)

        cx = W / 2
        cy = H / 2

        intr = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        return intr

    def _apply_poses(self, pts, intr, poses):
        """
        Args:
            pts (torch.Tensor): pointclouds coordinates [T, N, 3]
            intr (torch.Tensor): camera intrinsics [T, 3, 3]
            poses (numpy.ndarray): camera poses [T, 4, 4]
        """
        poses = torch.from_numpy(poses).float().to(self.device)
        
        T, N, _ = pts.shape
        ones = torch.ones(T, N, 1, device=self.device, dtype=torch.float)
        pts_hom = torch.cat([pts[:, :, :2], ones], dim=-1)  # (T, N, 3)
        pts_cam = torch.bmm(pts_hom, torch.linalg.inv(intr).transpose(1, 2))  # (T, N, 3)
        pts_cam[:,:, :3] /= pts[:, :, 2:3]

        # to homogeneous
        pts_cam = torch.cat([pts_cam, ones], dim=-1)  # (T, N, 4)
        
        if poses.shape[0] == 1:
            poses = poses.repeat(T, 1, 1)
        elif poses.shape[0] != T:
            raise ValueError(f"Poses length ({poses.shape[0]}) must match sequence length ({T})")
        
        pts_world = torch.bmm(pts_cam, poses.transpose(1, 2))[:, :, :3]  # (T, N, 3)

        pts_proj = torch.bmm(pts_world, intr.transpose(1, 2))  # (T, N, 3)
        pts_proj[:, :, :2] /= pts_proj[:, :, 2:3]

        return pts_proj
    
    def apply_traj_on_tracking(self, pred_tracks, camera_motion=None, fov=55, frame_num=49):
        intr = self._get_intr(fov).unsqueeze(0).repeat(frame_num, 1, 1).to(self.device)
        tracking_pts = self._apply_poses(pred_tracks.squeeze(), intr, camera_motion).unsqueeze(0)
        return tracking_pts
    
    ##============= SpatialTracker =============##
    
    def generate_tracking_spatracker(self, video_tensor, density=70):
        """Generate tracking video
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            
        Returns:
            str: Path to tracking video
        """
        print("Loading tracking models...")
        # Load tracking model
        tracker = SpaTrackerPredictor(
            checkpoint=os.path.join(project_root, 'checkpoints/spaT_final.pth'),
            interp_shape=(384, 576),
            seq_length=12
        ).to(self.device)
        
        # Load depth model
        self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.depth_preprocessor.to(self.device)
        
        try:
            video = video_tensor.unsqueeze(0).to(self.device)
            
            video_depths = []
            for i in range(video_tensor.shape[0]):
                frame = (video_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                depth = self.depth_preprocessor(Image.fromarray(frame))[0]
                depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
                video_depths.append(depth_tensor)
            video_depth = torch.stack(video_depths, dim=0).to(self.device)
            # print("Video depth shape:", video_depth.shape)
            
            segm_mask = np.ones((480, 720), dtype=np.uint8)
            
            pred_tracks, pred_visibility, T_Firsts = tracker(
                video * 255, 
                video_depth=video_depth,
                grid_size=density,
                backward_tracking=False,
                depth_predictor=None,
                grid_query_frame=0,
                segm_mask=torch.from_numpy(segm_mask)[None, None].to(self.device),
                wind_length=12,
                progressive_tracking=False
            )

            return pred_tracks, pred_visibility, T_Firsts
            
        finally:
            # Clean up GPU memory
            del tracker, self.depth_preprocessor
            torch.cuda.empty_cache()

    def visualize_tracking_spatracker(self, video, pred_tracks, pred_visibility, T_Firsts, save_tracking=True):
        video = video.unsqueeze(0).to(self.device)
        vis = Visualizer(save_dir=self.output_dir, grayscale=False, fps=24, pad_value=0)
        msk_query = (T_Firsts == 0)
        pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
        pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
        
        tracking_video = vis.visualize(video=video, tracks=pred_tracks,
                        visibility=pred_visibility, save_video=False,
                        filename="temp")
        
        tracking_video = tracking_video.squeeze(0) # [T, C, H, W]
        wide_list = list(tracking_video.unbind(0))
        wide_list = [wide.permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        clip = ImageSequenceClip(wide_list, fps=self.fps)

        tracking_path = None
        if save_tracking:
            try:
                tracking_path = os.path.join(self.output_dir, "tracking_video.mp4")
                clip.write_videofile(tracking_path, codec="libx264", fps=self.fps, logger=None)
                print(f"Video saved to {tracking_path}")
            except Exception as e:
                print(f"Warning: Failed to save tracking video: {e}")
                tracking_path = None
        
        # Convert tracking_video back to tensor in range [0,1]
        tracking_frames = np.array(list(clip.iter_frames())) / 255.0
        tracking_video = torch.from_numpy(tracking_frames).permute(0, 3, 1, 2).float()
        
        return tracking_path, tracking_video
    
    ##============= MoGe =============##

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds
        
        Args:
            pixels (numpy.ndarray): Pixel coordinates of shape [N, 2]
            W (int): Image width
            H (int): Image height
            
        Returns:
            numpy.ndarray: Boolean mask of valid pixels
        """
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values
        
        Args:
            points (numpy.ndarray): Points array of shape [N, 2]
            depths (numpy.ndarray): Depth values of shape [N]
            
        Returns:
            tuple: (sorted_points, sorted_depths, sort_index)
        """
        # Combine points and depths into a single array for sorting
        combined = np.hstack((points, depths[:, None]))  # Nx3 (points + depth)
        # Sort by depth (last column) in descending order
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
        # Split back into points and depths
        sorted_points = sorted_combined[:, :-1]
        sorted_depths = sorted_combined[:, -1]
        return sorted_points, sorted_depths, sort_index

    def draw_rectangle(self, rgb, coord, side_length, color=(255, 0, 0)):
        """Draw a rectangle on the image
        
        Args:
            rgb (PIL.Image): Image to draw on
            coord (tuple): Center coordinates (x, y)
            side_length (int): Length of rectangle sides
            color (tuple): RGB color tuple
        """
        draw = ImageDraw.Draw(rgb)
        # Calculate the bounding box of the rectangle
        left_up_point = (coord[0] - side_length//2, coord[1] - side_length//2)  
        right_down_point = (coord[0] + side_length//2, coord[1] + side_length//2)
        color = tuple(list(color))

        draw.rectangle(
            [left_up_point, right_down_point],
            fill=tuple(color),
            outline=tuple(color),
        )
    
    def visualize_tracking_moge(self, points, mask, save_tracking=True):
        """Visualize tracking results from MoGe model
        
        Args:
            points (numpy.ndarray): Points array of shape [T, H, W, 3]
            mask (numpy.ndarray): Binary mask of shape [H, W]
            save_tracking (bool): Whether to save tracking video
            
        Returns:
            tuple: (tracking_path, tracking_video)
                - tracking_path (str): Path to saved tracking video, None if save_tracking is False
                - tracking_video (torch.Tensor): Tracking visualization tensor of shape [T, C, H, W] in range [0,1]
        """
        # Create color array
        T, H, W, _ = points.shape
        colors = np.zeros((H, W, 3), dtype=np.uint8)

        # Set R channel - based on x coordinates (smaller on the left)
        colors[:, :, 0] = np.tile(np.linspace(0, 255, W), (H, 1))

        # Set G channel - based on y coordinates (smaller on the top)
        colors[:, :, 1] = np.tile(np.linspace(0, 255, H), (W, 1)).T

        # Set B channel - based on depth
        z_values = points[0, :, :, 2]  # get z values
        inv_z = 1 / z_values  # calculate 1/z
        # Calculate 2% and 98% percentiles
        p2 = np.percentile(inv_z, 2)
        p98 = np.percentile(inv_z, 98)
        # Normalize to [0,1] range
        normalized_z = np.clip((inv_z - p2) / (p98 - p2), 0, 1)
        colors[:, :, 2] = (normalized_z * 255).astype(np.uint8)
        colors = colors.astype(np.uint8)

        # colors = colors[mask]
        # points = points * mask[None, :, :, None]
        
        points = points.reshape(T, -1, 3)
        colors = colors.reshape(-1, 3)
        
        # Initialize list to store frames
        frames = []
        
        for i, pts_i in enumerate(tqdm(points)):
            pixels, depths = pts_i[..., :2], pts_i[..., 2]
            pixels[..., 0] = pixels[..., 0] * W
            pixels[..., 1] = pixels[..., 1] * H
            pixels = pixels.astype(int)
            
            valid = self.valid_mask(pixels, W, H)
            frame_rgb = colors[valid]
            pixels = pixels[valid]
            depths = depths[valid]
            
            img = Image.fromarray(np.uint8(np.zeros([H, W, 3])), mode="RGB")
            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            step = 1
            sorted_pixels = sorted_pixels[::step]
            sorted_rgb = frame_rgb[sort_index][::step]
            
            for j in range(sorted_pixels.shape[0]):
                self.draw_rectangle(
                    img,
                    coord=(sorted_pixels[j, 0], sorted_pixels[j, 1]),
                    side_length=2,
                    color=sorted_rgb[j],
                )
            frames.append(np.array(img))

        # Convert frames to video tensor in range [0,1]
        tracking_video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

        tracking_path = None
        if save_tracking:
            try:
                tracking_path = os.path.join(self.output_dir, "tracking_video_moge.mp4")
                # Convert back to uint8 for saving
                uint8_frames = [frame.astype(np.uint8) for frame in frames]
                clip = ImageSequenceClip(uint8_frames, fps=self.fps)
                clip.write_videofile(tracking_path, codec="libx264", fps=self.fps, logger=None)
                print(f"Video saved to {tracking_path}")
            except Exception as e:
                print(f"Warning: Failed to save tracking video: {e}")
                tracking_path = None

        return tracking_path, tracking_video
    
    def apply_tracking(self, video_tensor, fps=8, tracking_tensor=None, img_cond_tensor=None, prompt=None):
        """Generate final video with motion transfer
        
        Args:
            video_tensor (torch.Tensor): Input video tensor [T,C,H,W]
            fps (float): Input video FPS
            tracking_tensor (torch.Tensor): Tracking video tensor [T,C,H,W]
            image_tensor (torch.Tensor): First frame tensor [C,H,W] to use for generation
            prompt (str): Generation prompt
            checkpoint_path (str): Path to model checkpoint
        """
        self.fps = fps

        # Use first frame if no image provided
        if img_cond_tensor is None:
            img_cond_tensor = video_tensor[0]
        
        # Generate final video

        final_output = os.path.join(os.path.abspath(self.output_dir), "result.mp4")
        self._infer(
            prompt=prompt,
            tracking_tensor=tracking_tensor,
            image_tensor=img_cond_tensor,
            output_path=final_output,
            num_inference_steps=50,
            guidance_scale=6.0,
            dtype=torch.bfloat16,
            fps=self.fps
        )
        print(f"Final video generated successfully at: {final_output}")

    def _set_object_motion(self, motion_type):
        """Set object motion type
        
        Args:
            motion_type (str): Motion direction ('up', 'down', 'left', 'right')
        """
        self.object_motion = motion_type

class FirstFrameRepainter:
    def __init__(self, gpu_id=0, output_dir='outputs'):
        """Initialize FirstFrameRepainter
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        self.device = f"cuda:{gpu_id}"
        self.output_dir = output_dir
        self.max_depth = 65.0
        os.makedirs(output_dir, exist_ok=True)
        
    def repaint(self, image_tensor, prompt, depth_path=None, method="dav"):
        """Repaint first frame using Flux
        
        Args:
            image_tensor (torch.Tensor): Input image tensor [C,H,W]
            prompt (str): Repaint prompt
            depth_path (str): Path to depth image
            method (str): depth estimator, "moge" or "dav" or "zoedepth"
            
        Returns:
            torch.Tensor: Repainted image tensor [C,H,W]
        """
        print("Loading Flux model...")
        # Load Flux model
        flux_pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev", 
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Get depth map
        if depth_path is None:
            if method == "moge":
                self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
                depth_map = self.moge_model.infer(image_tensor.to(self.device))["depth"]
                depth_map = torch.clamp(depth_map, max=self.max_depth)
                depth_normalized = 1.0 - (depth_map / self.max_depth)
                depth_rgb = (depth_normalized * 255).cpu().numpy().astype(np.uint8)
                control_image = Image.fromarray(depth_rgb).convert("RGB")
            elif method == "zoedepth":
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
                self.depth_preprocessor.to(self.device)
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(image_np))[0].convert("RGB")
                control_image = control_image.point(lambda x: 255 - x) # the zoedepth depth is inverted
            else:
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                self.depth_preprocessor.to(self.device)
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(image_np))[0].convert("RGB")
        else:
            control_image = Image.open(depth_path).convert("RGB")

        try:
            repainted_image = flux_pipe(
                prompt=prompt,
                control_image=control_image,
                height=480,
                width=720,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # Save repainted image
            repainted_image.save(os.path.join(self.output_dir, "temp_repainted.png"))
            
            # Convert PIL Image to tensor
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            repainted_tensor = transform(repainted_image)
            
            return repainted_tensor
            
        finally:
            # Clean up GPU memory
            del flux_pipe
            if method == "moge":
                del self.moge_model
            else:
                del self.depth_preprocessor
            torch.cuda.empty_cache()

class CameraMotionGenerator:
    def __init__(self, motion_type, frame_num=49, H=480, W=720, fx=None, fy=None, fov=55, device='cuda'):
        self.motion_type = motion_type
        self.frame_num = frame_num
        self.fov = fov
        self.device = device
        self.W = W
        self.H = H
        self.intr = torch.tensor([
            [0, 0, W / 2],
            [0, 0, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        # if fx, fy not provided
        if not fx or not fy:
            fov_rad = math.radians(fov)
            fx = fy = (W / 2) / math.tan(fov_rad / 2)
 
        self.intr[0, 0] = fx
        self.intr[1, 1] = fy        

    def _apply_poses(self, pts, poses):
        """
        Args:
            pts (torch.Tensor): pointclouds coordinates [T, N, 3]
            intr (torch.Tensor): camera intrinsics [T, 3, 3]
            poses (numpy.ndarray): camera poses [T, 4, 4]
        """
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        intr = self.intr.unsqueeze(0).repeat(self.frame_num, 1, 1).to(torch.float)
        T, N, _ = pts.shape
        ones = torch.ones(T, N, 1, device=self.device, dtype=torch.float)
        pts_hom = torch.cat([pts[:, :, :2], ones], dim=-1)  # (T, N, 3)
        pts_cam = torch.bmm(pts_hom, torch.linalg.inv(intr).transpose(1, 2))  # (T, N, 3)
        pts_cam[:,:, :3] *= pts[:, :, 2:3]

        # to homogeneous
        pts_cam = torch.cat([pts_cam, ones], dim=-1)  # (T, N, 4)
        
        if poses.shape[0] == 1:
            poses = poses.repeat(T, 1, 1)
        elif poses.shape[0] != T:
            raise ValueError(f"Poses length ({poses.shape[0]}) must match sequence length ({T})")
        
        poses = poses.to(torch.float).to(self.device)
        pts_world = torch.bmm(pts_cam, poses.transpose(1, 2))[:, :, :3]  # (T, N, 3)
        pts_proj = torch.bmm(pts_world, intr.transpose(1, 2))  # (T, N, 3)
        pts_proj[:, :, :2] /= pts_proj[:, :, 2:3]

        return pts_proj
    
    def w2s(self, pts, poses):
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        assert poses.shape[0] == self.frame_num
        poses = poses.to(torch.float32).to(self.device)
        T, N, _ = pts.shape  # (T, N, 3)
        intr = self.intr.unsqueeze(0).repeat(self.frame_num, 1, 1)
        # Step 1: 扩展点的维度，使其变成 (T, N, 4)，最后一维填充1 (齐次坐标)
        ones = torch.ones((T, N, 1), device=self.device, dtype=pts.dtype)
        points_world_h = torch.cat([pts, ones], dim=-1)
        points_camera_h = torch.bmm(poses, points_world_h.permute(0, 2, 1))
        points_camera = points_camera_h[:, :3, :].permute(0, 2, 1)

        points_image_h = torch.bmm(points_camera, intr.permute(0, 2, 1))

        uv = points_image_h[:, :, :2] / points_image_h[:, :, 2:3]

        # Step 5: 提取深度 (Z) 并拼接
        depth = points_camera[:, :, 2:3]  # (T, N, 1)
        uvd = torch.cat([uv, depth], dim=-1)  # (T, N, 3)

        return uvd  # 屏幕坐标 + 深度 (T, N, 3)

    def apply_motion_on_pts(self, pts, camera_motion):
        tracking_pts = self._apply_poses(pts.squeeze(), camera_motion).unsqueeze(0)
        return tracking_pts
    
    def set_intr(self, K):
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.intr = K.to(self.device)

    def rot_poses(self, angle, axis='y'):
        """
        pts (torch.Tensor): [T, N, 3]
        angle (int): angle of rotation (degree)
        """
        angle_rad = math.radians(angle)
        angles = torch.linspace(0, angle_rad, self.frame_num)
        rot_mats = torch.zeros(self.frame_num, 4, 4)
    
        for i, theta in enumerate(angles):
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            if axis == 'x':
                rot_mats[i] = torch.tensor([
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
            elif axis == 'y':
                rot_mats[i] = torch.tensor([
                    [cos_theta, 0, sin_theta, 0],
                    [0, 1, 0, 0],
                    [-sin_theta, 0, cos_theta, 0],
                    [0, 0, 0, 1]
                ], dtype=torch.float32)
            
            elif axis == 'z':
                rot_mats[i] = torch.tensor([
                    [cos_theta, -sin_theta, 0, 0],
                    [sin_theta, cos_theta, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=torch.float32)
            else:
                raise ValueError("Invalid axis value. Choose 'x', 'y', or 'z'.")
            
        return rot_mats.to(self.device)

    def trans_poses(self, dx, dy, dz):
        """
        params:
        - dx: float, displacement along x axis。
        - dy: float, displacement along y axis。
        - dz: float, displacement along z axis。

        ret:
        - matrices: torch.Tensor
        """
        trans_mats = torch.eye(4).unsqueeze(0).repeat(self.frame_num, 1, 1)  # (n, 4, 4)

        delta_x = dx / (self.frame_num - 1)
        delta_y = dy / (self.frame_num - 1)
        delta_z = dz / (self.frame_num - 1)

        for i in range(self.frame_num):
            trans_mats[i, 0, 3] = i * delta_x
            trans_mats[i, 1, 3] = i * delta_y
            trans_mats[i, 2, 3] = i * delta_z

        return trans_mats.to(self.device)
    

    def _look_at(self, camera_position, target_position):
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        # calculate rotation matrix
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction])
        rotation_matrix = np.linalg.inv(rotation_matrix)
        return rotation_matrix

    def spiral_poses(self, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
        """Generate spiral camera poses
        
        Args:
            radius (float): Base radius of the spiral
            forward_ratio (float): Scale factor for forward motion
            backward_ratio (float): Scale factor for backward motion
            rotation_times (float): Number of rotations to complete
            look_at_times (float): Scale factor for look-at point distance
            
        Returns:
            torch.Tensor: Camera poses of shape [num_frames, 4, 4]
        """
        # Generate spiral trajectory
        t = np.linspace(0, 1, self.frame_num)
        r = np.sin(np.pi * t) * radius * rotation_times
        theta = 2 * np.pi * t
        
        # Calculate camera positions
        # Limit y motion for better floor/sky view
        y = r * np.cos(theta) * 0.3  
        x = r * np.sin(theta)
        z = -r
        z[z < 0] *= forward_ratio
        z[z > 0] *= backward_ratio
        
        # Set look-at target
        target_pos = np.array([0, 0, radius * look_at_times])
        cam_pos = np.vstack([x, y, z]).T
        cam_poses = []
        
        for pos in cam_pos:
            rot_mat = self._look_at(pos, target_pos)
            trans_mat = np.eye(4)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = pos
            cam_poses.append(trans_mat[None])
            
        camera_poses = np.concatenate(cam_poses, axis=0)
        return torch.from_numpy(camera_poses).to(self.device)

    def rot(self, pts, angle, axis):
        """
        pts: torch.Tensor, (T, N, 2)
        """
        rot_mats = self.rot_poses(angle, axis)
        pts = self.apply_motion_on_pts(pts, rot_mats)
        return pts
    
    def trans(self, pts, dx, dy, dz):
        if pts.shape[-1] != 3:
            raise ValueError("points should be in the 3d coordinate.")
        trans_mats = self.trans_poses(dx, dy, dz)
        pts = self.apply_motion_on_pts(pts, trans_mats)
        return pts

    def spiral(self, pts, radius):
        spiral_poses = self.spiral_poses(radius)
        pts = self.apply_motion_on_pts(pts, spiral_poses)
        return pts

    def get_default_motion(self):
        if self.motion_type == 'trans':
            motion = self.trans_poses(0.1, 0, 0)
        elif self.motion_type == 'spiral':
            motion = self.spiral_poses(1)
        elif self.motion_type == 'rot':
            motion = self.rot_poses(-25, 'y')
        else:
            raise ValueError(f'camera_motion must be in [trans, spiral, rot], but get {self.motion_type}.')
    
        return motion

class ObjectMotionGenerator:
    def __init__(self, device="cuda:0"):
        """Initialize ObjectMotionGenerator
        
        Args:
            device (str): Device to run on
        """
        self.device = device
        self.num_frames = 49
        
    def _get_points_in_mask(self, pred_tracks, mask):
        """Get points that fall within the mask in first frame
        
        Args:
            pred_tracks (torch.Tensor): [num_frames, num_points, 3]
            mask (torch.Tensor): [H, W] binary mask
            
        Returns:
            torch.Tensor: Boolean mask of selected points [num_points]
        """
        first_frame_points = pred_tracks[0]  # [num_points, 3]
        xy_points = first_frame_points[:, :2]  # [num_points, 2]
        
        # Convert xy coordinates to pixel indices
        xy_pixels = xy_points.round().long()  # Convert to integer pixel coordinates
        
        # Clamp coordinates to valid range
        xy_pixels[:, 0].clamp_(0, mask.shape[1] - 1)  # x coordinates
        xy_pixels[:, 1].clamp_(0, mask.shape[0] - 1)  # y coordinates
        
        # Get mask values at point locations
        points_in_mask = mask[xy_pixels[:, 1], xy_pixels[:, 0]]  # Index using y, x order
        
        return points_in_mask
        
    def generate_motion(self, mask, motion_type, distance, num_frames=49):
        """Generate motion dictionary for the given parameters
        
        Args:
            mask (torch.Tensor): [H, W] binary mask
            motion_type (str): Motion direction ('up', 'down', 'left', 'right')
            distance (float): Total distance to move
            num_frames (int): Number of frames
            
        Returns:
            dict: Motion dictionary containing:
                - mask (torch.Tensor): Binary mask
                - motions (torch.Tensor): Per-frame motion vectors [num_frames, 4, 4]
        """

        self.num_frames = num_frames
        # Define motion template vectors
        template = {
            'up': torch.tensor([0, -1, 0]),
            'down': torch.tensor([0, 1, 0]),
            'left': torch.tensor([-1, 0, 0]),
            'right': torch.tensor([1, 0, 0]),
            'front': torch.tensor([0, 0, 1]),
            'back': torch.tensor([0, 0, -1])
        }
        
        if motion_type not in template:
            raise ValueError(f"Unknown motion type: {motion_type}")
            
        # Move mask to device
        mask = mask.to(self.device)
        
        # Generate per-frame motion matrices
        motions = []
        base_vec = template[motion_type].to(self.device) * distance
        
        for frame_idx in range(num_frames):
            # Calculate interpolation factor (0 to 1)
            t = frame_idx / (num_frames - 1)
            
            # Create motion matrix for current frame
            current_motion = torch.eye(4, device=self.device)
            current_motion[:3, 3] = base_vec * t
            motions.append(current_motion)
            
        motions = torch.stack(motions)  # [num_frames, 4, 4]
        
        return {
            'mask': mask,
            'motions': motions
        }
        
    def apply_motion(self, pred_tracks, motion_dict, tracking_method="spatracker"):
        """Apply motion to selected points
        
        Args:
            pred_tracks (torch.Tensor): [num_frames, num_points, 3] for spatracker
                                      or [T, H, W, 3] for moge
            motion_dict (dict): Motion dictionary containing mask and motions
            tracking_method (str): "spatracker" or "moge"
            
        Returns:
            torch.Tensor: Modified pred_tracks with same shape as input
        """
        pred_tracks = pred_tracks.to(self.device).float()
        
        if tracking_method == "moge":
            T, H, W, _ = pred_tracks.shape
            
            selected_mask = motion_dict['mask']
            valid_selected = ~torch.any(torch.isnan(pred_tracks[0]), dim=2) & selected_mask
            valid_selected = valid_selected.reshape([-1])
            modified_tracks = pred_tracks.clone().reshape(T, -1, 3)

            for frame_idx in range(self.num_frames):
                motion_mat = motion_dict['motions'][frame_idx]
                motion_mat[0, 3] /= W
                motion_mat[1, 3] /= H
                points = modified_tracks[frame_idx, valid_selected]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)

                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, valid_selected] = transformed_points[:, :3]
            return modified_tracks
            
        else:
            points_in_mask = self._get_points_in_mask(pred_tracks, motion_dict['mask'])
            modified_tracks = pred_tracks.clone()
            
            for frame_idx in range(pred_tracks.shape[0]):
                motion_mat = motion_dict['motions'][frame_idx]
                points = modified_tracks[frame_idx, points_in_mask]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, points_in_mask] = transformed_points[:, :3]
            
            return modified_tracks
