import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path


def images_to_video(image_folder, output_video_path, fps=30, image_pattern='color_{:06d}.jpg'):
    """
    将图像序列转换为视频
    
    Args:
        image_folder: 图像文件夹路径
        output_video_path: 输出视频路径
        fps: 视频帧率
        image_pattern: 图像文件名模式
    """
    # 获取所有图像文件
    images = sorted([img for img in os.listdir(image_folder) if img.startswith('color_') and img.endswith('.jpg')])
    
    if len(images) == 0:
        print(f"警告: {image_folder} 中没有找到图像文件")
        return False
    
    # 读取第一张图像获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    
    if frame is None:
        print(f"错误: 无法读取图像 {first_image_path}")
        return False
    
    height, width, layers = frame.shape
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 写入所有帧
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"警告: 无法读取图像 {image_path}")
    
    video_writer.release()
    return True


def process_dexycb_dataset(input_root, output_root, fps=30):
    """
    处理DexYCB数据集，将所有图像序列转换为视频
    
    Args:
        input_root: DexYCB数据集根目录
        output_root: 输出视频根目录
        fps: 视频帧率
    """
    
    # 获取所有subject文件夹
    subjects = sorted([d for d in os.listdir(input_root) 
                      if os.path.isdir(os.path.join(input_root, d)) and 'subject' in d])
    
    print(f"找到 {len(subjects)} 个subjects")
    
    total_videos = 0
    success_videos = 0
    
    # 遍历所有subjects
    for subject in tqdm(subjects, desc="处理subjects"):
        subject_path = os.path.join(input_root, subject)
        
        # 获取所有序列文件夹
        sequences = sorted([d for d in os.listdir(subject_path) 
                          if os.path.isdir(os.path.join(subject_path, d))])
        
        for sequence in sequences:
            sequence_path = os.path.join(subject_path, sequence)
            
            # 获取所有相机序列文件夹
            camera_serials = sorted([d for d in os.listdir(sequence_path) 
                                   if os.path.isdir(os.path.join(sequence_path, d))])
            
            for camera_serial in camera_serials:
                camera_path = os.path.join(sequence_path, camera_serial)
                
                # 构建输出视频路径
                relative_path = os.path.relpath(camera_path, input_root)
                output_video_path = os.path.join(output_root, relative_path, 'video.mp4')
                
                # 如果视频已存在，跳过
                if os.path.exists(output_video_path):
                    # print(f"视频已存在，跳过: {output_video_path}")
                    total_videos += 1
                    success_videos += 1
                    continue
                
                # 转换图像为视频
                total_videos += 1
                if images_to_video(camera_path, output_video_path, fps):
                    success_videos += 1
    
    print(f"\n处理完成!")
    print(f"总共处理: {total_videos} 个视频")
    print(f"成功生成: {success_videos} 个视频")
    print(f"失败: {total_videos - success_videos} 个视频")


def main():
    parser = argparse.ArgumentParser(description='将DexYCB数据集的图像序列转换为视频')
    parser.add_argument('--input_root', type=str, 
                       default='/path/to/dexycb/',
                       help='DexYCB数据集根目录')
    parser.add_argument('--output_root', type=str,
                       default='data/dexycb_videos',
                       help='输出视频根目录')
    parser.add_argument('--fps', type=int, default=30,
                       help='视频帧率')
    args = parser.parse_args()
    
    print(f"输入目录: {args.input_root}")
    print(f"输出目录: {args.output_root}")
    print(f"视频帧率: {args.fps}")
    
    if not os.path.exists(args.input_root):
        print(f"错误: 输入目录不存在: {args.input_root}")
        return
    
    process_dexycb_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        fps=args.fps
    )


if __name__ == '__main__':
    main()
