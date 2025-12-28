import os
import glob

from argparse import ArgumentParser

def gen_oakink2_filelist(data_root, save_dir):
    video_filelist = glob.glob(os.path.join(data_root, "videos", "**", "*.mp4"), recursive=True)
    
    condition_dir_names = ["processed_hand_keypoints", "processed_seg_mask", "processed_depth"]
    condition_suffix_names = ["output_hand.mp4", "output_seg.mp4", "color.mp4"]
    save_names = ["hand_keypoints.txt", "seg_mask.txt", "depth.txt"]

    condition_filelist = [[], [], []]
    for video_path in video_filelist:
        for i in range(3):
            condition_file_name = video_path.replace("videos", condition_dir_names[i]).replace("video.mp4", condition_suffix_names[i])
            if os.path.exists(condition_file_name):
                condition_filelist[i].append(condition_file_name)
            else:
                print(f"Condition file {condition_file_name} does not exist")
    
    with open(os.path.join(save_dir, "videos.txt"), "w") as f:
        for video_path in video_filelist:
            f.write(video_path + "\n")
    
    for i in range(3):
        with open(os.path.join(save_dir, save_names[i]), "w") as f:
            for file in condition_filelist[i]:
                f.write(file + "\n")


def split_oakink2_filelist(filelist_path, save_path, training_samples=3000):
    with open(filelist_path, "r") as f:
        filelist = [line.strip() for line in f.readlines()]
    
    training_filelist = filelist[:training_samples]
    val_filelist = filelist[training_samples:]
    
    with open(save_path, "w") as f:
        for file in training_filelist:
            f.write(file + "\n")
    
    with open(save_path.replace("training", "val"), "w") as f:
        for file in val_filelist:
            f.write(file + "\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    
    gen_oakink2_filelist(args.data_root, args.save_dir)
    split_oakink2_filelist(os.path.join(args.save_dir, "videos.txt"), os.path.join(args.save_dir, "training_videos.txt"), 3000)
    split_oakink2_filelist(os.path.join(args.save_dir, "depths.txt"), os.path.join(args.save_dir, "training_depths.txt"), 3000)
    split_oakink2_filelist(os.path.join(args.save_dir, "seg_masks.txt"), os.path.join(args.save_dir, "training_seg_masks.txt"), 3000)
    split_oakink2_filelist(os.path.join(args.save_dir, "hand_keypoints.txt"), os.path.join(args.save_dir, "training_hand_keypoints.txt"), 3000)
    split_oakink2_filelist(os.path.join(args.save_dir, "prompts.txt"), os.path.join(args.save_dir, "training_prompts.txt"), 3000)