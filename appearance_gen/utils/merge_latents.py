import os
import shutil

from tqdm import tqdm

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="Source directory of the latents")
    parser.add_argument("--dst_dir", type=str, help="Destination directory of the merged latents")
    args = parser.parse_args()
    return args 

def split_path_func(path, split_list=["0", "1", "2", "3"]):
    split_path = path.split("/")
    for i, split in enumerate(split_path):
        if split in split_list:
            split_path[i] = ""
    
    # Remove empty strings
    split_path = list(filter(lambda x: x != "", split_path))
    return "/".join(split_path)
    

def merge_latent_files(
    src_dir: str,
    dst_dir: str,
    ext_lits: list = ["png", "mp4", "pt"],
):
    for root, dirs, files in tqdm(os.walk(src_dir)):
        for file in files:
            if file.split(".")[-1] in ext_lits:
                file_path = os.path.join(root, file)
                new_path = file_path.replace(src_dir, dst_dir)
                new_path = split_path_func(new_path)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copyfile(file_path, new_path)
                
if __name__ == "__main__":
    args = get_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    merge_latent_files(src_dir, dst_dir)
    