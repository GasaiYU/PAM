import os
from tqdm import tqdm
import argparse

def get_split(setup='s0', split='train'):
    if setup == 's0':
        if split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 != 4]
        elif split == 'val':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return subject_ind, serial_ind, sequence_ind


def gen_video_filelist(root_dir='data/dexycb', setup='s0', split='train', suffix='video.mp4'):
    subject_ind, serial_ind, sequence_ind = get_split(setup, split)
    
    # Get subjects ID
    subjects = [subject for subject in os.listdir(root_dir) if 'subject' in subject]
    subjects.sort(key=lambda x: int(x.split('-')[-1]))
    subjects = [subjects[i] for i in subject_ind]

    files = []
    for subject in subjects:
        subject_path = os.path.join(root_dir, subject)
        
        sequences = [seq for seq in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, seq))]
        sequences.sort()
        sequences = [sequences[i] for i in sequence_ind]
        
        for seq in sequences:
            seq_path = os.path.join(subject_path, seq)

            # Get Serials ID
            serials = [serial for serial in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, serial))]
            serials.sort()
            serials = [serials[i] for i in serial_ind]
            
            if suffix != '':
                files.extend([os.path.join(seq_path, serial, suffix) for serial in serials])
            else:
                files.extend([os.path.join(seq_path, serial) for serial in serials])

    return files

def gen_condition_filelist_from_video_filelist(video_files, condition_prefix, condition_suffix):
    condition_paths = []
    
    for video_file in tqdm(video_files):
        condition_path = os.path.dirname(video_file).replace('dexycb_videos', condition_prefix)
        if condition_suffix != '':
            condition_path = os.path.join(condition_path, condition_suffix)
        if not os.path.exists(condition_path):
            print(f'Condition path {condition_path} does not exist')
        condition_paths.append(condition_path)
            
    return condition_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/dexycb_videos')
    parser.add_argument('--setup', type=str, default='s0')
    parser.add_argument('--suffix', type=str, default='video.mp4')
    parser.add_argument('--save_path_root', type=str, default='data/dexycb_filelist/s0_split')
    parser.add_argument('--save_dict', type=dict, default={'train': 'training', 'val': 'val'})
    args = parser.parse_args()
    
    for split in ['train', 'val']:
        files = gen_video_filelist(args.root, args.setup, split, args.suffix)
        
        os.makedirs(os.path.join(args.save_path_root, split), exist_ok=True)
        with open(os.path.join(args.save_path_root, split, f'{args.save_dict[split]}_videos.txt'), 'w') as f:
            for file in files:
                f.write(file + '\n')
    pass
    
    conditions_prefix_dict = {
        "depth": "dexycb_depth",
        "label": "dexycb"
    }
    
    conditions_suffix_dict = {
        "depth": "masked_color.mp4",
        "label": ""
    }
    
    conditions_save_dict = {
        "depth": "depths",
        "label": "labels"
    }
    
    conditons = ["depth", "label"]
    
    for split in ['train', 'val']:
        files = gen_video_filelist(args.root, args.setup, split, args.suffix)
        for condition in conditons:
            condition_files = gen_condition_filelist_from_video_filelist(files, conditions_prefix_dict[condition], conditions_suffix_dict[condition])
            os.makedirs(os.path.join(args.save_path_root, split), exist_ok=True)
            with open(os.path.join(args.save_path_root, split, f'{args.save_dict[split]}_{conditions_save_dict[condition]}_videos.txt'), 'w') as f:
                for file in condition_files:
                    f.write(file + '\n')
if __name__ == '__main__':
    main() 