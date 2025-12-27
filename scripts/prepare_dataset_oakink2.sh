torchrun --nproc_per_node=1 --master_port 26226 training/prepare_dataset_oakink2.py \
    --data_root /path/to/oakink2/data/root \
    --caption_column "/path/to/caption/filelist" \
    --video_column "/path/to/video/filelist" \
    --depth_column "/path/to/depth/filelist" \
    --seg_column "/path/to/seg/filelist" \
    --hand_column "/path/to/hand/filelist" \
    --height_buckets 480 --width_buckets 720 \
    --save_image_latents --output_dir "/path/to/output/dir" \
    --target_fps 15 --save_latents_and_embeddings

