torchrun --nproc_per_node=4 training/prepare_dataset_dexycb.py \
    --data_root /path/to/dexycb/data/root \
    --caption_column "/path/to/caption/filelist" \
    --video_column "/path/to/video/filelist" \
    --depth_column "/path/to/depth/filelist" \
    --label_column "/path/to/label/filelist" \
    --height_buckets 480 --width_buckets 720 \
    --save_image_latents --output_dir "/path/to/output/dir" \
    --target_fps 15 --save_latents_and_embeddings

