accelerator launch  --main_process_port 0 \
                    --config ./config.yaml \
                    --data_root /path/to/data/root \
                    --train_caption_column /path/to/train/caption.txt \
                    --train_video_column /path/to/train/video.txt \
                    --train_depth_column /path/to/train/depth.txt \
                    --train_label_column /path/to/train/label.txt \
                    --valid_caption_column /path/to/valid/caption.txt \
                    --valid_video_column /path/to/valid/video.txt \
                    --valid_depth_column /path/to/valid/depth.txt \
                    --valid_label_column /path/to/valid/label.txt \
                    double_controlnet.py
                    