export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0,1,2,3"
PORT=29506

LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("100000")

# training dataset parameters

DATA_ROOT="."
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

VIDEO_COLUMN="/path/to/video/latents/filelist"
CAPTION_COLUMN="/path/to/caption/latents/filelist"
TRACKING_COLUMN="/path/to/tracking/latents/filelist"
NORMAL_COLUMN="/path/to/normal/latents/filelist"
DEPTH_COLUMN="/path/to/depth/latents/filelist"
SEG_MASK_COLUMN="/path/to/seg_mask/latents/filelist"
HAND_KEYPOINTS_COLUMN="/path/to/hand_keypoints/latents/filelist"

INITIAL_FRAMES_NUM=1

# validation parameters
TRACKING_MAP_PATH="./val_resources/comb_val/val_tracking.mp4"
DEPTH_MAP_PATH="./val_resources/comb_val/val_depth.mp4"
NORMAL_MAP_PATH="./val_resources/comb_val/val_normal.mp4"
SEG_MASK_PATH="./val_resources/comb_val/val_seg_mask.mp4"
HAND_KEYPOINTS_PATH="./val_resources/comb_val/val_hand_keypoints.mp4"
VALIDATION_PROMPT="Initially, a man in a dark t-shirt and blue jeans is seen in an industrial setting, holding a red object with a focused expression, surrounded by a black table with a white box and a red mug, and a robotic arm with a camera. The scene shifts to the man in a black t-shirt and blue jeans, holding a red object with a focused expression, with a robotic arm and a camera system nearby, suggesting a demonstration or testing scenario. Finally, the man, now in a black t-shirt and blue shorts, holds a red cube with a focused expression, in an environment equipped with a robotic arm and a camera system, indicating a technological demonstration or testing scenario."
VALIDATION_IMAGES="./val_resources/comb_val/val_image.jpg"

ACCELERATE_CONFIG_FILE="./accelerate_configs/deepspeed.yaml"

TRAIN_BATCH_SIZE=4
CHECKPOINT_STEPS=100
WARMUP_STEPS=100

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./exps/main_all_cond_HOI_s0_split_${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS --main_process_port $PORT training/cogvideox_image_to_video_comb_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --used_conditions hand_keypoints,depth,seg_mask \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --image_column $IMAGE_COLUMN \
          --tracking_column $TRACKING_COLUMN \
          --tracking_image_column $TRACKING_IMAGE_COLUMN \
          --tracking_map_path $TRACKING_MAP_PATH \
          --depth_column $DEPTH_COLUMN \
          --depth_image_column $DEPTH_IMAGE_COLUMN \
          --depth_map_path $DEPTH_MAP_PATH \
          --normal_column $NORMAL_COLUMN \
          --normal_image_column $NORMAL_IMAGE_COLUMN \
          --normal_map_path $NORMAL_MAP_PATH \
          --seg_mask_column $SEG_MASK_COLUMN \
          --seg_mask_image_column $SEG_MASK_IMAGE_COLUMN \
          --seg_mask_path $SEG_MASK_PATH \
          --hand_keypoints_column $HAND_KEYPOINTS_COLUMN \
          --hand_keypoints_image_column $HAND_KEYPOINTS_IMAGE_COLUMN \
          --hand_keypoints_path $HAND_KEYPOINTS_PATH \
          --initial_frames_num $INITIAL_FRAMES_NUM \
          --random_mask --load_tensors \
          --num_tracking_blocks 12 \
          --height_buckets 480 \
          --width_buckets 720 \
          --height 480 \
          --width 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 4 \
          --pin_memory \
          --validation_prompt \"$VALIDATION_PROMPT\" \
          --validation_images $VALIDATION_IMAGES \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps 4 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --resume_from_checkpoint \"latest\" \
          --nccl_timeout 1800 --initial_frames_num 1
          "
    
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
