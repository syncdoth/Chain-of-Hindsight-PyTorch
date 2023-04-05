ids=$1
N_GPU=$2

WORLD_SIZE=$N_GPU CUDA_VISIBLE_DEVICES=$ids torchrun --nproc_per_node $N_GPU --master_port 9808 \
    -m coh.coh_train \
    --model_name 'google/flan-ul2' \
    --wandb_project_name CoH \
    --wandb_run_name 'flan-ul2-seq512-bs128' \
    --hf_weights "" \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --eval_steps 1000 \
    --max_steps 100000 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps 1000 \
    --save_total_limit 3 \
    --load_best_model_at_end False \
    --pt_loss_weight 0.75 \
    --seq_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --use_lora --fp16 --train_8bit
