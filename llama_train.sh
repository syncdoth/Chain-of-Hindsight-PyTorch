ids=$1
LLAMA_PATH=$2

TORCH_DISTRIBUTED_DEBUG=DETAIL WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=$ids torchrun --nproc_per_node 2 --master_port 9808 \
    -m coh.coh_train \
    --model_name $LLAMA_PATH/llama-7b \
    --tokenizer_name $LLAMA_PATH/tokenizer \
    --wandb_project_name CoH \
    --wandb_run_name 'CoH-LLaMA-7B' \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --use_lora --fp16 --train_8bit
