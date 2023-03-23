ids=$1
LLAMA_PATH=$2

CUDA_VISIBLE_DEVICES=$ids deepspeed --module coh.coh_train \
    --deepspeed ds_config/auto_zero3.json \
    --model_name $LLAMA_PATH/llama-7b \
    --tokenizer_name $LLAMA_PATH/tokenizer \
    --cache_dir $XDG_CACHE_HOME \
    --wandb_project_name CoH \
    --wandb_run_name 'CoH-LLaMA-7B' \
    --hf_weights "" \
    --learning_rate 2e-5 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --eval_steps 10000 \
    --max_steps 1000000 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps 10000 \
    --pt_loss_weight 1.0 \
    --seq_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --use_lora \
    --bf16 True --tf32 True
