ids=$1

CUDA_VISIBLE_DEVICES=$ids python -m coh.coh_train \
    --model_name 'EleutherAI/gpt-j-6B' \
    --cache_dir cache \
    --project_name CoH \
    --run_name 'CoH-GPT-J-6B' \
    --seq_length 512 \
    --batch_size 512 \
    --hf_weights "" \
    --learning_rate 5e-4 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --eval_steps 10000 \
    --max_steps 1000000 \
    --report_to 'wandb' \
    --logging_steps 100 \
    --save_strategy 'no' \
    --gradient_accumulation_steps 1 \
    --pt_loss_weight 1.0

    # --fp16 --bf16  # mixed precisions; use them as appropriate