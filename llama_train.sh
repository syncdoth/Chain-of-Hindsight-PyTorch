ids=$1
LLAMA_PATH=$2

CUDA_VISIBLE_DEVICES=$ids torchrun --nproc_per_node=4 --master_port=9808 -m coh.coh_train \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
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
    --logging_steps 100 \
    --save_strategy 'steps' \
    --save_steps 10000 \
    --seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --pt_loss_weight 1.0 \
    --bf16 True --tf32 True
