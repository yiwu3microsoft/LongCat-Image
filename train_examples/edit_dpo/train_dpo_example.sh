export TOKENIZERS_PARALLELISM=False
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=12000

script_dir=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd -P)
project_root=$(dirname "$(dirname "$script_dir")")
echo "script_dir" ${script_dir}

deepspeed_config_file=${project_root}/misc/accelerate_config.yaml

accelerate launch  --mixed_precision bf16 --num_processes 2 --config_file ${deepspeed_config_file} \
${script_dir}/train_edit_dpo_low_VRAM.py \
--config ${script_dir}/train_config_dpo_example.yaml \
--allow_tf32