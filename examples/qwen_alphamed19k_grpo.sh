#!/bin/bash
export PATH=/opt/conda/bin:/opt/conda/condabin:/etc/inspire/node/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
source ~/.bashrc
conda init
conda activate verl
source activate verl

set -x
ray stop
export WANDB_MODE=offline

export WANDB_API_KEY=

export PYTHONUNBUFFERED=1

MODEL_PATH=llms/Qwen2.5-7B-Instruct  # replace it with your local file path



OFF_LOAD=false
PROJ_NAME=XXX
EXP_NAME=qwen_alphamed_19k_vinilla_grpo
ADA_METHOD=FAST
mkdir -p tensorboard_log/$PROJ_NAME/$EXP_NAME


GPUNUM=8

ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUNUM --port 8263
python3 -m verl.trainer.main \
    config=examples/config_grpo.yaml \
    data.debug=false \
    data.debug_data_size=100 \
    data.train_files=MedicalDataset/AlphaMed19K/train19k.parquet \
    data.val_files=MedicalDataset/AlphaMedTest/updated_test.json \
    data.prompt_key=question \
    data.answer_key=answer \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.val_batch_size=512 \
    data.rollout_batch_size=256 \
    worker.actor.global_batch_size=128 \
    data.format_prompt=examples/format_prompt/box.jinja \
    data.filter_overlong_prompts=false \
    algorithm.use_entropy=false \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.offload.offload_params=$OFF_LOAD \
    worker.actor.offload.offload_optimizer=$OFF_LOAD \
    worker.actor.micro_batch_size_per_device_for_update=32 \
    worker.actor.micro_batch_size_per_device_for_experience=64 \
    worker.actor.model.model_path=$MODEL_PATH \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=0.99 \
    worker.rollout.gpu_memory_utilization=0.65 \
    worker.rollout.tensor_parallel_size=4 \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/box.py:compute_score \
    worker.reward.length_reward_method=$ADA_METHOD \
    trainer.max_steps=300 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUNUM \
    trainer.val_freq=50 \
    trainer.val_before_train=true \
    trainer.val_only=false \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=50 \
    trainer.save_limit=-1 \
    2>&1 | tee tensorboard_log/$PROJ_NAME/$EXP_NAME/log.txt

    
