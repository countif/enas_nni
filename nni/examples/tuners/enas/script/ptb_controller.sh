#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/ptb/nni_controller_ptb.py \
  --search_for="enas" \
  --noreset_output_dir \
  --data_path="data/ptb/ptb.pkl" \
  --output_dir="outputs" \
  --batch_size=20 \
  --num_epochs=100 \
  --child_steps=1327 \
  --child_rhn_depth=12 \
  --log_every=50 \
  --controller_training \
  --controller_train_every=1 \
  --controller_lr=0.001 \
  --controller_sync_replicas \
  --controller_train_steps=100 \
  --controller_num_aggregate=10 \
  --controller_tanh_constant=2.5 \
  --controller_temperature=5.0 \
  --controller_entropy_weight=0.001 \
  --eval_every_epochs=1

