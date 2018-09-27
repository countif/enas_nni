#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/cifar10/nni_controller_cifar10.py \
  --search_for="micro" \
  --reset_output_dir \
  --output_dir="outputs" \
  --train_data_size=45000 \
  --batch_size=160 \
  --num_epochs=150 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_num_layers=6 \
  --child_out_filters=20 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=30 \
  --controller_lr=0.0035 \
  --controller_tanh_constant=1.10 \
  --controller_op_tanh_reduce=2.5 \
  "$@"

