#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/cifar10/nni_controller_cifar10.py \
  --search_for="macro" \
  --reset_output_dir \
  --output_dir="outputs" \
  --batch_size=128 \
  --num_epochs=310 \
  --train_data_size=45000 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_num_layers=12 \
  --child_out_filters=36 \
  --child_num_branches=6 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_num_aggregate=20 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.5 \
  --controller_op_tanh_reduce=2.5 \
  --controller_skip_target=0.4 \
  --controller_skip_weight=0.8 \
  "$@"

