#!/bin/bash

rm -rf ./outputs/frozen
mkdir ./outputs/frozen

python ./freeze_graph.py \
  --input_binary true \
  --input_meta_graph ./outputs/tf_save/mnist.meta \
  --input_checkpoint ./outputs/tf_save/mnist \
  --output_node_names dense_1/Softmax \
  --output_graph ./outputs/frozen/frozen_mnist.pb
