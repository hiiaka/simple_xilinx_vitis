#!/bin/bash

rm -rf ./outputs/frozen
mkdir ./outputs/frozen

python ./freeze_graph.py \
  --input_binary true \
  --input_meta_graph ./outputs/model/yolov3.meta \
  --input_checkpoint ./outputs/model/yolov3 \
  --output_node_names conv2d_59/BiasAdd,conv2d_67/BiasAdd,conv2d_75/BiasAdd \
  --output_graph ./outputs/frozen/frozen_yolov3.pb
