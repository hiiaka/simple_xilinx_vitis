#!/bin/bash

vai_q_tensorflow quantize \
  --input_frozen_graph ./outputs/frozen/frozen_yolov3.pb \
  --input_nodes        input_1 \
  --input_shapes       ?,416,416,3 \
  --output_nodes       conv2d_59/BiasAdd,conv2d_67/BiasAdd,conv2d_75/BiasAdd \
  --output_dir         ./outputs/quantize \
  --input_fn           input_fn.calib_input \
  --calib_iter         10
