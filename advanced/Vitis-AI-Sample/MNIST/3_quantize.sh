#!/bin/bash

vai_q_tensorflow quantize \
  --input_frozen_graph ./outputs/frozen/frozen_mnist.pb \
  --input_nodes        mnist_input \
  --input_shapes       ?,28,28,1 \
  --output_nodes       dense_1/Softmax \
  --output_dir         ./outputs/quantize \
  --input_fn           input_fn.calib_input \
  --calib_iter         100
