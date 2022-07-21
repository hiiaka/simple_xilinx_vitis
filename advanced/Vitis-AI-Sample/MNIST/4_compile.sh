#!/bin/bash

vai_c_tensorflow \
  -f ./outputs/quantize/deploy_model.pb \
  -a ultra96v2.json \
  -o ./outputs/compile \
  -n mnist