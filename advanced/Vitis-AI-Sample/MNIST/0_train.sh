#!/bin/bash

rm -rf ./outputs/model
mkdir -p ./outputs
mkdir ./outputs/model

python3 ./train.py