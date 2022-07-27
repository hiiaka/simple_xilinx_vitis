#!/bin/bash -eu

python3.7 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

export MNIST_DIR=../../../advanced/Vitis-AI-Sample/MNIST

mkdir work
cp ${MNIST_DIR}/*.sh work
cp ${MNIST_DIR}/*.py work
cp -r ${MNIST_DIR}/model work
pushd work
./0_train.sh
./1_convert_to_tf.sh
./2_frozen.sh
popd

