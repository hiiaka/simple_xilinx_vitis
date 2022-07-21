#!/bin/bash -eu

source ./env.sh

# make sysroots
# $XILINX/common_image/2022.1/xilinx-zynqmp-common-v2022.1/sdk.sh -y -d /tmp/tmp -p

export ROOTFS=${XILINX}/common_image/2022.1/xilinx-zynqmp-common-v2022.1
export SRC='../basic/krnl_add/src'
#export TARGET='sw_emu'
export TARGET='hw'

mkdir work
pushd work

aarch64-linux-gnu-g++ -Wall -g -std=c++11 ../${SRC}/main.cpp -o app.exe -I${SYSROOT}/usr/include/xrt -L${SYSROOT}/usr/lib -lOpenCL -lpthread -lrt -lstdc++ --sysroot=${SYSROOT}
v++ -c -t ${TARGET} --platform ${DEVICE} -k krnl_add -I../${SRC} ../${SRC}/krnl_add.cpp -o krnl_add.xo
v++ -l -t ${TARGET} --platform ${DEVICE} ./krnl_add.xo -o krnl_add.xclbin
v++ -p -t ${TARGET} --platform ${DEVICE} ./krnl_add.xclbin --package.out_dir package --package.rootfs ${ROOTFS}/rootfs.ext4 --package.kernel_image ${ROOTFS}/Image

popd