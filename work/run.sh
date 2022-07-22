#!/bin/bash -eu

source ./env.sh

# make sysroots
# $XILINX/common_image/2022.1/xilinx-zynqmp-common-v2022.1/sdk.sh -y -d /tmp/tmp -p

export ROOTFS=${XILINX}/common_image/2022.1/xilinx-zynqmp-common-v2022.1
export SRC='../basic/krnl_add_cyclic/src'
export SRCNAME='krnl_add_cyclic'
#export TARGET='sw_emu'
export TARGET='hw'

mkdir work
pushd work

aarch64-linux-gnu-g++ -Wall -g -std=c++11 ../${SRC}/main.cpp -o app.exe -I${SYSROOT}/usr/include/xrt -L${SYSROOT}/usr/lib -lOpenCL -lpthread -lrt -lstdc++ --sysroot=${SYSROOT}
v++ -c -t ${TARGET} --platform ${DEVICE} -k ${SRCNAME} -I../${SRC} ../${SRC}/${SRCNAME}.cpp -o ${SRCNAME}.xo --config ../run.cfg
v++ -l -t ${TARGET} --platform ${DEVICE} ./${SRCNAME}.xo -o ${SRCNAME}.xclbin --config ../run.cfg
v++ -p -t ${TARGET} --platform ${DEVICE} ./${SRCNAME}.xclbin --package.out_dir package --package.rootfs ${ROOTFS}/rootfs.ext4 --package.kernel_image ${ROOTFS}/Image --config ../run.cfg

popd