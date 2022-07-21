export XILINX=/usr/tools/xilinx
export XILINX_XRT=/opt/xilinx/xrt
export PYTHONPATH=''
source $XILINX/Vitis/2022.1/settings64.sh
source $XILINX_XRT/setup.sh

export LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
export PATH=$XILINX_XRT/bin:$PATH
export PYTHONPATH=$XILINX_XRT/python:$PYTHONPATH
export DEVICE=$XILINX/Vitis/2022.1/base_platforms/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm
export SYSROOT=/tmp/tmp/sysroots/cortexa72-cortexa53-xilinx-linux
export EDGE_COMMON_SW=$XILINX/common_image/2022.1/xilinx-zynqmp-common-v2022.1
