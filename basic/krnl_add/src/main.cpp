#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

static const int DATA_SIZE = 128;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                device = devices[0];
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }

    cl::Context context(device);
    char* xclbinFilename = argv[1];
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);

    size_t a_size_in_bytes = DATA_SIZE * sizeof(float);
    size_t b_size_in_bytes = DATA_SIZE * sizeof(float);
    size_t y_size_in_bytes = DATA_SIZE * sizeof(float);
    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, a_size_in_bytes);
    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, b_size_in_bytes);
    cl::Buffer buffer_y(context, CL_MEM_WRITE_ONLY, y_size_in_bytes);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    float *ptr_a = (float *)q.enqueueMapBuffer(buffer_a, CL_TRUE, CL_MAP_WRITE, 0, a_size_in_bytes);
    float *ptr_b = (float *)q.enqueueMapBuffer(buffer_b, CL_TRUE, CL_MAP_WRITE, 0, b_size_in_bytes);
    float *ptr_y = (float *)q.enqueueMapBuffer(buffer_y, CL_TRUE, CL_MAP_READ, 0, y_size_in_bytes);
    for (int i = 0; i < DATA_SIZE; i++) { ptr_a[i] = (float)1.0; }
    for (int i = 0; i < DATA_SIZE; i++) { ptr_b[i] = (float)2.0; }

    unsigned int sec;
    int nsec;
    double d_sec;
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_REALTIME, &start_time);

    cl::Kernel krnl(program, "krnl_add");
    int narg = 0;
    krnl.setArg(narg++, buffer_a);
    krnl.setArg(narg++, buffer_b);
    krnl.setArg(narg++, buffer_y);

    q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
    q.enqueueTask(krnl);
    q.enqueueMigrateMemObjects({buffer_y}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    clock_gettime(CLOCK_REALTIME, &end_time);
    sec = end_time.tv_sec - start_time.tv_sec;
    nsec = end_time.tv_nsec - start_time.tv_nsec;
    d_sec = (double)sec + (double)nsec / (1000 * 1000 * 1000);

    for (int i = 0; i < DATA_SIZE; i++) { std::cout << "A[" << i << "] : " << ptr_a[i] << std::endl; }
    for (int i = 0; i < DATA_SIZE; i++) { std::cout << "B[" << i << "] : " << ptr_b[i] << std::endl; }
    for (int i = 0; i < DATA_SIZE; i++) { std::cout << "Y[" << i << "] : " << ptr_y[i] << std::endl; }

    q.enqueueUnmapMemObject(buffer_a, ptr_a);
    q.enqueueUnmapMemObject(buffer_b, ptr_b);
    q.enqueueUnmapMemObject(buffer_y, ptr_y);
    q.finish();

    std::cout << "time : " << d_sec << std::endl;

    return 0;
}
