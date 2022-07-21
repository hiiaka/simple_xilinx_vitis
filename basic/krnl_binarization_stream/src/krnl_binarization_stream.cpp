#include "hls_stream.h"

static const int HEIGHT = 24;
static const int WIDTH = 24;
static const int DATA_SIZE = HEIGHT * WIDTH;
static const float THRESHOLD = 0.5;

extern "C" {
void krnl_binarization_stream(const float *input, float *output) {

    hls::stream<float> img1;
    hls::stream<float> img2;

#pragma HLS DATAFLOW
    Read: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        img1.write(input[i]);
    }

    Binary: for (int i = 0; i < HEIGHT; i++) {
    #pragma HLS PIPELINE
        for (int j = 0; j < WIDTH; j++) {
        #pragma HLS PIPELINE
            float tmp = img1.read();
            if (tmp >= THRESHOLD) {
                img2.write(1.0);
            } else {
                img2.write(0.0);
            }
        }
    }

    Write: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        output[i] = img2.read();
    }
}
}
