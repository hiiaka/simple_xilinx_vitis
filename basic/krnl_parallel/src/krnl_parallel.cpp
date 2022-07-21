#include "hls_stream.h"

static const int HEIGHT = 24;
static const int WIDTH = 24;
static const int DATA_SIZE = HEIGHT * WIDTH;
static const float THRESHOLD = 0.5;

extern "C" {
void krnl_parallel(const float *input, float *output) {

    hls::stream<int> strm;
    hls::stream<int> strm_1;
    hls::stream<int> strm_2;
    int array1[128];
    int array2[128];

#pragma HLS DATAFLOW
    Task_1: for (int i = 0 ; i < DATA_SIZE ; i++) {
    #pragma HLS PIPELINE
        strm.write(input[i]);
    }

    Task_2: for (int i = 0 ; i < DATA_SIZE ; i++) {
    #pragma HLS PIPELINE
        int x = strm.read();
        strm_1.write(x);
        strm_2.write(x);
    }

    Task_3: for (int i = 0 ; i < DATA_SIZE ; i++) {
    #pragma HLS PIPELINE
        array1[i] = strm_1.read() + 10;
    }

    Task_4: for (int i = 0 ; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        array2[i] = strm_2.read() * 2;
    }

    Task_5: for ( int i = 0; i < DATA_SIZE; i++ ) {
    #pragma HLS PIPELINE
        output[i] = array1[i] + array2[i];
    }

}
}
