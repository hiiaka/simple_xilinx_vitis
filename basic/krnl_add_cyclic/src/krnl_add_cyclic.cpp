static const int DATA_SIZE = 128;

extern "C" {
void krnl_add_cyclic(const float *input_A, const float *input_B, float *output) {

    float tmp_A[DATA_SIZE];
    float tmp_B[DATA_SIZE];
    float tmp_Y[DATA_SIZE];
#pragma HLS ARRAY_PARTITION variable=tmp_A cyclic factor=DATA_SIZE/4
#pragma HLS ARRAY_PARTITION variable=tmp_B cyclic factor=DATA_SIZE/4
#pragma HLS ARRAY_PARTITION variable=tmp_Y cyclic factor=DATA_SIZE/4

    Read_A: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        tmp_A[i] = input_A[i];
    }

    Read_B: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        tmp_B[i] = input_B[i];
    }

    Add: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS UNROLL factor=DATA_SIZE/4
        tmp_Y[i] = tmp_A[i] + tmp_B[i];
    }

    Write: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        output[i] = tmp_Y[i];
    }
}
}
