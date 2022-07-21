static const int DATA_SIZE = 128;

extern "C" {
void krnl_add_complete(const float *input_A, const float *input_B, float *output) {

    float tmp_A[DATA_SIZE];
    float tmp_B[DATA_SIZE];
    float tmp_Y[DATA_SIZE];
#pragma HLS ARRAY_PARTITION variable=tmp_A complete
#pragma HLS ARRAY_PARTITION variable=tmp_B complete
#pragma HLS ARRAY_PARTITION variable=tmp_Y complete

    Read_A: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        tmp_A[i] = input_A[i];
    }

    Read_B: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        tmp_B[i] = input_B[i];
    }

    Add: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS UNROLL
        tmp_Y[i] = tmp_A[i] + tmp_B[i];
    }

    Write: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        output[i] = tmp_Y[i];
    }
}
}
