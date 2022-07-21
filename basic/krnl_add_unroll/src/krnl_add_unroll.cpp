static const int DATA_SIZE = 128;

extern "C" {
void krnl_add_unroll(const float *input_A, const float *input_B, float *output) {

    float tmp_A[DATA_SIZE];
    float tmp_B[DATA_SIZE];
    float tmp_Y[DATA_SIZE];

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
