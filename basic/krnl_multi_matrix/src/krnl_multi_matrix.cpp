static const int A_ROWS = 24;
static const int A_COLS = 24;
static const int B_ROWS = 24;
static const int B_COLS = 24;
static const int Y_ROWS = A_ROWS;
static const int Y_COLS = B_COLS;
static const int A_DATA_SIZE = A_ROWS * A_COLS;
static const int B_DATA_SIZE = B_ROWS * B_COLS;
static const int Y_DATA_SIZE = Y_ROWS * Y_COLS;

extern "C" {
void krnl_multi_matrix(const float *input_A, const float *input_B, float *output) {

    float tmp_A[A_ROWS][A_COLS];
    float tmp_B[B_ROWS][B_COLS];
    float tmp_Y[Y_ROWS][Y_COLS];

    Read_A: for (int i = 0; i < A_DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / A_COLS;
        int w = i % A_COLS;
        tmp_A[h][w] = input_A[i];
    }

    Read_B: for (int i = 0; i < B_DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / B_COLS;
        int w = i % B_COLS;
        tmp_B[h][w] = input_B[i];
    }

    Multi: for (int i = 0; i < Y_COLS; i++) {
    #pragma HLS PIPELINE
        for (int j = 0; j < Y_ROWS; j++) {
        #pragma HLS PIPELINE
            tmp_Y[j][i] = 0.0;
            for (int k = 0; k < B_ROWS; k++) {
            #pragma HLS PIPELINE
                tmp_Y[j][i] += tmp_A[j][k] * tmp_B[k][i];
            }
        }
    }

    Write: for (int i = 0; i < Y_DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / Y_COLS;
        int w = i % Y_COLS;
        output[i] = tmp_Y[h][w];
    }
}
}
