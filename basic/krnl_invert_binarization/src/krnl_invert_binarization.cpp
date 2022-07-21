static const int HEIGHT = 24;
static const int WIDTH = 24;
static const int DATA_SIZE = HEIGHT * WIDTH;
static const float THRESHOLD = 0.5;

extern "C" {
void krnl_invert_binarization(const float *input, float *output) {

    float img1[HEIGHT][WIDTH];
    float img2[HEIGHT][WIDTH];
    float img3[HEIGHT][WIDTH];
#pragma HLS STREAM variable=img1
#pragma HLS STREAM variable=img2
#pragma HLS STREAM variable=img3

#pragma HLS DATAFLOW
    Read: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / WIDTH;
        int w = i % WIDTH;
        img1[h][w] = input[i];
    }

    Binary: for (int i = 0; i < HEIGHT; i++) {
    #pragma HLS PIPELINE
        for (int j = 0; j < WIDTH; j++) {
        #pragma HLS PIPELINE
            float tmp = img1[i][j];
            if (tmp >= THRESHOLD) {
                img2[i][j] = 1.0;
            } else {
                img2[i][j] = 0.0;
            }
        }
    }

    Invert: for (int i = 0; i < HEIGHT; i++) {
    #pragma HLS PIPELINE
        for (int j = 0; j < WIDTH; j++) {
        #pragma HLS PIPELINE
            img3[i][j] = 1.0 - img2[i][j];
        }
    }

    Write: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / WIDTH;
        int w = i % WIDTH;
        output[i] = img3[h][w];
    }
}
}
