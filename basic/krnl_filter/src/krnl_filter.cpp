static const int HEIGHT = 24;
static const int WIDTH = 24;
static const int DATA_SIZE = HEIGHT * WIDTH;
static const int FILTER_SIZE = 3;
static const float FILTER[FILTER_SIZE][FILTER_SIZE] = {{0.0, -1.0, 0.0}, {-1.0, 4.0, -1.0}, {0.0, -1.0, 0.0}};

extern "C" {
void krnl_filter(const float *input, float *output) {

    float img1[HEIGHT][WIDTH];
    float img2[HEIGHT][WIDTH];
#pragma HLS STREAM variable=img1
#pragma HLS STREAM variable=img2

#pragma HLS DATAFLOW
    Read: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / WIDTH;
        int w = i % WIDTH;
        img1[h][w] = input[i];
    }

    Filter: for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            img2[i][j] = 0.0;
            for (int k = 0; k < FILTER_SIZE; k++) {
                for (int l = 0; l < FILTER_SIZE; l++) {
                    int h = i + k - (int)(FILTER_SIZE / 2);
                    int w = j + l - (int)(FILTER_SIZE / 2);
                    if (h < 0 || w < 0 || h >= HEIGHT || w >= WIDTH) continue;
                    img2[i][j] += FILTER[k][l] * img1[h][w];
                }
            }
        }
    }

    Write: for (int i = 0; i < DATA_SIZE; i++) {
    #pragma HLS PIPELINE
        int h = i / WIDTH;
        int w = i % WIDTH;
        output[i] = img2[h][w];
    }
}
}
