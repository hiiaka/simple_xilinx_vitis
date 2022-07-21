#define DATA_SIZE 4096
#define ARRAY_SIZE 100

extern "C" {
void krnl_multiple_pipeline(int *input_A, int *input_B, int *output, int param) {

        int array[ARRAY_SIZE];
#pragma HLS array_partition variable=array complete

        Set: for (int i = 0 ; i < ARRAY_SIZE ; i++) {
        #pragma HLS PIPELINE
                array[i] = i + param;
        }

        Loop_1: for (int i = 0 ; i < DATA_SIZE ; i++) {
        #pragma HLS PIPELINE off
                int x_A = input_A[i];            // proc1
                int x_B = input_B[i] * 2;        // proc2

                Loop_2: for (int j = 0 ; j < ARRAY_SIZE ; j++) {
                #pragma HLS PIPELINE
                        int tmp = param * j;     // proc3-1
                        tmp += x_B;              // proc3-2
                        x_A += array[j] + tmp;   // proc3-3
                }
                output[i] = x_A + x_B;           // proc4
        }
}
}
