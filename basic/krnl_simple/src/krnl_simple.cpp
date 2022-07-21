#define DATA_SIZE 1000

extern "C" {
void krnl_multi(const int *input, int *output, int size) {

    Multi: for (int i = 0; i < size; i++) {
        output[i] = input[i] * 2;
    }
}

void krnl_add(const int *input, int *output, int size) {

    Add: for (int i = 0; i < size; i++) {
        output[i] = input[i] + 10;
    }
}

void krnl_simple(const int *input, int *output, int size) {

    int tmp[DATA_SIZE];

    krnl_add(input,tmp, size);
    krnl_multi(tmp, output, size);
}
}
