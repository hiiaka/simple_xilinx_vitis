extern "C" {
void krnl_multi(const int *input, int *output, int size) {

    Multi: for (int i = 0; i < size; i++) {
        output[i] = input[i] * 2;
    }
}
}
