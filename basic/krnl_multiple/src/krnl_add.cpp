extern "C" {
void krnl_add(const int *input, int *output, int size) {

    Add: for (int i = 0; i < size; i++) {
        output[i] = input[i] + 10;
    }
}
}
