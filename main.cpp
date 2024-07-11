#include <cblas.h>
#include <iostream>

int main() {
    double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double C[9] = {0.0};

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1.0, A, 2, B, 2, 0.0, C, 3);

    for (int i = 0; i < 9; i++) {
        std::cout << C[i] << " ";
        if ((i + 1) % 3 == 0)
            std::cout << std::endl;
    }

    return 0;
}