#include <iostream>
#include <string>
import squint;
using namespace squint;
using namespace squint::quantities;

// Power iteration method to find the dominant eigenvector and eigenvalue
template <typename T, int N>
tensor<T, N> powerIteration(const tensor<T, N, N> &A, int maxIterations = 1000, T tolerance = 1e-6) {
    tensor<T, N> v = tensor<T, N>::random();
    T lambda = 0.0;

    for (int i = 0; i < maxIterations; ++i) {
        auto Av = A * v;
        auto newLambda = dot(Av.transpose(), v);

        if (i > 0 && fabs(newLambda - lambda) < tolerance)
            break;

        lambda = newLambda;
        v = Av/norm(Av);
    }

    return v;
}

int main() {
    dmat2 A = dmat2({0.5, 0.2, 0.5, 0.8});
    // Find the dominant eigenvector and eigenvalue
    auto dominantEigenvector = powerIteration(A);
    auto dominantEigenvalue = dot((A * dominantEigenvector).transpose(), dominantEigenvector);

    // Display results
    std::cout << "Dominant Eigenvector:\n" << dominantEigenvector << "\n\n";
    std::cout << "Dominant Eigenvalue:\n" << dominantEigenvalue << "\n";

    // invert A
    auto Ainv = inv(A);
    dominantEigenvector = powerIteration(Ainv);
    dominantEigenvalue = dot((Ainv * dominantEigenvector).transpose(), dominantEigenvector);

    // Display results
    std::cout << "Dominant Eigenvector:\n" << dominantEigenvector << "\n\n";
    std::cout << "Dominant Eigenvalue:\n" << dominantEigenvalue << "\n";
    return 0;
}