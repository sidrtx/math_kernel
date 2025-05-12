#include "../include/blas/blas3.hpp" // Include the BLAS3 header
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;

void benchmark_blas3() {
    const size_t m = 1000; // Rows of the matrix
    const size_t n = 1000; // Columns of the matrix
    const size_t k = 1000; // Shared dimension for GEMM

    // Initialize matrices and scalars
    vector<vector<double>> A(m, vector<double>(k, 1.0)); // Matrix A filled with 1.0
    vector<vector<double>> B(k, vector<double>(n, 1.0)); // Matrix B filled with 1.0
    vector<vector<double>> C(m, vector<double>(n, 0.0)); // Matrix C initialized to 0.0
    vector<vector<double>> L(m, vector<double>(m, 0.0)); // Lower triangular matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            L[i][j] = 1.0; // Fill lower triangular part with 1.0
        }
    }

    double alpha = 2.0;
    double beta = 1.0;

    // Measure GEMM performance
    auto start = chrono::high_resolution_clock::now();
    gemm<double>(alpha, A, B, beta, C);
    auto end = chrono::high_resolution_clock::now();
    cout << "GEMM Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure TRSM performance
    vector<vector<double>> B_trsm = B; // Copy of B for TRSM
    start = chrono::high_resolution_clock::now();
    trsm<double>(L, B_trsm);
    end = chrono::high_resolution_clock::now();
    cout << "TRSM Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure TRMM performance
    vector<vector<double>> B_trmm = B; // Copy of B for TRMM
    start = chrono::high_resolution_clock::now();
    trmm<double>(L, B_trmm, alpha);
    end = chrono::high_resolution_clock::now();
    cout << "TRMM Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure SYMM performance
    vector<vector<double>> C_symm = C; // Copy of C for SYMM
    start = chrono::high_resolution_clock::now();
    symm<double>(L, B, C_symm, alpha, beta);
    end = chrono::high_resolution_clock::now();
    cout << "SYMM Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
}

int main() {
    benchmark_blas3();
    return 0;
}