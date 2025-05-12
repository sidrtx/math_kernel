#include "../include/blas/blas2.hpp"  // Include the BLAS2 header
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;

void test_performance() {
    const size_t m = 1000; // Rows of the matrix
    const size_t n = 1000; // Columns of the matrix

    // Initialize matrix A and vectors x, y
    vector<vector<double>> A(m, vector<double>(n, 1.0)); // Matrix A filled with 1.0
    vector<double> x(n, 1.0);                            // Vector x filled with 1.0
    vector<double> y(m, 2.0);                            // Vector y filled with 2.0

    // Declare end variable
    chrono::high_resolution_clock::time_point start, end;

    // Measure GEMV performance
    start = chrono::high_resolution_clock::now();
    gemv<double>(2.0, A, x, 1.0, y); // Explicitly specify template type
    end = chrono::high_resolution_clock::now();
    cout << "GEMV Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure SYMV performance
    start = chrono::high_resolution_clock::now();
    symv<double>(2.0, A, x, 1.0, y); // Explicitly specify template type
    end = chrono::high_resolution_clock::now();
    cout << "SYMV Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure TRMV performance
    vector<vector<double>> L(m, vector<double>(m, 0.0)); // Lower triangular matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            L[i][j] = 1.0; // Fill lower triangular part with 1.0
        }
    }
    start = chrono::high_resolution_clock::now();
    trmv<double>(L, x, y); // Explicitly specify template type
    end = chrono::high_resolution_clock::now();
    cout << "TRMV Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
}

int main() {
    test_performance();
    return 0;
}