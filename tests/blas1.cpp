#include "../include/blas/blas1.hpp" // Include the BLAS1 header
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;

void test_performance() {
    const size_t N = 1000000; // Size of the vectors
    vector<double> x(N, 1.0); // Initialize vector x with 1.0
    vector<double> y(N, 2.0); // Initialize vector y with 2.0

    // Measure dot product performance
    auto start = chrono::high_resolution_clock::now();
    double dot_result = dot(x, y);
    auto end = chrono::high_resolution_clock::now();
    cout << "Dot Product Result: " << dot_result << endl;
    cout << "Dot Product Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure axpy performance
    start = chrono::high_resolution_clock::now();
    axpy(2.0, x, y);
    end = chrono::high_resolution_clock::now();
    cout << "AXPY Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure scal performance
    start = chrono::high_resolution_clock::now();
    scal(2.0, x);
    end = chrono::high_resolution_clock::now();
    cout << "SCAL Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure nrm2 performance
    start = chrono::high_resolution_clock::now();
    double nrm2_result = nrm2(x);
    end = chrono::high_resolution_clock::now();
    cout << "L2 Norm Result: " << nrm2_result << endl;
    cout << "L2 Norm Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // Measure iamax performance
    start = chrono::high_resolution_clock::now();
    size_t iamax_result = iamax(x);
    end = chrono::high_resolution_clock::now();
    cout << "IAMAX Result: " << iamax_result << endl;
    cout << "IAMAX Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
}

int main() {
    test_performance();
    return 0;
}