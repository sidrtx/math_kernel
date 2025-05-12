#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

/**
 * @brief General Matrix-Vector Multiply: y = alpha*A*x + beta*y
 *
 * @param alpha scalar.
 * @param A (m x n) Matrix.
 * @param x Vector of size n.
 * @param beta scalar.
 * @param y Vector of size m.
 */
template <typename T>
void gemv(T alpha, const vector<vector<T>> &A, const vector<T> &x, T beta, vector<T> &y) {
    size_t m = A.size();    // row size
    size_t n = A[0].size(); // col size

    if (x.size() != n || y.size() != m)
        throw invalid_argument("Matrix dimensions do not match vector sizes.");

#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        T temp = T(0);
        for (size_t j = 0; j < n; ++j)
            temp += A[i][j] * x[j];
        y[i] = alpha * temp + beta * y[i];
    }
}

/**
 * @brief Symmetric Matrix-Vector Multiply: y = alpha*A*x + beta*y
 *
 * @param alpha scalar.
 * @param A (m x n) Matrix.
 * @param x Vector of size n.
 * @param beta scalar.
 * @param y Vector of size m.
 */
template <typename T>
void symv(T alpha, const vector<vector<T>> &A, const vector<T> &x, T beta, vector<T> &y) {
    size_t m = A.size();
    size_t n = A[0].size();

    if (x.size() != n || y.size() != m)
        throw invalid_argument("Matrix dimensions do not match vector sizes.");

#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        T temp = T(0);
        for (size_t j = 0; j < n; ++j) {
            if (i <= j) // use symmetry, A[i][j] = A[j][i]
                temp += A[i][j] * x[j];
            else
                temp += A[j][i] * x[j];
        }
        y[i] = alpha * temp + beta * y[i];
    }
}

/**
 * @brief Triangular Matrix-Vector Multiply: y = L * x
 *
 * @param L lower triangular matrix.
 * @param x vector.
 */
template <typename T> 
void trmv(const vector<vector<T>> &L, const vector<T> &x, vector<T> &y) {
    size_t m = L.size();

    if (x.size() != m || y.size() != m)
        throw invalid_argument("Matrix and Vector sizes do not match.");

#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        T temp = T(0);
        for (size_t j = 0; j <= i; ++j)
            temp += L[i][j] * x[j];
        y[i] = temp;
    }
}