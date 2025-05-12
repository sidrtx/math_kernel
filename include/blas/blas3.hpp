#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

/**
 * @brief General Matrix-Matrix Multiply: C = alpha*A*B + beta*C
 *
 * @param alpha is scalar.
 * @param beta is scalar.
 * @param A (m x k) Matrix.
 * @param B (k x n) Matrix.
 * @param C (m x n) Matrix.
 */
template <typename T>
void gemm(T alpha, const vector<vector<T>> &A, const vector<vector<T>> &B, T beta,
          vector<vector<T>> &C) {

    size_t m = A.size();    // rows in A
    size_t k = A[0].size(); // cols in A == rows in B
    size_t n = B[0].size(); // cols in B

    if (B.size() != k || C.size() != m || C[0].size() != n) {
        throw invalid_argument("Matrix dimensions do not match for GEMM");
    }

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {     // loop over rows of A
        for (size_t j = 0; j < n; ++j) { // loop over cols of B
            T sum = T(0);
            for (size_t l = 0; l < k;
                 ++l) // loop over shared dimension (k) to accumulate dot product
                sum += A[i][l] * B[l][j];
            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}

/**
 * @brief Triangular Solve Matrix: Solving for X where AX = B or XA = B
 *
 * @param A is a lower triangular matrix (m x m)
 * @param B is a general matrix (m x n)
 * @param X is the matrix we want to solve for (m x n)
 */
template <typename T> 
void trsm(const vector<vector<T>> &A, vector<vector<T>> &B) {
    size_t m = A.size();
    size_t n = B[0].size();

    if (A.size() != A[0].size() || A.size() != B.size()) {
        throw invalid_argument("Invalid Matrix Dimensions for TRSM.");
    }

    // Assume A is lower triangle, solve AX = B -> overwrite B with X
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = B[i][j];
            for (size_t k = 0; k < i; ++k)
                sum -= A[i][k] * B[k][j];
            B[i][j] = sum / A[i][j];
        }
    }
}

/**
 * @brief Triangular Matrix Matrix Multiply: B := alpha * op(A) * B
 *
 * @param A is a triangular matrix
 * @param B is a general matrix (overwritten)
 * @param alpha is Scalar
 * @param op(A) means either A.T (Transpose) or A.H (Hermitian)
 */
template <typename T> 
void trmm(const vector<vector<T>> &A, vector<vector<T>> &B, T alpha) {

    size_t m = A.size();    // A is m x m
    size_t n = B[0].size(); // B is m x n

    if (A.size() != A[0].size() || A.size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for TRMM.");
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t k = 0; k <= i; ++k) { // lower triangular A
                sum += A[i][k] * B[k][j];
            }
            B[i][j] = alpha * sum;
        }
    }
}

/**
 * @brief Performs symmetric matrix-matrix multiplication: C := alpha * A * B + beta * C.
 *
 * @param A Symmetric matrix of size m x m.
 * @param B General matrix of size m x n.
 * @param C General matrix of size m x n (overwritten with the result).
 * @param alpha Scalar multiplier for the product of A and B.
 * @param beta Scalar multiplier for the matrix C.
 */
template <typename T>
void symm(const vector<vector<T>> &A, const vector<vector<T>> &B, vector<vector<T>> &C, T alpha,
          T beta) {

    size_t m = A.size();    // A is m x m
    size_t n = B[0].size(); // B is m x n

    if (A.size() != A[0].size() || A.size() != B.size() || C.size() != m || C[0].size() != n) {
        throw std::invalid_argument("Matrix dimensions do not match for SYMM.");
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t k = 0; k < m; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}
