#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

// Computes the dot product of two vectors
template <typename T> 
T dot(const vector<T> &x, const vector<T> &y) {
    if (x.size() != y.size()) // Check for equal sizes
        throw invalid_argument("Vectors must be of same length!");

    T result = T(0);

#pragma omp parallel for reduction(+ : result) // Parallel reduction
    for (size_t i = 0; i < x.size(); ++i)
        result += x[i] * y[i]; // Element-wise multiplication

    return result; 
}

// Performs AXPY operation: y = alpha * x + y
template <typename T> 
void axpy(double alpha, const vector<T> &x, vector<T> &y) {
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = alpha * x[i] + y[i];
}

// Scales a vector by a scalar: x = alpha * x
template <typename T> 
void scal(T alpha, vector<T> &x) {
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i)
        x[i] *= alpha;
}

// Swaps the contents of two vectors
template <typename T> 
void swap(vector<T> &x, vector<T> &y) {
    if (x.size() != y.size()) // Check for equal sizes
        throw invalid_argument("Vectors must be of same length!");

#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i)
        swap(x[i], y[i]);
}

// Copies the contents of one vector to another
template <typename T> 
void copy(const vector<T> &x, vector<T> &y) {
    if (x.size() != y.size()) // Check for equal sizes
        throw invalid_argument("Vectors must be of same length!");

#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i];
}

// Computes the L2 norm (Euclidean norm) of a vector
template <typename T> 
T nrm2(const vector<T> &x) {
    T sum_sq = T(0);

#pragma omp parallel for reduction(+ : sum_sq) // Parallel reduction
    for (size_t i = 0; i < x.size(); ++i)
        sum_sq += x[i] * x[i];

    return sqrt(sum_sq);
}

// Computes the L1 norm (sum of absolute values) of a vector
template <typename T> 
T nrm1(const vector<T> &x) {
    T sum_abs = T(0);

#pragma omp parallel for reduction(+ : sum_abs) // Parallel reduction
    for (size_t i = 0; i < x.size(); ++i)
        sum_abs += abs(x[i]);

    return sum_abs;
}

// Finds the index of the element with the maximum absolute value
template <typename T> 
size_t iamax(const vector<T> &x) {
    size_t max_index = 0;
    T max_val = abs(x[0]);

#pragma omp parallel
    {
        size_t local_max_index = 0;
        T local_max_val = abs(x[0]);

#pragma omp for
        for (size_t i = 0; i < x.size(); ++i) {
            T abs_val = abs(x[i]);
            if (abs_val > local_max_val) {
                local_max_val = abs_val;
                local_max_index = i;
            }
        }

#pragma omp critical // Update global max in a critical section
        {
            if (local_max_val > max_val) {
                max_val = local_max_val;
                max_index = local_max_index;
            }
        }
    }

    return max_index;
}

// Finds the index of the element with the minimum absolute value
template <typename T> 
size_t iamin(const vector<T> &x) {
    size_t min_index = 0;
    T min_val = abs(x[0]);

#pragma omp parallel
    {
        size_t local_min_index = 0;
        T local_min_val = abs(x[0]);

#pragma omp for
        for (size_t i = 0; i < x.size(); ++i) {
            T abs_val = abs(x[i]);
            if (abs_val < local_min_val) {
                local_min_val = abs_val;
                local_min_index = i;
            }
        }

#pragma omp critical // Update global min in a critical section
        {
            if (local_min_val < min_val) {
                min_val = local_min_val;
                min_index = local_min_index;
            }
        }
    }

    return min_index;
}
