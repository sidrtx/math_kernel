### Prerequiste

```bash
sudo pacman -S clang openmp 
```

### Running Guide

```bash
clang++ -std=c++17 -O2 -fopenmp -o out test/blas1.cpp 
```

### Project Structure

```
math_kernel/
|
├── include/                       # Public headers
│   └── blas/
│       ├── blas1.hpp              # Template function definitions
│       ├── blas2.hpp
│       └── blas3.hpp
│
├── src/
│   ├── lapack/
│   │   └── lapack.cpp            
│   └── autodiff/
│       ├── tensor.cpp
│       ├── ops.cpp
│       └── tape.cpp
│
├── tests/                         # Time profiling
│   ├── blas1.cpp
│   ├── blas2.cpp
│   ├── blas3.cpp
│   └── benchmark_results.txt


```