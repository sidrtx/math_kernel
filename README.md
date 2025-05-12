### Prerequiste
For Arch Linux: 

```bash
sudo pacman -S clang openmp 
```

### Running Guide
For Running Tests: (e.g. Benchmarking BLAS lvl 1)

```bash
clang++ -std=c++17 -O2 -fopenmp -o out test/blas1.cpp 
```

### Project Structure

```
math_kernel/
│
├── include/                   # headers (template-based, public)
│   ├── blas/
│   │   ├── blas1.hpp
│   │   ├── blas2.hpp
│   │   └── blas3.hpp
│   │
│   ├── lapack/
│   │   └── lapack.hpp
│   │
│   └── autodiff/
│       ├── tensor.hpp
│       ├── ops.hpp
│       └── tape.hpp
│
├── tests/                      # unit tests, benchmarking
│   ├── blas1.cpp
│   ├── blas2.cpp
│   ├── lapack.cpp
│   ├── autodiff.cpp
│   └── benchmark_results.txt
```