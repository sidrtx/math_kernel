### Prerequiste

```bash
sudo pacman -S clang openmp 
```

### Project Structure

```
math_kernel/
│
├── include/                       # Public headers (installable)
│   ├── math_kernel/              # Namespaced headers
│   │   ├── blas1.hpp
│   │   ├── blas2.hpp
│   │   ├── blas3.hpp
│   │   ├── lapack.hpp
│   │   └── autodiff.hpp
│
├── src/                           # Source files (internal implementation)
│   ├── blas/
│   │   ├── blas1.cpp
│   │   ├── blas2.cpp
│   │   └── blas3.cpp
│   ├── lapack/
│   │   └── lapack.cpp
│   └── autodiff/
│       ├── tensor.cpp
│       ├── ops.cpp
│       └── tape.cpp
│
├── tests/                         # Unit tests
│   ├── test_blas.cpp
│   ├── test_autodiff.cpp
│   └── test_lapack.cpp
│
├── examples/                      # Example usage and demos
│   ├── demo_blas1.cpp
│   ├── demo_autodiff.cpp
│   └── demo_nn_training.cpp
│
├── cmake/                         # CMake modules (optional)
│   └── FindOpenMP.cmake
│
├── CMakeLists.txt                 # CMake build file (top level)
├── README.md                      # Project description and usage
├── LICENSE                        # Open source license (MIT License)
└── .gitignore

```