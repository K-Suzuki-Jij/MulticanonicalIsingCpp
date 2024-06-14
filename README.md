### C++ Tests
```bash
g++ -Og -std=c++17 tests/cpp_tests/test_main.cpp -Xclang -fopenmp -lgtest -lomp -o tests/cpp_tests/cpp_test -I $(brew --prefix eigen)/include/eigen3/ -I $(brew --prefix libomp)/include -I $(brew --prefix googletest)/include -L $(brew --prefix libomp)/lib -L $(brew --prefix googletest)/lib
```

