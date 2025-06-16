#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64

#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>

double run_dgemm_test(int n, double duration_sec) {
    using clk = std::chrono::steady_clock;
    auto start_time = clk::now();
    auto end_time = start_time + std::chrono::duration<double>(duration_sec);

    std::vector<double> A(n * n, 1.0), B(n * n, 2.0), C(n * n, 0.0);

    int64_t N = static_cast<int64_t>(n);  // ILP64 uses int64_t
    uint64_t flops = 0;

    while (clk::now() < end_time) {
        // C := alpha * A * B + beta * C
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);
        flops += 2ULL * n * n * n;
    }

    return flops / duration_sec / 1e9; // return GFLOPS
}

int main() {
    const double duration = 5.0;
    const int matrix_size = 1024; // You can adjust to increase pressure

    std::ofstream fout("accelerate_output.txt");

    auto both = [&](const std::string& msg) {
        std::cout << msg;
        fout << msg;
    };

    both("==== Accelerate DGEMM ILP64 TEST ====\n");
    both("Matrix size: " + std::to_string(matrix_size) + " x " + std::to_string(matrix_size) + "\n");

    double gflops = run_dgemm_test(matrix_size, duration);
    both("  DGEMM GFLOPS: " + std::to_string(gflops) + "\n");

    return 0;
}
