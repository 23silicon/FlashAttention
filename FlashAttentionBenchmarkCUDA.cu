#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

extern "C" void solve_flash(const half* Q, const half* K, const half* V, half* output, int M, int N, int d);

void print_matrix_slice(const std::string& name, const std::vector<half>& h_mat, int rows, int cols, int d_model) {
    std::cout << "\n--- " << name << " (4x4 corner) ---" << std::endl;
    for (int i = 0; i < 4 && i < rows; ++i) {
        for (int j = 0; j < 4 && j < d_model; ++j) {
            float val = __half2float(h_mat[i * d_model + j]);
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "..." << std::endl;
    }
    std::cout << "..." << std::endl;
}

void initialize_random_half(std::vector<half>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = __float2half(dis(gen));
    }
}

int main() {
    // Profiling size: 2<<16 | Debugging size: 16
    int M = 2<<15, N = 2<<15, d = 128;
    size_t num_elements = M * d;
    size_t matrix_size = num_elements * sizeof(half);

    std::vector<half> h_Q(num_elements), h_K(num_elements), h_V(num_elements), h_O(num_elements);
    
    initialize_random_half(h_Q);
    initialize_random_half(h_K);
    initialize_random_half(h_V);

    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_K, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_V, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_O, matrix_size));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), matrix_size, cudaMemcpyHostToDevice));

    std::cout << "Executing FlashAttention (M=" << M << ", N=" << N << ", d=" << d << ")" << std::endl;

    solve_flash(d_Q, d_K, d_V, d_O, M, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, matrix_size, cudaMemcpyDeviceToHost));

    // The full view of the transformation
    print_matrix_slice("Input Q", h_Q, M, d, d);
    print_matrix_slice("Input K", h_K, N, d, d);
    print_matrix_slice("Input V", h_V, N, d, d);
    print_matrix_slice("Output O (Result)", h_O, M, d, d);

    std::cout << "\nVerification Note:" << std::endl;
    std::cout << "- If Softmax is correct, O[i] is a weighted average of rows in V." << std::endl;
    std::cout << "- Since inputs were [-1, 1], Output O should also reside within ~[-1, 1]." << std::endl;

    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));

    return 0;
}
