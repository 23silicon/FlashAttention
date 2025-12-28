#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define Br 128 //canonical Q tile height name in FlashAttention paper
#define Bc 16 //canonical K/V tiles height

__global__ void flash_attention(const half* Q, const half* K, const half* V, half* output,
                                int M, int N, int d, int Tr, int Tc, float scale) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ half sram[];
    int sram_stride = d+8; //sram padding for resolving memory bank conflicts (again hopefully speeding up the kernel)

    //pointers to the beginning of each tile's allocated region in shared memory
    half* Qtile = sram; //rows 0-Br are for Q tile, height Br
    half* Ktile = &sram[Br * sram_stride]; //rows Br-(Br+Bc) are for K tile, height Bc
    half* Vtile = &sram[(Br + Bc) * sram_stride]; //rows (Br+Bc) to end are for Vtile, height Bc


    //loop 1: load tiles of Q into sram.
    // ***Each thread is responsible for loading 1 full row of Q into its respective tile
    if (id < M) {
        for (int i = 0; i < d; i++) {
            Qtile[threadIdx.x * sram_stride + i] = Q[id * d + i];
        }
    }
    __syncthreads();

    //callocs to 0, fixed size typical max for d_model is 128 and explodes to registers
    float acc[128] = {0.0f}; //O matrix accumulator
    float l = 0.0f; //running denominator sum for softmax
    float m = -1e30f; //starts with a very low number, basically -infinity

    //ENTERING MAIN LOOP: here we stream tiles of K and V
    for (int i = 0; i < Tc; i++) {
        //load K/V
        for (int j = 0; j < d; j++) {
            int tilerow_KV = Bc * i + threadIdx.x;
            if (threadIdx.x < Bc) {
                if (tilerow_KV < N) {
                    Ktile[threadIdx.x * sram_stride + j] = K[tilerow_KV * d + j];
                    Vtile[threadIdx.x * sram_stride + j] = V[tilerow_KV * d + j];
                } else {
                    Ktile[threadIdx.x * sram_stride + j] = __float2half(0.0f);
                    Vtile[threadIdx.x * sram_stride + j] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        //Next step: compute dot product of this thread's corresponding Q row and every Ktile row (dim: Bc x d)
        float attention_scores[Bc];
        float blockmax = -1e30f;
        #pragma unroll //compiler hint to unroll cus Bc is known at compile time
        for (int row = 0; row < Bc; row++) {
            int global_K_row = i * Bc + row;
            if (global_K_row < N) {
                float sum = 0.0f;
                for (int col = 0; col < d; col++) {
                    // Convert half to float for math stability
                    float q_val = __half2float(Qtile[threadIdx.x * sram_stride + col]);
                    float k_val = __half2float(Ktile[row * sram_stride + col]);
                    sum += q_val * k_val;
                }
                sum *= scale;
                blockmax = max(blockmax, sum);
                attention_scores[row] = sum;
            } else {
                //check for padded rows to set score to -inf instead of 0 because e^0 is 1, not 0.
                attention_scores[row] = -1e30f;
            }
        }

        //Most technically beefy part of this kernel: online safe softmax

        //part 1: Adjust attention scores by subtracting blockmax from each element, find running sum
        float blocksum = 0.0f;
        for (int j = 0; j < Bc; j++) {
            attention_scores[j] = __expf(attention_scores[j] - blockmax);
            blocksum += attention_scores[j];
        }

        //part 2: calculate new global max and scaling factors
        float newmax = max(m, blockmax);
        float scale_f1 = __expf(m - newmax);
        float scale_fb = __expf(blockmax - newmax);

        m = newmax;
        l = (l * scale_f1) + (blocksum * scale_fb); //apply scaling factors

        //part 3: adjust and update accumulator
        for (int col = 0; col < d; col++) {
            acc[col] *= scale_f1;
        }
        for (int j = 0; j < Bc; j++) {
            float scaled_p = attention_scores[j] * scale_fb;
            for (int col = 0; col < d; col++) {
                float v_val_inner = __half2float(Vtile[j * sram_stride + col]);
                acc[col] += scaled_p * v_val_inner;
            }
        }

        __syncthreads();
    }

    //divide each element by l to complete the softmax and write to output
    if (id < M) { //final matrix is M x d, this block write the entire vector of length d to row @id
        float divL = 1.0f / l; //avoid repeated division
        // MODIFIED: Removed vectorization logic
        for (int col = 0; col < d; col++) {
            output[id * d + col] = __float2half(acc[col] * divL);
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve_flash(const half* Q, const half* K, const half* V, half* output, int M, int N,
                      int d) {
    int Tr = (M + Br - 1) / Br, Tc = (N + Bc - 1) / Bc; //tile counts
    dim3 threadsPerBlock(Br);
    dim3 blocksPerGrid(Tr);
    //extra sram for strided access, hopefully solving sram bank conflicts
    int sram_size = (Br * (d+8) + Bc * (d+8) + Bc * (d+8)) * sizeof(half); //1 tile of Q, 1 tile of K, 1 tile of V
    float scale = 1.0f / sqrtf(d); //scaling factor multiplied by each element before softmax

    flash_attention<<<blocksPerGrid, threadsPerBlock, sram_size>>>(Q, K, V, output, M, N, d, Tr, Tc, scale);
}
