#include <cuda_runtime.h>
#include <cmath>

#define Br 32 //canonical Q tile height name in FlashAttention paper
#define Bc 32 //canonical K/V tiles height

__global__ void flash_attention(const float* Q, const float* K, const float* V, float* output, 
                                int M, int N, int d, int Tr, int Tc, float scale) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sram[];
    //pointers to the beginning of each tile's allocated region in shared memory
    float* Qtile = sram; //rows 0-Br are for Q tile, height Br
    float* Ktile = &sram[Br * d]; //rows Br-(Br+Bc) are for K tile, height Bc
    float* Vtile = &sram[(Br + Bc) * d]; //rows (Br+Bc) to end are for Vtile, height Bc


    //loop 1: load tiles of Q into sram.
    // ***Each thread is responsible for loading 1 full row of Q into its respective tile
    if (d % 4 == 0) {
        for (int i = 0; i < (d >> 2); i++) {
            /*
            GPU doesn't retrieve a single float per query, instead it retrieves up to 32 bytes. 
            Therefore, it's very inefficient to load 4 bytes at a time into sram.
            */
            float4* Q4 = (float4*)Q;
            float4* Q4tile = (float4*)Qtile;
            Q4tile[threadIdx.x * (d >> 2) + i] = Q4[id * (d >> 2) + i];
        }
    } else {
        for (int i = 0; i < d; i++) {
            Qtile[threadIdx.x * d + i] = Q[id * d + i];
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
        if (d % 4 == 0) {
            float4* K4 = (float4*)K;
            float4* V4 = (float4*)V;
            float4* K4tile = (float4*)Ktile;
            float4* V4tile = (float4*)Vtile;            
            for (int j = 0; j < d >> 2; j++) {
                int tilerow_KV = Bc * i + threadIdx.x;
                if (tilerow_KV < N) {
                    K4tile[threadIdx.x * (d >> 2) + j] = K4[tilerow_KV * (d >> 2) + j];
                    V4tile[threadIdx.x * (d >> 2) + j] = V4[tilerow_KV * (d >> 2) + j];
                } else {
                    K4tile[threadIdx.x * (d >> 2) + j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    V4tile[threadIdx.x * (d >> 2) + j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        } else {
            for (int j = 0; j < d; j++) {
                int tilerow_KV = Bc * i + threadIdx.x;
                Ktile[threadIdx.x * d + j] = (tilerow_KV < N) ? K[tilerow_KV * d + j] : 0.0f;
                //V tile loaded by row as well to compute streamed dot product at the end
                Vtile[threadIdx.x * d + j] = (tilerow_KV < N) ? V[tilerow_KV * d + j] : 0.0f; 
            }
        }
        __syncthreads();

        //Next step: compute dot product of this thread's corresponding Q row and every Ktile row (dim: Bc x d)
        float attention_scores[Bc];
        float blockmax = -1e30f;
        #pragma unroll //compiler hint to unroll cus Bc is known at compile time
        for (int row = 0; row < Bc; row++) {
            float global_K_row = i * Bc + row; 
            if (global_K_row < N) {
                float sum = 0.0f;
                for (int col = 0; col < d; col++) {
                    sum += Qtile[threadIdx.x * d + col] * Ktile[row * d + col];
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
            attention_scores[j] = __expf(attention_scores[j]-blockmax);
            blocksum += attention_scores[j];
        }

        //part 2: calculate new global max and scaling factors
        float newmax = max(m, blockmax);
        float scale_f1 = __expf(m-newmax);
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
                acc[col] += scaled_p * Vtile[j * d + col];
            }
        }

        __syncthreads();
    }

    //divide each element by l to complete the softmax and write to output
    if (id < M) { //final matrix is M x d, this block write the entire vector of length d to row @id
        float divL = 1.0f/l; //avoid repeated division
        for (int col = 0; col < d; col++) {
            output[id * d + col] = acc[col] * divL;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    //Q is Mxd, K is Nxd, V is Nxd
    //QK^T is MxN
    //output is MxN
    int Tr = (M + Br - 1) / Br, Tc = (N + Bc - 1) / Bc; //tile counts
    dim3 threadsPerBlock (Br);
    dim3 blocksPerGrid (Tr); 
    int sram_size ((Br * d + Bc * d + Bc * d) * sizeof(float)); //1 tile of Q, 1 tile of K, 1 tile of V
    float scale = 1.0f/sqrtf(d); //scaling factor multiplied by each element before softmax

    flash_attention<<<blocksPerGrid, threadsPerBlock, sram_size>>>(Q, K, V, output, M, N, d, Tr, Tc, scale);
}

