#include <cuda_runtime.h>
#include <stdio.h>

/*
SOFTMAX
*/
//STEP 1************************************************************FIND MAX
__global__ void maximum(const float* input, float* blockmaxes, int N0, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    bool active = !(id < N0 || id >= N);
    //printf("\n---In Kernel, running thread #%d on bounds | %d %d", id, N0, N);
    float max = -INFINITY;
    for (int i = id; i < N; i += gridDim.x*blockDim.x) {
        max = fmaxf(input[i], max);
    }
    extern __shared__ float sdata[]; //allocate shared memory for blockwise reduction
    sdata[threadIdx.x] = active ? max : -INFINITY;

    __syncthreads();

    //blockwide max reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        blockmaxes[blockIdx.x] = sdata[0];
    }
}


//STEP 2********************************************REDUCE WITH MAX
__inline__ __device__ float warpReduce(float val) {
    auto mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset); //masks out inactive threads
    }
    return val;
}

// last step of softmax
__global__ void reduce(const float* input, float* total, float* max, int N0, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    bool active = !(id < N0 || id >= N); //per row

    float sum = active ? expf(input[id]-*max) : 0.0f; //can also try expf
    sum = warpReduce(sum);
    extern __shared__ float warpvals[]; //allocate shared memory for each warp
    
    int laneId = threadIdx.x&31; //finds out which lane of the warp we're in (mod by 32)
    int warpId = threadIdx.x >> 5; //finds out which warp we're in (div by 32)

    if (laneId == 0)
        warpvals[warpId] = sum;
    
    __syncthreads();
    
    //smart secondary reduction. Loads the warpvals values into the sums of a single thread
    //and then sums those up
    if (warpId == 0) {
        sum = (threadIdx.x < (blockDim.x >> 5)) ? warpvals[threadIdx.x] : 0.0f;
        sum = warpReduce(sum);
        if (threadIdx.x == 0) {
            atomicAdd(total, sum);  
            //__syncthreads(); printf("\nCALCULATED: %f\n", total[0]);
        }
    }
}

__global__ void softmax_kernel(float* input, float* total, float* max, int N0, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N0 || id >= N) return; //per row
    for (int i = id; i < N; i += gridDim.x*blockDim.x) {
        input[i] = __expf(input[i]-*max)/(*total);
    }
}

/*
MATMUL + TRANSPOSE + divide by sqrt(d), creating params for softmax
*/
#define tilesize 16
__global__ void matmul(const float* A, const float* B, float* output, int m, int n, int k, float muldiv) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float Atile[tilesize][tilesize];
    __shared__ float Btile[tilesize][tilesize];

    float value = 0.0f;
    #pragma unroll
    for (int t = 0; t < (n + tilesize - 1) / tilesize; t++) {
        int tilecol = t * tilesize + threadIdx.x;
        int tilerow = t * tilesize + threadIdx.y;
        Atile[threadIdx.y][threadIdx.x] = (row < m && tilecol < n) ? A[row * n + tilecol] : 0.0f;
        Btile[threadIdx.y][threadIdx.x] = (tilerow < n && col < k) ? B[tilerow * k + col] : 0.0f;

        __syncthreads();
        
        for (int p = 0; p < tilesize; p++) {
            value += Atile[threadIdx.y][p] * Btile[p][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < k) { //final matrix is m x k
        value *= muldiv; //does nothing during second matmul
        output[row * k + col] = value;
    }
}

__global__ void transpose(const float* K, float* QKT, int N, int d) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    __shared__ float tile[tilesize][tilesize];
    if (col < d && row < N) {
        tile[threadIdx.y][threadIdx.x] = K[row * d + col];
    }
    __syncthreads();
    
    int trow = blockIdx.x * tilesize + threadIdx.y;
    int tcol = blockIdx.y * tilesize + threadIdx.x;
    //recalculated transposed row and column index
    if (tcol < N && trow < d) {
        QKT[trow * N + tcol] = tile[threadIdx.x][threadIdx.y];
    }
}



// Q, K, V, output are device pointers
// naive implementation: K is transposed separately to reuse matmul kernel
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    dim3 threadsPerBlock (tilesize, tilesize);
    dim3 blocksPerGridQK ((N+tilesize-1)/tilesize,(M+tilesize-1)/tilesize); //Final matrix is MxN
    dim3 blocksPerGridV ((d+tilesize-1)/tilesize,(M+tilesize-1)/tilesize);
    dim3 blocksPerGridKT ((N+tilesize-1)/tilesize,(d+tilesize-1)/tilesize);
    float *KT, *QKT;
    cudaMalloc(&KT, sizeof(float) * N * d);
    cudaMalloc(&QKT, sizeof(float) * M * N); //QK^T is size MxN
    float muldiv = 1/sqrtf(d); //so that each element can be multiplied instead of division, which takes longer
    transpose<<<blocksPerGridKT, threadsPerBlock>>>(K, KT, N, d);
    matmul<<<blocksPerGridQK, threadsPerBlock>>>(Q, KT, QKT, M, d, N, muldiv);
    
    /*
    float* hQKT = (float*)malloc(M*N*sizeof(float));
    cudaMemcpy(hQKT, QKT, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nQK^T/sqrt(d):\n");
    for (int i = 0; i < M; i++) {
        printf("[ ");
        for (int j = 0; j < N; j++) {
            printf("%f ", hQKT[i * N + j]);
        }
        printf(" ]\n");
    }
    free(hQKT);
    */
    //softmaxing
    int softthreads = 1024;
    int softblocks = (softthreads + N - 1)/softthreads;
    int lb = -N, ub = 0;
    //Will optimize later by using a vertical array to do all the softmax operations at once.
    //printf("Entering the for loop");
    for (int i = 0; i < M; i++) {
        //printf("\nIteration %d", i);
        lb += N;
        ub += N;
        float* d_max;
        cudaMalloc(&d_max, sizeof(float));
        float* blockmaxes;
        cudaMalloc(&blockmaxes, softblocks*sizeof(float));
        maximum<<<softblocks, softthreads, softthreads*sizeof(float)>>>(QKT, blockmaxes, lb, ub);
        maximum<<<1, softblocks, softblocks*sizeof(float)>>>(blockmaxes, d_max, 0, softblocks);
        
        /*float* hmax = (float*)malloc(sizeof(float));
        cudaMemcpy(hmax, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        printf("\nMax %d: %f\n", i, *hmax);
        free(hmax);*/

        double* d_total;
        cudaMalloc(&d_total, sizeof(double));
        cudaMemset(d_total, 0, sizeof(double));

        reduce<<<softblocks, softthreads, softthreads*sizeof(double)>>>(QKT, d_total, d_max, lb, ub);
        
        
        softmax_kernel<<<softblocks, softthreads>>>(QKT, d_total, d_max, lb, ub);

        cudaFree(d_max);
        cudaFree(blockmaxes);
        cudaFree(d_total);   
    }
    //printf("\nFinal matmul");
    matmul<<<blocksPerGridV, threadsPerBlock>>>(QKT, V, output, M, N, d, 1);
    //printf("\nTweaking");
    cudaFree(KT);
    cudaFree(QKT);
}
