#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <math.h>
#include "layers.h"
#include "flappie_stdlib.h"
#include "util.h"
#include <cublas_v2.h>


#include <cblas.h>

#define GEMV

#    define _A 12102203.161561485f
#    define _B 1065353216.0f
#    define _BOUND 88.02969193111305
__device__ static inline float gpu_expf(float x) {
    x = fmaxf(-_BOUND, fminf(_BOUND, x));
    union {
        uint32_t i;
        float f;
    } value = {
    .i = (uint32_t) (_A * x + _B)};
    return value.f;
}

__device__ static inline float gpu_logisticf(float x) {
    return 1.0 / (1.0 + gpu_expf(-x));
}

__device__ static inline float gpu_tanhf(float x) {
    const float y = gpu_logisticf(x + x);
    return y + y - 1.0;
}


#define THREADS_PER_ROW 32
#define ROWS_PER_BLOCK  16
#define THREADS_PER_BLOCK  (ROWS_PER_BLOCK*THREADS_PER_ROW)
#define NUM_BLOCKS  (768/ROWS_PER_BLOCK)

__global__ void spmv_csr_vector_kernel_v1 ( const int num_rows , int num_cols, const float * data , const float * x , float * y) 
{
    __shared__ float vals [ROWS_PER_BLOCK * 32];

    const int thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_ROW - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_ROW;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_ROW;               // vector index within the block
    const int num_vectors = ROWS_PER_BLOCK * gridDim.x;                   // total number of active vectors
    int pos = threadIdx.x;
    //int pos = vector_id * 32 + thread_lane;

    for(int row = vector_id; row < num_rows; row += num_vectors)
    {
    		float sum = 0;
                for ( int jj = 0 + thread_lane ; jj < num_cols ; jj += THREADS_PER_ROW-1)
                  sum += data [ (row*num_cols)+jj ] * x [jj];
                // parallel reduction in shared memory
  		vals[pos] = sum;
        	if (thread_lane < 16) vals[pos] += vals[pos + 16];
        	if (thread_lane <  8) vals[pos] += vals[pos +  8];
        	if (thread_lane <  4) vals[pos] += vals[pos +  4];
        	if (thread_lane <  2) vals[pos] += vals[pos +  2];
        	if (thread_lane <  1) vals[pos] += vals[pos +  1];
                // first thread OF EACH WARP ACCUMULATES the result
                if ( thread_lane == 0)
                  y[row] += vals [ pos];
        }
}

__global__ void spmv_csr_vector_kernel_v2 ( const int num_rows , int num_cols, const float * data , const float * x , float * y) 
{
    __shared__ float vals [ROWS_PER_BLOCK * 32];

    const int thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_ROW - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_ROW;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_ROW;               // vector index within the block
    const int num_vectors = ROWS_PER_BLOCK * gridDim.x;                   // total number of active vectors
    int pos = threadIdx.x;

    for(int row = vector_id; row < num_rows; row += num_vectors)
    {
    		float sum = 0;
                for ( int jj = 0 + thread_lane ; jj < num_cols ; jj += THREADS_PER_ROW)
                  sum += data [ (row*num_cols)+jj ] * x [jj];
                // parallel reduction in shared memory
  		vals[pos] = sum;
        	if (THREADS_PER_ROW > 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
        	if (THREADS_PER_ROW >  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
        	if (THREADS_PER_ROW >  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
        	if (THREADS_PER_ROW >  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
        	if (THREADS_PER_ROW >  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
                // first thread OF EACH WARP ACCUMULATES the result
                if ( thread_lane == 0)
                  y[row] += vals [ pos];
        }
}

__global__ void
 spmv_csr_scalar_kernel ( const int num_rows , const int cols , const float * data , const float * x , float * y)
 {
     int row = blockDim.x * blockIdx.x + threadIdx.x ;
     float dot = 0;
     if( row < num_rows )
     {
         for (int jj = 0 ; jj < cols ; jj ++)
             dot += data [ (row*cols)+jj ] * x[ jj ];
         y[ row ] += dot ;
     }
 }

flappie_matrix aes_grumod_linear_gpu( const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, int backward, const_flappie_matrix W, const_flappie_matrix b, int layer) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

#ifdef GEMV
    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
#endif

    const size_t size = sW->nr;
    const size_t N = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    ostate = remake_flappie_matrix(ostate, size, N);
    flappie_matrix xColTmp = make_flappie_matrix(3 * size, 1);

    _Mat xCol, sCol1, sCol2, XnextBuf;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    if(backward) {
      xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
      sCol1.data.v = ostate->data.v;
      sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
      grumod_step(&xCol, &sCol1, sW, xColTmp, &sCol2);
    }
    else {
      sCol1.data.v = ostate->data.v + ostate->nrq;
      sCol2.data.v = ostate->data.v;
      grumod_step(&xCol, &sCol1, sW, xColTmp, &sCol2);
    }

    flappie_matrix Xnext = remake_flappie_matrix(NULL, W->nc, ostate->nc);
    RETURN_NULL_IF(NULL == Xnext, NULL);
    for (size_t c = 0; c < Xnext->nc; c++) {
        memcpy(Xnext->data.v + c * Xnext->nrq, b->data.v, Xnext->nrq * sizeof(__m128));
    }

    float Cin[768], Cout[768], A[256*768];
    float *ostate_ptr;
    float *istate_ptr;

    memcpy(A, sW->data.f, 256*768*sizeof(float));

#ifdef GEMV
    float *d_a, *d_x, *d_y, *d_cin ;
    cudaStat = cudaMalloc (( void **)& d_a , 768*256*sizeof(float)); // device // memory alloc for a
    cudaStat = cudaMalloc (( void **)& d_x , 256*sizeof(float)); // device // memory alloc for x
    cudaStat = cudaMalloc (( void **)& d_y , 768*sizeof(float)); // device // memory alloc for y
    cudaStat = cudaMalloc (( void **)& d_cin , 768*sizeof(float)); // device // memory alloc for cin 
    cudaMemcpy(d_a, A, 768*256*sizeof(float), cudaMemcpyHostToDevice);
    float al =1.0f;
    float bet =1.0f;
#endif

    for (int i = 1; i < N; i++) {
        size_t index;
        // LOAD
        {
                if(backward) {
                        index = N - i - 1;
                        xCol.data.f = X->data.f + index * X->nr;
                        ostate_ptr = ostate->data.f + index * ostate->nr;
                        istate_ptr = ostate_ptr + 256;
        	}
                else {
                        index = i;
                        xCol.data.f = X->data.f + index * X->nr;
                        ostate_ptr = ostate->data.f + index * ostate->nr;
                        istate_ptr = ostate_ptr - 256;
                }

                memcpy(Cin, xCol.data.f, 768*sizeof(float));
                memcpy(Cout, xColTmp->data.f, 768*sizeof(float));
        }

        // COMPUTE
        {
                const size_t size = 256;
    		int M=768, N=256;
                memcpy(Cout, Cin, 768 * sizeof(float) );
                memset(Cout + size + size, 0, size *sizeof(float));

#ifdef GEMV
                cudaMemcpy(d_x, istate_ptr, N*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_y, Cout, M*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_cin, Cin, M*sizeof(float), cudaMemcpyHostToDevice);
                spmv_csr_vector_kernel_v1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(M, N, d_a, d_x, d_y);
                //spmv_csr_vector_kernel_v2<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(M, N, d_a, d_x, d_y);
                //spmv_csr_scalar_kernel<<<1, 768>>>(M, N, d_a, d_x, d_y);

                cudaMemcpy(Cout, d_y, M*sizeof(float), cudaMemcpyDeviceToHost);
#else
                cblas_sgemv(CblasRowMajor, CblasNoTrans, 768, 256, 1.0, A, 256, istate_ptr, 1, 1.0, Cout, 1);
#endif

                for (size_t i = 0; i < size; i++) {
                        Cout[i] = LOGISTICF(Cout[i]);
                        Cout[size+i] = LOGISTICF(Cout[size+i]);
                        Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] + Cin[i+size+size]);
                        ostate_ptr[i] = (-1) * Cout[i] * Cout[i+size+size] + Cout[i+size+size];
                        ostate_ptr[i] = Cout[i] * istate_ptr[i] + ostate_ptr[i];
                        //ostate_ptr[i] = Cout[i];
                }

        }
        {
        }
    } // end of N iterations
    xColTmp = free_flappie_matrix(xColTmp);
    assert(validate_flappie_matrix (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));

#ifdef GEMV
    cudaFree (d_a );
    cudaFree (d_x );
    cudaFree (d_y );
#endif

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, X->nc, W->nr, 1.0, W->data.f, W->stride, ostate->data.f, ostate->stride, 1.0, Xnext->data.f, Xnext->stride);
    return Xnext;
}

