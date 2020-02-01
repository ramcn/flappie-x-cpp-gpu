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


 __global__ void
 spmv_csr_scalar_kernel_with_activation ( const int num_rows , const int cols , const float * sW , const float *W, float * x , float * y, float *xnext, float *b, const int index1, float *d_g_y, int index2)
 {
     int row = blockDim.x * blockIdx.x + threadIdx.x ;
     float c1 = 0; 
     float c2 = 0; 
     float c3 = 0; 
     float cinlocal = 0;
   
     y = d_g_y + index1 * 768;
     xnext = d_g_y + index2 * 768;

     if( row < num_rows )
     {
         cinlocal = y[row+512];
         y[row+512] = 0;

         for (int jj = 0 ; jj < cols ; jj ++) {
             c1 += sW [ (row*cols)+jj ] * x[ jj ];
             c2 += sW [ ((row+256)*cols)+jj ] * x[ jj ];
             c3 += sW [ ((row+512)*cols)+jj ] * x[ jj ];
         }
         y[row] += c1 ;
         y[row+256] += c2 ;
         y[row+512] += c3 ;
     }

      y[row] = gpu_logisticf(y[row]);
      y[row+256] = gpu_logisticf(y[row+256]);
      y[row+512] = gpu_tanhf(y[row+256] * y[row+512] + cinlocal);
      y[row+512] = (-1) * y[row] * y[row+512] + y[row+512];
      y[row] = y[row] * x[row] + y[row+512];

      __syncthreads();

      c1 = b[row]; c2 = b[row+256]; c3=b[row+512];

      if( row < num_rows )
      {
         for (int jj = 0 ; jj < cols ; jj ++) {
             c1 += W [ (row*cols)+jj ] * y[ jj ];
             c2 += W [ ((row+256)*cols)+jj ] * y[ jj ];
             c3 += W [ ((row+512)*cols)+jj ] * y[ jj ];
         }
	  
         x[row] = y[row]; // next invocation istate is from current ostate
         xnext[row] = c1; xnext[row+256] = c2 ; xnext[row+512] = c3;
      }

 }


float *d_g_y;

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

    float Cin[768], Cout[768], A[256*768];
    float *ostate_ptr;
    float *istate_ptr;

#ifdef GEMV
    float *d_a1, *d_a2, *d_x, *d_y, *d_cin, *d_xnext, *d_b ;
    cudaStat = cudaMalloc (( void **)& d_a1 , 768*256*sizeof(float)); // device // memory alloc for a
    cudaStat = cudaMalloc (( void **)& d_a2 , 768*256*sizeof(float)); // device // memory alloc for a
    cudaStat = cudaMalloc (( void **)& d_x , 256*sizeof(float)); // device // memory alloc for x
    cudaStat = cudaMalloc (( void **)& d_y , 768*sizeof(float)); // device // memory alloc for y
    cudaStat = cudaMalloc (( void **)& d_xnext , 768*sizeof(float)); // device // memory alloc for xnext 
    cudaStat = cudaMalloc (( void **)& d_b , 768*sizeof(float)); // device // memory alloc for bias 
    if(layer == 1) {
      cudaStat = cudaMalloc (( void **)& d_g_y , 768*N*sizeof(float)); // device // memory alloc for x 
      cudaMemcpy(d_g_y, X->data.f, 768*N*sizeof(float), cudaMemcpyHostToDevice);
    }
    memcpy(A, sW->data.f, 256*768*sizeof(float));
    cudaMemcpy(d_a1, sW->data.f, 768*256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, W->data.f, 768*256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data.f, 768*sizeof(float), cudaMemcpyHostToDevice);
    float al =1.0f;
    float bet =1.0f;
#endif

    for (int i = 1; i < N; i++) {
        size_t index, index2;
        // LOAD
        {
                if(backward) {
                        index = N - i - 1;
                        xCol.data.f = X->data.f + index * X->nr;
                        ostate_ptr = ostate->data.f + index * ostate->nr;
                        istate_ptr = ostate_ptr + 256;
                        XnextBuf.data.f = Xnext->data.f + (index+1) * Xnext->nr;
			index2 = index + 1;
                }

                else {
                        index = i;
                        xCol.data.f = X->data.f + index * X->nr;
                        ostate_ptr = ostate->data.f + index * ostate->nr;
                        istate_ptr = ostate_ptr - 256;
                        XnextBuf.data.f = Xnext->data.f + (index-1) * Xnext->nr;
			index2 = index - 1; 
                }
        }

        // COMPUTE
        {
                const size_t size = 256;
    		int M=768, N=256;

#ifdef GEMV
		int threads_per_row = 32; // warp size
                int threads_per_block = 512 ; //threads per block 512 or 768
		int rows_per_block = threads_per_block/threads_per_row; // 16 or 24
                int num_blocks = 768/rows_per_block; // 48 or 32
                if(i == 1) 
                  cudaMemcpy(d_x, istate_ptr, N*sizeof(float), cudaMemcpyHostToDevice);
                //cudaMemcpy(d_y, xCol.data.f, M*sizeof(float), cudaMemcpyHostToDevice);
                spmv_csr_scalar_kernel_with_activation<<<1, 256>>>(M/3, N, d_a1, d_a2, d_x, d_y, d_xnext, d_b, index, d_g_y, index2);
                //cudaMemcpy(XnextBuf.data.f, d_xnext, M*sizeof(float), cudaMemcpyDeviceToHost);
#else
                cblas_sgemv(CblasRowMajor, CblasNoTrans, 768, 256, 1.0, A, 256, istate_ptr, 1, 1.0, Cout, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, W->nc, W->nr, 1.0, W->data.f, W->stride, ostate_ptr, 1, 1.0, XnextBuf.data.f, 1);
#endif

        }
    } // end of N iterations
    xColTmp = free_flappie_matrix(xColTmp);
    assert(validate_flappie_matrix (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));

#ifdef GEMV
    cudaFree (d_a1 );
    cudaFree (d_a2 );
    cudaFree (d_x );
    cudaFree (d_y );
    cudaFree (d_xnext );
    if(layer == 4) {
    	cudaMemcpy(Xnext->data.f, d_g_y, 768*N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree (d_g_y );
    }
#endif

    //cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, X->nc, W->nr, 1.0, W->data.f, W->stride, ostate->data.f, ostate->stride, 1.0, Xnext->data.f, Xnext->stride);
    return Xnext;
}

