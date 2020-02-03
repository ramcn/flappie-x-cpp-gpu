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

//#define GEMV

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
 spmv_csr_vector_kernel_with_activation ( const int num_rows , const int cols , const float * sW , const float *W, float * x , float * y, float *xnext, float *b, const int index1, float *d_g_y, int index2)
 {
     __shared__ float vals1 [512];
     __shared__ float vals2 [512];
     __shared__ float vals3 [512];
     int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
     int warp_id = thread_id / 32 ; // global warp index
     int lane = thread_id & (32-1) ; // thread index within the warp

     int local_warp_id = threadIdx.x / 32;
     int local_lane_id =  threadIdx.x % 32;
     int pos = threadIdx.x;
     //int pos = local_warp_id * 32 + local_lane_id;

     float c1 = 0; float c2 = 0; float c3 = 0; float cinlocal = 0;
   
     y = d_g_y + index1 * 768;
     xnext = d_g_y + index2 * 768;
     // one warp per row
     int row = warp_id ;

     if( row < num_rows )
     {
         cinlocal = y[row+512];
         y[row+512] = 0;
         vals1 [ pos ] = 0;
         vals2 [ pos ] = 0;
         vals3 [ pos ] = 0;

         for ( int jj = 0 + lane ; jj < cols ; jj += 31) {
              vals1 [ pos ] += sW[ (row*cols)+jj ] * x [jj];
              vals2 [ pos ] += sW[ ((row+256)*cols)+jj ] * x [jj];
              vals3 [ pos ] += sW[ ((row+512)*cols)+jj ] * x [jj];
	 }
                // parallel reduction in shared memory
                if ( lane < 16) { vals1 [ pos  ] += vals1 [ pos + 16]; vals2 [ pos  ] += vals2 [ pos + 16]; vals3 [ pos  ] += vals3 [ pos + 16];}
                if ( lane < 8) { vals1 [ pos ] += vals1 [ pos + 8]; vals2 [ pos ] += vals2 [ pos + 8]; vals3 [ pos ] += vals3 [ pos + 8];}
                if ( lane < 4) { vals1 [ pos  ] += vals1 [ pos + 4]; vals2 [ pos  ] += vals2 [ pos + 4]; vals3 [ pos  ] += vals3 [ pos + 4]; }
                if ( lane < 2) { vals1 [ pos ] += vals1 [ pos + 2]; vals2 [ pos ] += vals2 [ pos + 2]; vals3 [ pos ] += vals3 [ pos + 2]; }
                if ( lane < 1) { vals1 [ pos ] += vals1 [ pos + 1]; vals2 [ pos ] += vals2 [ pos + 1];vals3 [ pos ] += vals3 [ pos + 1]; }
                // first thread OF EACH WARP ACCUMULATES the result
                if ( lane == 0) {
                  y[row] += vals1 [ pos ];
                  y[row+256] += vals2 [ pos ];
                  y[row+512] += vals3 [ pos ];
      		  y[row] = gpu_logisticf(y[row]);
      		  y[row+256] = gpu_logisticf(y[row+256]);
      		  y[row+512] = gpu_tanhf(y[row+256] * y[row+512] + cinlocal);
      		  y[row+512] = (-1) * y[row] * y[row+512] + y[row+512];
      		  y[row] = y[row] * x[row] + y[row+512];
                }
     }

      __syncthreads();

      if( row < num_rows )
      {
      	 vals1 [  pos ] = b[row];
      	 vals2 [ pos ] = b[row+256];
      	 vals3 [ pos ] = b[row+512];
         for ( int jj = 0 + lane ; jj < cols ; jj += 31) {
              vals1 [ pos ] += W[ (row*cols)+jj ] * y [jj];
              vals2 [ pos ] += W[ ((row+256)*cols)+jj ] * y [jj];
              vals3 [ pos ] += W[ ((row+512)*cols)+jj ] * y [jj];
         }
                // parallel reduction in shared memory
                if ( lane < 16) { vals1 [ pos  ] += vals1 [ pos + 16]; vals2 [ pos  ] += vals2 [ pos + 16]; vals3 [ pos  ] += vals3 [ pos + 16];}
                if ( lane < 8) { vals1 [ pos ] += vals1 [ pos + 8]; vals2 [ pos ] += vals2 [ pos + 8]; vals3 [ pos ] += vals3 [ pos + 8];}
                if ( lane < 4) { vals1 [ pos  ] += vals1 [ pos + 4]; vals2 [ pos  ] += vals2 [ pos + 4]; vals3 [ pos  ] += vals3 [ pos + 4]; }
                if ( lane < 2) { vals1 [ pos ] += vals1 [ pos + 2]; vals2 [ pos ] += vals2 [ pos + 2]; vals3 [ pos ] += vals3 [ pos + 2]; }
                if ( lane < 1) { vals1 [ pos ] += vals1 [ pos + 1]; vals2 [ pos ] += vals2 [ pos + 1];vals3 [ pos ] += vals3 [ pos + 1]; }
                // first thread OF EACH WARP ACCUMULATES the result
                if ( lane == 0) {
         	  x[row] = y[row]; // next invocation istate is from current ostate
                  xnext[row] += vals1 [ pos ];
                  xnext[row+256] += vals2 [ pos ];
                  xnext[row+512] += vals3 [ pos ];
                }
      }

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

    float Cin[768], Cout[768];
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
    fprintf(stderr,"Allocating feature vector of bytes xnr=%d xnc=%d %d\n on device",X->nr,X->nc,768*N*sizeof(float)); 
    fprintf(stderr,"Allocating ostate vector of bytes onr=%d onc=%d %d\n on device",ostate->nr,ostate->nc,ostate->nr*ostate->nc*sizeof(float)); 
    if(layer == 1) {
      cudaStat = cudaMalloc (( void **)& d_g_y , 768*N*sizeof(float)); // device // memory alloc for x 
      cudaMemcpy(d_g_y, X->data.f, 768*N*sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_a1, sW->data.f, 768*256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, W->data.f, 768*256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data.f, 768*sizeof(float), cudaMemcpyHostToDevice);
    float al =1.0f;
    float bet =1.0f;
#else
    for (size_t c = 0; c < Xnext->nc; c++) {
        memcpy(Xnext->data.v + c * Xnext->nrq, b->data.v, Xnext->nrq * sizeof(__m128));
    }
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
                //int num_blocks = 768/rows_per_block; // 48 or 32
                int num_blocks = 768/rows_per_block; // 48 or 32
                if(i == 1) 
                  cudaMemcpy(d_x, istate_ptr, N*sizeof(float), cudaMemcpyHostToDevice);
                //cudaMemcpy(d_y, xCol.data.f, M*sizeof(float), cudaMemcpyHostToDevice);
                spmv_csr_scalar_kernel_with_activation<<<1, 256>>>(M/3, N, d_a1, d_a2, d_x, d_y, d_xnext, d_b, index, d_g_y, index2);
                //spmv_csr_vector_kernel_with_activation<<<num_blocks, threads_per_block>>>(M, N, d_a1, d_a2, d_x, d_y, d_xnext, d_b, index, d_g_y, index2);
                //cudaMemcpy(XnextBuf.data.f, d_xnext, M*sizeof(float), cudaMemcpyDeviceToHost);
#else
                memcpy(Cin, xCol.data.f, 768*sizeof(float));
                memcpy(Cout, xColTmp->data.f, 768*sizeof(float));
                memcpy(Cout, Cin, 768 * sizeof(float) );
                memset(Cout + size + size, 0, size *sizeof(float));

                cblas_sgemv(CblasRowMajor, CblasNoTrans, 768, 256, 1.0, sW->data.f, 256, istate_ptr, 1, 1.0, Cout, 1);
                for (size_t i = 0; i < size; i++) {
                        Cout[i] = LOGISTICF(Cout[i]);
                        Cout[size+i] = LOGISTICF(Cout[size+i]);
                        Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] + Cin[i+size+size]);
                        ostate_ptr[i] = (-1) * Cout[i] * Cout[i+size+size] + Cout[i+size+size];
                        ostate_ptr[i] = Cout[i] * istate_ptr[i] + ostate_ptr[i];
		}
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
#else
    //cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, X->nc, W->nr, 1.0, W->data.f, W->stride, ostate->data.f, ostate->stride, 1.0, Xnext->data.f, Xnext->stride);
#endif
    return Xnext;
}

