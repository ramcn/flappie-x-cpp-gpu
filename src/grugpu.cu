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

//#define GPU

__global__ void kernel (void){
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
}

int grugpu( void ) {
  kernel<<<1,1>>>();
  printf( "Hello, World!\n" );
  return 0;
}

flappie_matrix aes_grumod_linear_gpu( const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, int backward, const_flappie_matrix W, const_flappie_matrix b) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

#ifdef GPU
    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
    cublasHandle_t handle ; // CUBLAS context
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

    //float *Cin, *Cout, *A, *Bnext;
    float Cin[768], Cout[768], A[256*768];
    float *Bnext;

#ifdef GPU
    float *d_a, *d_x, *d_y;
    cudaStat = cudaMalloc (( void **)& d_a , 768*256*sizeof(float)); // device // memory alloc for a
    cudaStat = cudaMalloc (( void **)& d_x , 256*sizeof(float)); // device // memory alloc for x
    cudaStat = cudaMalloc (( void **)& d_y , 768*sizeof(float)); // device // memory alloc for y
    float al =1.0f;
    float bet =1.0f;
    stat = cublasCreate (&handle);
    grugpu();
#endif

    for (int i = 1; i < N; i++) {
      #pragma HLS pipeline
        size_t index;
        // LOAD
        {
                if(backward) {
                        index = N - i - 1;
                        xCol.data.f = X->data.f + index * X->nr;
                        sCol1.data.f = ostate->data.f + (index + 1) * ostate->nr;
                        sCol2.data.f = ostate->data.f + index * ostate->nr;
        }
                else {
                        index = i;
                        xCol.data.f = X->data.f + index * X->nr;
                        sCol1.data.f = ostate->data.f + (index - 1) * ostate->nr;
                        sCol2.data.f = ostate->data.f + index * ostate->nr;
                }
                memcpy(Cin, xCol.data.f, 768*sizeof(float));
                memcpy(Cout, xColTmp->data.f, 768*sizeof(float));
                memcpy(A, sW->data.f, 256*768*sizeof(float));
                //memcpy(Bnext, sCol2.data.f, 256 * sizeof(float));
                Bnext = sCol2.data.f;

        }

        // COMPUTE
        {
                //flappie_matrix Cin = &xCol; flappie_matrix Cout = xColTmp;  flappie_matrix A = sW; flappie_matrix Bnext = &sCol2;
                float *B;
                int M=768, N=256;
                if(backward) B = Bnext + 256; //B is ostate
                else B = Bnext - 256;
                const size_t size = 256;
                memcpy(Cout, Cin, 768 * sizeof(float) );
                memset(Cout + size + size, 0, size *sizeof(float));

                if(backward) {
                        XnextBuf.data.f = Xnext->data.f + (index+1) * Xnext->nr;
                }else {
                        XnextBuf.data.f = Xnext->data.f + (index-1) * Xnext->nr;
                }
                cblas_sgemv(CblasColMajor, CblasTrans, W->nr, W->nc, 1.0, W->data.f, W->stride, B, 1, 1.0, XnextBuf.data.f, 1);
                //cblas_sgemv(CblasColMajor, CblasTrans, W->nr, W->nc, 1.0, W->data.f, W->stride, Bnext, 1, 1.0, sCol2.data.f, 1);

                cblas_sgemv(CblasColMajor, CblasTrans, 256, 768, 1.0, A, 256, B, 1, 1.0, Cout, 1);
#ifdef GPU
                stat = cublasSetMatrix (M,N, sizeof(float),A,M,d_a,M);
                stat = cublasSetVector (N,sizeof(float),B,1,d_x,1);
                stat = cublasSetVector (M,sizeof(float),Cout,1,d_y,1);
 		stat=cublasSgemv(handle,CUBLAS_OP_T,M,N,&al,d_a, M,d_x,1,&bet, d_y,1);
		stat = cublasGetVector (M, sizeof(float) ,d_y ,1 ,Cout ,1); 
#endif

                for (size_t i = 0; i < size; i++) {
                        Cout[i] = LOGISTICF(Cout[i]);
                        Cout[size+i] = LOGISTICF(Cout[size+i]);
                        Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] + Cin[i+size+size]);
                        Bnext[i] = (-1) * Cout[i] * Cout[i+size+size] + Cout[i+size+size];
                        Bnext[i] = Cout[i] * B[i] + Bnext[i];
                }

        }
        {
        }
    } // end of N iterations
    xColTmp = free_flappie_matrix(xColTmp);
    assert(validate_flappie_matrix (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));

#ifdef GPU
    cudaFree (d_a );
    cudaFree (d_x );
    cudaFree (d_y );
    cublasDestroy ( handle );
#endif
    return Xnext;
}

