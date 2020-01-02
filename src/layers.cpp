/*  This Source Code Form is subject to the terms of the Oxford Nanoiore
 *  Technologies, Ltd. Public License, v. 1.0. If a copy of the License 
 *  was not  distributed with this file, You can obtain one at
 *  http://nanoporetech.com
 */

#ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#else
#    include <cblas.h>
#endif
#include <math.h>
#include "layers.h"
#include "flappie_stdlib.h"
#include "util.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

//#include "hw/ekf_hw.h"
//#include "ref_model/ekf_sw.h"
//#include "hw/vio_regs.h"
//#include "hw/vio_utils.h"
//#include "stimuli_gen/matrix_generation.h"

#define STREAMING_ENABLED 1

/**  Apply tanh to a matrix element-wise
 *  @param C Matrix
 *
 **/
void tanh_activation_inplace(flappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = TANHFV(C->data.v[offset + r]);
        }
    }
    (void)validate_flappie_matrix(C, -1.0, 1.0, 0.0, true, __FILE__, __LINE__);
}


/**  Apply exp to a matrix element-wise
 *  @param C Matrix
 *
 **/
void exp_activation_inplace(flappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = EXPFV(C->data.v[offset + r]);
        }
    }
    (void)validate_flappie_matrix(C, 0.0, INFINITY, 1.0, true, __FILE__,
                                   __LINE__);
}


/**  Apply log to a matrix element-wise
 *  @param C Matrix
 *
 **/
void log_activation_inplace(flappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = LOGFV(C->data.v[offset + r]);
        }
    }
}


/**  Apply ELU activation function to a matrix element-wise
 *  @param C Matrix
 *
 **/
void elu_activation_inplace(flappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = ELUFV(C->data.v[offset + r]);
        }
    }
}


/** Apply robost log activation
 *
 *  Applies log(min_prob / nrow + (1 - min_prob) * x) elementwise to matrix
 *  where x in element and nrow is the number of rows
 *
 *  @param C Matrix
 *  @param min_prob  Minimum probability
 *
 **/
void robustlog_activation_inplace(flappie_matrix C, float min_prob) {
    assert(min_prob >= 0.0);
    assert(min_prob <= 1.0);
    RETURN_NULL_IF(NULL == C, );

    const size_t nblock = C->nc;
    const __m128 mpv = _mm_set1_ps(min_prob);
    const __m128 mpvm1 = _mm_set1_ps(1.0f - min_prob);
    for (size_t i = 0; i < nblock; i++) {
        const size_t offset = i * C->nrq;
        for (size_t r = 0; r < C->nrq; r++) {
            C->data.v[offset + r] =
                LOGFV(mpv + mpvm1 * C->data.v[offset + r]);
        }
    }
}


flappie_matrix embedding(int const * index, size_t n, const_flappie_matrix E, flappie_matrix C){
    RETURN_NULL_IF(NULL == index, NULL);

    const size_t nr = E->nr;
    const size_t nrq = E->nrq;
    const size_t nc = n;
    C = remake_flappie_matrix(C, nr, nc);
    RETURN_NULL_IF(NULL == C, NULL);

    for(size_t c=0 ; c < nc ; c++){
        assert(index[c] >= 0 && index[c] < E->nc);
        const size_t offsetC = c * nrq;
        const size_t offsetE = index[c] * nrq;
        for(size_t r=0 ; r < nrq ; r++){
            C->data.v[offsetC + r] = E->data.v[offsetE + r];
        }
    }

    return C;
}


flappie_matrix window(const_flappie_matrix input, size_t w, size_t stride) {
    RETURN_NULL_IF(NULL == input, NULL);
    assert(w > 0);
    const size_t wh = (w + 1) / 2;

    flappie_matrix output = make_flappie_matrix(input->nr * w,
                                                  (size_t)ceilf(input->nc /
                                                             (float)stride));
    RETURN_NULL_IF(NULL == output, NULL);

    for (size_t col = 0; col < output->nc; col++) {
        // First and last columns are special cases
        const size_t out_offset = col * output->stride;
        const int icol = (int)(col * stride);
        for (int i = 0, w1 = (icol - wh + 1); w1 <= icol + wh; w1++) {
            if (w1 < 0 || w1 >= input->nc) {
                i += input->nr;
                continue;
            }
            const size_t in_offset = w1 * input->stride;
            for (size_t row = 0; row < input->nr; row++, i++) {
                output->data.f[out_offset + i] = input->data.f[in_offset + row];
            }
        }
    }

    return output;
}


/**  Convolution of the input data
 *  @param X Input data matrix (features x nobs)
 *  @param W Filter matrix (winlen * features x nfilter)
 *
 *  The input is padded with zeros such that the resultant matrix has the
 *  same size as the input (under a stride of 1).
 *
 *  Note: The rows of the input matrix X are padded with zeros to make them
 *  a multiple of the SSE vector size (4).  The filter matrix must have been
 *  expanded accordingly.
 **/
flappie_matrix convolution_linear(const_flappie_matrix X, const_flappie_matrix W,
                            const_flappie_matrix b, size_t stride,
                            flappie_matrix C, const_flappie_matrix iW, const_flappie_matrix bG) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != W);
    assert(NULL != b);
    assert(W->nc == b->nr);
    assert(stride > 0);
    // Window length of filter
    assert((W->nrq % X->nrq) == 0);
    const size_t winlen = W->nrq / X->nrq;
    const size_t nfilter = W->nc;
    // Padding -- right-hand side is longer when asymmetric padding is required
    const size_t padL = (winlen - 1) / 2;
    const size_t padR = winlen / 2;
    const size_t ncolC = iceil(X->nc, stride);
    C = remake_flappie_matrix(C, nfilter, ncolC);
    RETURN_NULL_IF(NULL == C, NULL);

    // Matrix strides
    const size_t ldC = C->stride;
    const size_t ldW = W->stride;
    // FIX padding disable changes
    //const size_t ldX = X->stride; 
    const size_t ldX = 1;
    const size_t ldFeature = ldX;


    /*#define VIO_EKF_CHOL_IN_BUFSIZE 128
    float *res_cov    = (float *)vio_malloc(VIO_EKF_CHOL_IN_BUFSIZE);
    float *result_cov    = (float *)vio_malloc(VIO_EKF_CHOL_IN_BUFSIZE);
    float *lc_cov     = (float *)vio_malloc(VIO_EKF_MMUL_A_BUFSIZE);
    float *gain_cov   = (float *)vio_malloc(VIO_EKF_MMUL_B_BUFSIZE);
    float *cov_in    = (float *)vio_malloc(VIO_EKF_MMUL_C_BUFSIZE);
    uint32_t gain_rows=4, gain_cols=4, gain_str=1, cov_in_str=1 ;*/
    //ekf_sw sw;
    //ekf_hw hw;
    //sw.mat_mul_c(gain_cov,gain_cov,cov_in,gain_cols,gain_rows, gain_cols - 1, false, true, -1.0f, 1.0f, gain_str, gain_str, cov_in_str, cov_in, 0, 0);


    //fprintf(stderr, "convolution xnr=%lu xnc=%lu wnr=%lu wnc=%lubnr=%lu bnc=%lu cnr=%lu cnc=%lu\n",X->nr, X->nc, W->nr, W->nc, b->nr, b->nc, C->nr, C->nc);
    //fprintf(stderr, "wnrq=%lu xnrq=%lu windlen=%lu nfliter=%lu padr=%lu padl=%lu \n",W->nrq, X->nrq, winlen, nfilter, padR, padL);

    // Copy bias into result matrix
    for (size_t i = 0; i < C->nc; i++) {
        memcpy(C->data.v + i * C->nrq, b->data.v, C->nrq * sizeof(__m128));
    }

    // Left-hand side edge case where only part of the filter covers the input
    for (size_t w = 0; w < padL; w += stride) {
        const size_t offsetW = ldFeature * (padL - w);
        const size_t ncol = w / stride;
	    //printf("1: Number of M=%lu N=%lu",W->nr - offsetW, W->nc);
        cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc,
                    1.0, W->data.f + offsetW, ldW,
                    X->data.f, 1, 1.0, C->data.f + ldC * ncol, 1);
    }

    // Number of columns of X already filled * ldC
    const size_t ncolsL_complete = iceil(padL, stride);
    const size_t offsetC_L = ldC * ncolsL_complete;
    // Because of stride, first valid filter may not start at beginning of X
    //const int shiftX_L = stride - (padL % stride);
    const size_t shiftX_L = ncolsL_complete * stride - padL;
    const size_t offsetX_L = shiftX_L * ldX;
    // Find multiple of stride greater or equal to winlen
    const size_t nstepC = iceil(winlen, stride);
    const size_t nstepX = stride * nstepC;

    for (size_t w = 0; w < winlen; w += stride) {
        //  Multiply reshaped X matrix by filter matrix
        //  The rows of X are padded by zeros to make a multiple of 4.
        //  Input matrix 'X'
        //   - stride is ldX * nstepX
        //   - offset by ldX * w (w cols)
        //   - Ncolumns is (X->nc - w) / nstepX + adjustment if a final window fits
        //  Filter matrix needs to be padded appropriately for the padding of X.
        //
        const size_t ncol_processed = ifloor(X->nc - shiftX_L - w, nstepX);
        const size_t initial_col = ifloor(w, stride);

        //fprintf(stderr, "2: Number of M=%lu N=%lu K=%lu\n",W->nc,ncol_processed,W->nr);
        //fprintf(stderr, "3.1: Offset in X is %lu\n", ldX * w + offsetX_L);
        //fprintf(stderr, "3.2: Offset in C is %lu\n", ldC * initial_col + offsetC_L);
        //fprintf(stderr, "4: Lda=%lu Ldb=%lu Ldc=%lu\n", ldW, ldX * nstepX, ldC * nstepC );
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc,
                    ncol_processed, W->nr, 1.0, W->data.f, ldW,
                    X->data.f + ldX * w + offsetX_L, ldX * nstepX, 1.0,
                    C->data.f + ldC * initial_col + offsetC_L, ldC * nstepC);
    }

    // Right-hand side edge case where only part of the filter covers the input
    const size_t maxCol_reshape = ifloor(X->nc - shiftX_L, nstepX);
    const size_t remainder_reshape = (X->nc - shiftX_L) % nstepX;
    const size_t offsetC_R =
        offsetC_L + ldC * nstepC * (maxCol_reshape - 1) +
        ldC * (remainder_reshape / stride) + ldC;
    const size_t offsetX_R = (X->nc - winlen + 1) * ldX;
    // How far into padding is first block
    const int startR = stride - (padL + X->nc - winlen) % stride - 1;
    for (size_t w = startR; w < padR; w += stride) {
        const size_t offsetW = ldFeature * (w + 1);
	//printf("3: Number of M=%d N=%d\n",W->nr - offsetW, W->nc);
        cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc, 1.0,
                    W->data.f, ldW,
                    X->data.f + offsetX_R + ldX * w, 1, 1.0,
                    C->data.f + offsetC_R + ldC * (w / stride), 1);
    }

    assert(validate_flappie_matrix
           (C, NAN, NAN, 0.0, true, __FILE__, __LINE__));

    tanh_activation_inplace(C);

    flappie_matrix gruB1in = affine_map(C, iW, bG, NULL);

    return gruB1in;
}

flappie_matrix convolution(const_flappie_matrix X, const_flappie_matrix W,
                            const_flappie_matrix b, size_t stride,
                            flappie_matrix C) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != W);
    assert(NULL != b);
    assert(W->nc == b->nr);
    assert(stride > 0);
    // Window length of filter
    assert((W->nrq % X->nrq) == 0);
    const size_t winlen = W->nrq / X->nrq;
    const size_t nfilter = W->nc;
    // Padding -- right-hand side is longer when asymmetric padding is required
    const size_t padL = (winlen - 1) / 2;
    const size_t padR = winlen / 2;
    const size_t ncolC = iceil(X->nc, stride);
    C = remake_flappie_matrix(C, nfilter, ncolC);
    RETURN_NULL_IF(NULL == C, NULL);

    // Matrix strides
    const size_t ldC = C->stride;
    const size_t ldW = W->stride;
    // FIX padding disable changes
    //const size_t ldX = X->stride; 
    const size_t ldX = 1;
    const size_t ldFeature = ldX;


    /*#define VIO_EKF_CHOL_IN_BUFSIZE 128
    float *res_cov    = (float *)vio_malloc(VIO_EKF_CHOL_IN_BUFSIZE);
    float *result_cov    = (float *)vio_malloc(VIO_EKF_CHOL_IN_BUFSIZE);
    float *lc_cov     = (float *)vio_malloc(VIO_EKF_MMUL_A_BUFSIZE);
    float *gain_cov   = (float *)vio_malloc(VIO_EKF_MMUL_B_BUFSIZE);
    float *cov_in    = (float *)vio_malloc(VIO_EKF_MMUL_C_BUFSIZE);
    uint32_t gain_rows=4, gain_cols=4, gain_str=1, cov_in_str=1 ;*/
    //ekf_sw sw;
    //ekf_hw hw;
    //sw.mat_mul_c(gain_cov,gain_cov,cov_in,gain_cols,gain_rows, gain_cols - 1, false, true, -1.0f, 1.0f, gain_str, gain_str, cov_in_str, cov_in, 0, 0);


    //fprintf(stderr, "convolution xnr=%lu xnc=%lu wnr=%lu wnc=%lubnr=%lu bnc=%lu cnr=%lu cnc=%lu\n",X->nr, X->nc, W->nr, W->nc, b->nr, b->nc, C->nr, C->nc);
    //fprintf(stderr, "wnrq=%lu xnrq=%lu windlen=%lu nfliter=%lu padr=%lu padl=%lu \n",W->nrq, X->nrq, winlen, nfilter, padR, padL);

    // Copy bias into result matrix
    for (size_t i = 0; i < C->nc; i++) {
        memcpy(C->data.v + i * C->nrq, b->data.v, C->nrq * sizeof(__m128));
    }

    // Left-hand side edge case where only part of the filter covers the input
    for (size_t w = 0; w < padL; w += stride) {
        const size_t offsetW = ldFeature * (padL - w);
        const size_t ncol = w / stride;
	    //printf("1: Number of M=%lu N=%lu",W->nr - offsetW, W->nc);
        cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc,
                    1.0, W->data.f + offsetW, ldW,
                    X->data.f, 1, 1.0, C->data.f + ldC * ncol, 1);
    }

    // Number of columns of X already filled * ldC
    const size_t ncolsL_complete = iceil(padL, stride);
    const size_t offsetC_L = ldC * ncolsL_complete;
    // Because of stride, first valid filter may not start at beginning of X
    //const int shiftX_L = stride - (padL % stride);
    const size_t shiftX_L = ncolsL_complete * stride - padL;
    const size_t offsetX_L = shiftX_L * ldX;
    // Find multiple of stride greater or equal to winlen
    const size_t nstepC = iceil(winlen, stride);
    const size_t nstepX = stride * nstepC;

    for (size_t w = 0; w < winlen; w += stride) {
        //  Multiply reshaped X matrix by filter matrix
        //  The rows of X are padded by zeros to make a multiple of 4.
        //  Input matrix 'X'
        //   - stride is ldX * nstepX
        //   - offset by ldX * w (w cols)
        //   - Ncolumns is (X->nc - w) / nstepX + adjustment if a final window fits
        //  Filter matrix needs to be padded appropriately for the padding of X.
        //
        const size_t ncol_processed = ifloor(X->nc - shiftX_L - w, nstepX);
        const size_t initial_col = ifloor(w, stride);

        //fprintf(stderr, "2: Number of M=%lu N=%lu K=%lu\n",W->nc,ncol_processed,W->nr);
        //fprintf(stderr, "3.1: Offset in X is %lu\n", ldX * w + offsetX_L);
        //fprintf(stderr, "3.2: Offset in C is %lu\n", ldC * initial_col + offsetC_L);
        //fprintf(stderr, "4: Lda=%lu Ldb=%lu Ldc=%lu\n", ldW, ldX * nstepX, ldC * nstepC );
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc,
                    ncol_processed, W->nr, 1.0, W->data.f, ldW,
                    X->data.f + ldX * w + offsetX_L, ldX * nstepX, 1.0,
                    C->data.f + ldC * initial_col + offsetC_L, ldC * nstepC);
    }

    // Right-hand side edge case where only part of the filter covers the input
    const size_t maxCol_reshape = ifloor(X->nc - shiftX_L, nstepX);
    const size_t remainder_reshape = (X->nc - shiftX_L) % nstepX;
    const size_t offsetC_R =
        offsetC_L + ldC * nstepC * (maxCol_reshape - 1) +
        ldC * (remainder_reshape / stride) + ldC;
    const size_t offsetX_R = (X->nc - winlen + 1) * ldX;
    // How far into padding is first block
    const int startR = stride - (padL + X->nc - winlen) % stride - 1;
    for (size_t w = startR; w < padR; w += stride) {
        const size_t offsetW = ldFeature * (w + 1);
	//printf("3: Number of M=%d N=%d\n",W->nr - offsetW, W->nc);
        cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc, 1.0,
                    W->data.f, ldW,
                    X->data.f + offsetX_R + ldX * w, 1, 1.0,
                    C->data.f + offsetC_R + ldC * (w / stride), 1);
    }

    assert(validate_flappie_matrix
           (C, NAN, NAN, 0.0, true, __FILE__, __LINE__));

    return C;
}


flappie_matrix feedforward_linear(const_flappie_matrix X,
                                   const_flappie_matrix W,
                                   const_flappie_matrix b, flappie_matrix C) {
    return affine_map(X, W, b, C);
}


flappie_matrix feedforward_tanh(const_flappie_matrix X,
                                 const_flappie_matrix W,
                                 const_flappie_matrix b, flappie_matrix C) {
    C = affine_map(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    tanh_activation_inplace(C);

    assert(validate_flappie_matrix
           (C, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return C;
}


flappie_matrix feedforward_exp(const_flappie_matrix X,
                                const_flappie_matrix W,
                                const_flappie_matrix b, flappie_matrix C) {
    C = affine_map(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    exp_activation_inplace(C);

    assert(validate_flappie_matrix
           (C, 0.0, NAN, 1.0, true, __FILE__, __LINE__));
    return C;
}


flappie_matrix residual(const_flappie_matrix X, const_flappie_matrix fX, flappie_matrix C) {
    RETURN_NULL_IF(NULL == X, NULL);
    RETURN_NULL_IF(NULL == fX, NULL);
    const size_t nr = X->nr;
    const size_t nrq = X->nrq;
    const size_t nc = X->nc;
    assert(nr == fX->nr);
    assert(nrq == fX->nrq);
    assert(nc == fX->nc);

    C = remake_flappie_matrix(C, nr, nc);
    RETURN_NULL_IF(NULL == C, NULL);

    for(size_t c=0 ; c < nc ; c++){
        const size_t offset = c * nrq;
        for(size_t r=0 ; r < nrq ; r++){
            C->data.v[offset + r] = X->data.v[offset + r] + fX->data.v[offset + r];
        }
    }

    return C;
}


void residual_inplace(const_flappie_matrix X, flappie_matrix fX) {
    RETURN_NULL_IF(NULL == X, );
    RETURN_NULL_IF(NULL == fX, );

    const size_t nrq = X->nrq;
    const size_t nc = X->nc;
    assert(X->nr == fX->nr);
    assert(nrq == fX->nrq);
    assert(nc == fX->nc);

    for(size_t c=0 ; c < nc ; c++){
        const size_t offset = c * nrq;
        for(size_t r=0 ; r < nrq ; r++){
            fX->data.v[offset + r] += X->data.v[offset + r];
        }
    }
}


flappie_matrix softmax(const_flappie_matrix X, const_flappie_matrix W,
                        const_flappie_matrix b, flappie_matrix C) {
    C = feedforward_exp(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    row_normalise_inplace(C);

    assert(validate_flappie_matrix
           (C, 0.0, 1.0, NAN, true, __FILE__, __LINE__));
    return C;
}


/**   Softmax with separate temperatures on weights and bias
 *
 *    Calculates softmax( A x / tempW + b / tempb ) as
 *    softmax( (A (x * tempb / tempW ) + b) / tempb )
 *
 *    @returns matrix containing softmax
 **/
flappie_matrix softmax_with_temperature(flappie_matrix X, const_flappie_matrix W,
                                         const_flappie_matrix b, float tempW, float tempb,
                                         flappie_matrix C) {
    RETURN_NULL_IF(NULL == X, NULL);

    shift_scale_matrix_inplace(X, 0.0f, tempW / tempb);

    C = feedforward_linear(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    shift_scale_matrix_inplace(C, 0.0f, tempb);
    exp_activation_inplace(C);
    row_normalise_inplace(C);

    assert(validate_flappie_matrix
           (C, 0.0, 1.0, NAN, true, __FILE__, __LINE__));
    return C;
}


flappie_matrix feedforward2_tanh(const_flappie_matrix Xf,
                                  const_flappie_matrix Xb,
                                  const_flappie_matrix Wf,
                                  const_flappie_matrix Wb,
                                  const_flappie_matrix b, flappie_matrix C) {
    C = affine_map2(Xf, Xb, Wf, Wb, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    tanh_activation_inplace(C);

    assert(validate_flappie_matrix(C, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return C;
}


flappie_matrix gru_forward(const_flappie_matrix X, const_flappie_matrix sW,
                            const_flappie_matrix sW2, flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);

    assert(NULL != sW);
    assert(NULL != sW2);

    const size_t bsize = X->nc;
    const size_t size = sW2->nc;
    assert(X->nr == 3 * size);
    assert(sW->nr == size);
    assert(sW2->nr == size);
    assert(sW->nc == 2 * size);
    assert(sW2->nc == size);

    ostate = remake_flappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    flappie_matrix tmp = make_flappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set second column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v + ostate->nrq, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = ostate->data.v + ostate->nrq;
    sCol2.data.v = ostate->data.v;
    gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = X->data.v + i * X->nrq;
        sCol1.data.v = ostate->data.v + (i - 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + i * ostate->nrq;
        gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    }

    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}


flappie_matrix gru_backward(const_flappie_matrix X, const_flappie_matrix sW,
                             const_flappie_matrix sW2, flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);
    assert(NULL != sW2);

    const size_t size = sW2->nc;
    const size_t bsize = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nr == size);
    assert(sW2->nr == size);
    assert(sW->nc == 2 * size);
    assert(sW2->nc == size);

    ostate = remake_flappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    flappie_matrix tmp = make_flappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
    sCol1.data.v = ostate->data.v;
    sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
    gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = X->data.v + index * X->nrq;
        sCol1.data.v = ostate->data.v + (index + 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + index * ostate->nrq;
        gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    }

    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}


void gru_step(const_flappie_matrix x, const_flappie_matrix istate,
              const_flappie_matrix sW, const_flappie_matrix sW2,
              flappie_matrix xF, flappie_matrix ostate) {
    /* Perform a single GRU step
     * x      is [isize]
     * istate is [size]
     * xW     is [isize, 3 * size]
     * sW     is [size, 2 * size]
     * sW2    is [size, size]
     * bias   is [3 * size]
     * xF     is [3 * size]
     * ostate is [size]
     */
    assert(NULL != x);
    assert(NULL != sW);
    assert(NULL != sW2);
    const size_t size = istate->nr;
    assert(x->nr == 3 * size);
    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    assert(size == sW->nr);
    assert(2 * size == sW->nc);
    assert(size == sW2->nr);
    assert(size == sW2->nc);
    assert(3 * size == xF->nr);
    assert(size == ostate->nr);

    // Copy input vector = iW x + b to temporary vector
    memcpy(xF->data.v, x->data.v, x->nrq * sizeof(__m128));
    /*  Add sW * istate to first 2 * size elts of xF
     *  then apply gate function to get r and z
     */
	//printf("4: Number of M=%d N=%d\n",sW->nr,sW->nc); 
    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
                sW->stride, istate->data.f, 1, 1.0, xF->data.f, 1);
    for (size_t i = 0; i < (sizeq +sizeq); i++) {
        xF->data.v[i] = LOGISTICFV(xF->data.v[i]);
    }

    const __m128 *z = xF->data.v;
    __m128 *r = xF->data.v + sizeq;
    __m128 *hbar = xF->data.v + sizeq + sizeq;
    for (size_t i = 0; i < sizeq; i++) {
        r[i] *= istate->data.v[i];
    }
	//printf("5: Number of M=%d N=%d\n",sW2->nr,sW2->nc); 
    cblas_sgemv(CblasColMajor, CblasTrans, sW2->nr, sW2->nc, 1.0, sW2->data.f,
                sW2->stride, (float *)r, 1, 1.0, (float *)hbar, 1);
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = TANHFV(hbar[i]);
    }

    const __m128 ones = _mm_set1_ps(1.0f);
    for (size_t i = 0; i < sizeq ; i++) {
        ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
    }
}


flappie_matrix grumod_forward(const_flappie_matrix X, const_flappie_matrix sW,
                               flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);

    assert(NULL != sW);

    const size_t bsize = X->nc;
    const size_t size = sW->nr;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    //fprintf(stderr, "grumod_forward A rows=%lu A cols=%lu sWnr=%lu sWnc=%lu\n",X->nr, X->nc, sW->nr, sW->nc); 

    ostate = remake_flappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    flappie_matrix tmp = make_flappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set second column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v + ostate->nrq, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = ostate->data.v + ostate->nrq;
    sCol2.data.v = ostate->data.v;
    grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = X->data.v + i * X->nrq;
        sCol1.data.v = ostate->data.v + (i - 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + i * ostate->nrq;
        grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    }

    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

#if STREAMING_ENABLED
void cblas_gemv_wrapper(const_flappie_matrix xCol, flappie_matrix xColTmp, const_flappie_matrix sW, flappie_matrix sCol1);

flappie_matrix aes_grumod_linear( const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, int backward, const_flappie_matrix W, const_flappie_matrix b) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
    cublasHandle_t handle ; // CUBLAS context

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
    float *d_a, *d_x, *d_y;	
    cudaStat = cudaMalloc (( void **)& d_a , 768*256*sizeof(float)); // device // memory alloc for a
    cudaStat = cudaMalloc (( void **)& d_x , 256*sizeof(float)); // device // memory alloc for x
    cudaStat = cudaMalloc (( void **)& d_y , 768*sizeof(float)); // device // memory alloc for y
    float al =1.0f;
    float bet =0.0f;
    stat = cublasCreate (&handle);

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
    		//Cin = xCol.data.f; Cout = xColTmp->data.f;  A = sW->data.f; Bnext = sCol2.data.f;
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

        	cblas_sgemv(CblasColMajor, CblasTrans, 256, 768, 1.0, A, 256, B, 1, 1.0, Cout, 1);
		stat = cublasSetMatrix (M,N, sizeof(float),A,M,d_a,M); 
		stat = cublasSetVector (N,sizeof(float),B,1,d_x,1); 
		stat = cublasSetVector (M,sizeof(float),Cout,1,d_y,1); 

        	for (size_t i = 0; i < size; i++) {
                	Cout[i] = LOGISTICF(Cout[i]); 
                	Cout[size+i] = LOGISTICF(Cout[size+i]); 
                	Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] + Cin[i+size+size]); 
                	Bnext[i] = (-1) * Cout[i] * Cout[i+size+size] + Cout[i+size+size]; 
                	Bnext[i] = Cout[i] * B[i] + Bnext[i];
        	}

                if(backward) {
       			XnextBuf.data.f = Xnext->data.f + (index+1) * Xnext->nr;
		}else {
       			XnextBuf.data.f = Xnext->data.f + (index-1) * Xnext->nr;
		}
        	cblas_sgemv(CblasColMajor, CblasTrans, W->nr, W->nc, 1.0, W->data.f, W->stride, B, 1, 1.0, XnextBuf.data.f, 1);
        	//cblas_sgemv(CblasColMajor, CblasTrans, W->nr, W->nc, 1.0, W->data.f, W->stride, Bnext, 1, 1.0, sCol2.data.f, 1);
	}

	// STORE
        {
	}
    }
    xColTmp = free_flappie_matrix(xColTmp);
    assert(validate_flappie_matrix (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));

    cudaFree (d_a ); 
    cudaFree (d_x );
    cudaFree (d_y );
    cublasDestroy ( handle );
    return Xnext;
}

flappie_matrix aes_grumod( const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, bool backward) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

    const size_t size = sW->nr;
    const size_t N = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);


    ostate = remake_flappie_matrix(ostate, size, N);
    flappie_matrix xColTmp = make_flappie_matrix(3 * size, 1);

    _Mat xCol, sCol1, sCol2;
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


    //float *Cin, *Cout, *A, *Bnext;
    float Cin[768], Cout[768], A[256*768]; 
    float *Bnext;

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
    		//Cin = xCol.data.f; Cout = xColTmp->data.f;  A = sW->data.f; Bnext = sCol2.data.f;
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
                if(backward) B = Bnext + 256; //B is ostate
                else B = Bnext - 256;
        	const size_t size = 256;
        	memcpy(Cout, Cin, 768 * sizeof(float) );
        	memset(Cout + size + size, 0, size *sizeof(float));

        	cblas_sgemv(CblasColMajor, CblasTrans, 256, 768, 1.0, A, 256, B, 1, 1.0, Cout, 1);
        	//cblas_sgemv(CblasRowMajor, CblasNoTrans, 768, 256, 1.0, A, 768, B, 1, 1.0, Cout, 1);

        	for (size_t i = 0; i < size; i++) {
                	Cout[i] = LOGISTICF(Cout[i]); 
                	Cout[size+i] = LOGISTICF(Cout[size+i]); 
                	Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] + Cin[i+size+size]); 
                	Bnext[i] = (-1) * Cout[i] * Cout[i+size+size] + Cout[i+size+size]; 
                	Bnext[i] = Cout[i] * B[i] + Bnext[i];
        	}
	}

	// STORE
        {
	}
    }
    xColTmp = free_flappie_matrix(xColTmp);
    assert(validate_flappie_matrix (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

flappie_matrix grumod_backward(const_flappie_matrix X, const_flappie_matrix sW,
                                flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

    const size_t size = sW->nr;
    const size_t N = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    ostate = remake_flappie_matrix(ostate, size, N);

    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
    sCol1.data.v = ostate->data.v;
    sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;

    flappie_matrix xColTmp = make_flappie_matrix(3 * size, 1);

    for (int i = 1; i < N; i++) {
        const size_t index = N - i - 1;
        xCol.data.f = X->data.f + index * X->nr;
        sCol1.data.f = ostate->data.f + (index + 1) * ostate->nr;
        sCol2.data.f = ostate->data.f + index * ostate->nr;
        //cblas_gemv_wrapper(&xCol, xColTmp, sW, &sCol2);
        { 
                flappie_matrix Cin = &xCol; flappie_matrix Cout = xColTmp;  flappie_matrix A = sW; flappie_matrix Bnext = &sCol2;
        	float *B = Bnext->data.f + Bnext->nr;
        	const size_t size = Bnext->nr;
        	memcpy(Cout->data.f, Cin->data.f, Cin->nr * sizeof(float) );
        	memset(Cout->data.f + size + size, 0, size *sizeof(float));

        	cblas_sgemv(CblasColMajor, CblasTrans, A->nr, A->nc, 1.0, A->data.f, A->stride, B, 1, 1.0, Cout->data.f, 1);

        	for (size_t i = 0; i < size; i++) {
                	Cout->data.f[i] = LOGISTICF(Cout->data.f[i]); 
                	Cout->data.f[size+i] = LOGISTICF(Cout->data.f[size+i]); 
                	Cout->data.f[i+size+size] = TANHF(Cout->data.f[i+size] * Cout->data.f[i+size+size] + Cin->data.f[i+size+size]); 
                	Bnext->data.f[i] = (-1) * Cout->data.f[i] * Cout->data.f[i+size+size] + Cout->data.f[i+size+size]; 
                	Bnext->data.f[i] = Cout->data.f[i] * B[i] + Bnext->data.f[i];
        	}
	}
    }
    xColTmp = free_flappie_matrix(xColTmp);
    assert(validate_flappie_matrix (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

static int data_available=0;

void cblas_gemv_wrapper(const_flappie_matrix Cin, flappie_matrix Cout, const_flappie_matrix A, flappie_matrix Bnext) {

        float *B = Bnext->data.f + Bnext->nr;
        const size_t size = Bnext->nr;
        memcpy(Cout->data.f, Cin->data.f, Cin->nr * sizeof(float) );
        memset(Cout->data.f + size + size, 0, size *sizeof(float));

        cblas_sgemv(CblasColMajor, CblasTrans, A->nr, A->nc, 1.0, A->data.f, A->stride, B, 1, 1.0, Cout->data.f, 1);

        for (size_t i = 0; i < size; i++) {
                Cout->data.f[i] = LOGISTICF(Cout->data.f[i]); // UPDATE gate z(t)
                Cout->data.f[size+i] = LOGISTICF(Cout->data.f[size+i]); // RESET gate r(t)
                Cout->data.f[i+size+size] = TANHF(Cout->data.f[i+size] * Cout->data.f[i+size+size] + Cin->data.f[i+size+size]); // ~O(t)
                Bnext->data.f[i] = (-1) * Cout->data.f[i] * Cout->data.f[i+size+size] + Cout->data.f[i+size+size];  // O(t) part 2
                Bnext->data.f[i] = Cout->data.f[i] * B[i] + Bnext->data.f[i];  // O(t) part 1
        }

}
#else 
flappie_matrix grumod_backward(const_flappie_matrix X, const_flappie_matrix sW,
                                flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

    const size_t size = sW->nr;
    const size_t bsize = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    ostate = remake_flappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    flappie_matrix tmp = make_flappie_matrix(3 * size, 1);
    //fprintf(stderr, "grumod_backward Xnr=%lu Xnc=%lu sWnr=%lu sWnc=%lu\n",X->nr, X->nc, sW->nr, sW->nc);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
    sCol1.data.v = ostate->data.v;
    sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
    grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = X->data.v + index * X->nrq;
        sCol1.data.v = ostate->data.v + (index + 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + index * ostate->nrq;
        grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    }

    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}
#endif

void PrintMatrix(float* pMatrix, const size_t nR, const size_t nC, const CBLAS_ORDER Order) {
    unsigned int i, j;
    if (Order == CblasRowMajor) {
        for (i = 0; i < nR; i++) {
            for (j = 0; j < nC; j++) {
                printf("%f \t ", pMatrix[i * nC + j]); // !!!
            }
            printf("\n"); // !!!
        }
    } else {
        for (i = 0; i < nR; i++) {
            for (j = 0; j < nC; j++) {
                printf("%f \t ", pMatrix[i + j* nR ]); // !!!
            }
            printf("\n"); // !!!
        }
    }
    printf("\n"); // !!!
}

static int print_flag=0;
int max(int x, int y){
  if(x>y) return x;
  else return y;
}

void grumod_step(const_flappie_matrix x, const_flappie_matrix istate,
                 const_flappie_matrix sW, flappie_matrix xF,
                 flappie_matrix ostate) {
    /* Perform a single modified GRU step
     * x      is [isize]
     * istate is [size]
     * xW     is [isize, 3 * size]
     * sW     is [size, 2 * size]
     * sW2    is [size, size]
     * bias   is [3 * size]
     * xF     is [3 * size]
     * ostate is [size]
     */
    //ekf_hw hw;
    assert(NULL != x);
    assert(NULL != sW);
    const size_t size = istate->nr;
    assert(x->nr == 3 * size);
    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    assert(size == sW->nr);
    assert(3 * size == sW->nc);
    assert(3 * size == xF->nr);
    assert(size == ostate->nr);
    int lda, ldb, ldc;

    //fprintf(stderr,"grumod_step: Number of istate->nr=%lu istate->nc=%lu xF->nr=%lu xF->nc=%lu,\n",istate->nr,istate->nc, xF->nr, xF->nc); // 25i6256x1 and 768x1

    // Copy input vector = iW x + b to temporary vector and zero last chunk
    memcpy(xF->data.v, x->data.v, x->nrq * sizeof(__m128));
    memset(xF->data.v + sizeq + sizeq, 0, sizeq *sizeof(__m128));
    /*  Add sW * istate to first 3 * size elts of xF
     *  then apply gate function to get r and z
     */

    //            order, transa, M, N, alpha, A, lda, X, incX, beta, Y, incY

    if(print_flag) {
      printf("Printing xF matrix before multiplication\n"); 
      PrintMatrix(xF->data.f, xF->nr, xF->nc, CblasColMajor);
    }


    /*int i;
    for(i=0 ; i<6; i++) { 
        cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc/6, 1.0, sW->data.f+(i*128*256), sW->stride, istate->data.f, 1, 1.0, xF->data.f+i*128, 1);
    }
     works cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, sW->nc, 1 , sW->nr, 1.0, sW->data.f, 256, istate->data.f, 256, 1.0, xF->data.f, 768);
     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 768, 1 , 256, 1.0, sW->data.f, 256, istate->data.f, 1, 1.0, xF->data.f, 1);

    if(print_flag) {
      printf("Printing sW matrix\n");
      PrintMatrix(sW->data.f, sW->nr, sW->nc, CblasColMajor);
      printf("Printing istate matrix\n");
      PrintMatrix(istate->data.f, istate->nr, istate->nc, CblasColMajor);
      printf("Printing xFmatrix after multiplication\n");
      PrintMatrix(xF->data.f, xF->nr, xF->nc, CblasColMajor);
      exit(1);
    }
    print_flag = 0;*/

    /*float *a1 = (float *) vio_malloc(sW->nr * sW->nc *sizeof(float)); //256x768
    float *b1 = (float *) vio_malloc(istate->nr * istate->nc * sizeof(float)); // 256x1
    float *c1 = (float *) vio_malloc(xF->nr * xF->nc * sizeof(float)); // 768x1
    memcpy(a1, sW->data.f, sW->nr * sW->nc * sizeof(float));
    memcpy(b1, istate->data.f, istate->nr * istate->nc * sizeof(float));
    memcpy(c1, xF->data.f, xF->nr * xF->nc * sizeof(float)); 

    if(!data_available) {
        hw.mat_mul(a1,b1,c1,c1, 768, 256, 1, 0,0, 1.0, 0.0, 256, 4, 4, 0,0,0);
        data_available = 1;
    }
    else {
        hw.mat_mul(NULL, b1,c1,c1, 768, 256, 1, 0,0, 1.0, 0.0, 256, 4, 4, 0,0,0);
    }
 
    
    memcpy(xF->data.f, c1, xF->nr * xF->nc * sizeof(float));
    vio_free(a1);
    vio_free(b1);
    vio_free(c1); */

    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f, sW->stride, istate->data.f, 1, 1.0, xF->data.f, 1);

    for (size_t i = 0; i < (size + size); i++) {
        xF->data.f[i] = LOGISTICF(xF->data.f[i]);
    }

    const float *z = xF->data.f;
    const float *a = xF->data.f + size;
    float *b = xF->data.f + size + size;
    float *c =  x->data.f + size + size;


    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] * b[i] + c[i]; // cin and cout are same and below tanh will be done in place
        //hbar[i] = r[i] * hbar[i] + x->data.f[size + size + i];
    }

    for (size_t i = 0; i < size; i++) {
        c[i] = TANHF(c[i]);
        //hbar[i] = TANHF(hbar[i]);
    }

    const float ones = 1.0f;
    float *c2 = ostate->data.f; 

    for (size_t i = 0; i < size ; i++) {
        c2[i] = (-1) * z[i] * c[i] + c[i]; // cin and cout are different. alpha is -1.
        c2[i] = z[i] * istate->data.f[i] + c2[i]; // cin and cout are same. 
        //ostate->data.f[i] = z[i] * istate->data.f[i] + (ones - z[i]) * hbar[i];
    }

    /* const __m128 *z = xF->data.v;
    const __m128 *r = xF->data.v + sizeq;
    __m128 *hbar = xF->data.v + sizeq + sizeq;
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = r[i] * hbar[i] + x->data.v[sizeq + sizeq + i];
    }
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = TANHFV(hbar[i]);
    }

    const __m128 ones = _mm_set1_ps(1.0f);
    for (size_t i = 0; i < sizeq ; i++) {
        ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
    } */
}


flappie_matrix gru_relu_forward(const_flappie_matrix X, const_flappie_matrix sW,
                                const_flappie_matrix sW2, flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);

    assert(NULL != sW);
    assert(NULL != sW2);

    const size_t bsize = X->nc;
    const size_t size = sW2->nc;
    assert(X->nr == 3 * size);
    assert(sW->nr == size);
    assert(sW2->nr == size);
    assert(sW->nc == 2 * size);
    assert(sW2->nc == size);

    ostate = remake_flappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    flappie_matrix tmp = make_flappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set second column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v + ostate->nrq, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = ostate->data.v + ostate->nrq;
    sCol2.data.v = ostate->data.v;
    gru_relu_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = X->data.v + i * X->nrq;
        sCol1.data.v = ostate->data.v + (i - 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + i * ostate->nrq;
        gru_relu_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    }

    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (ostate, 0.0, HUGE_VAL, 0.0, true, __FILE__, __LINE__));
    return ostate;
}


flappie_matrix gru_relu_backward(const_flappie_matrix X, const_flappie_matrix sW,
                                 const_flappie_matrix sW2, flappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);
    assert(NULL != sW2);

    const size_t size = sW2->nc;
    const size_t bsize = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nr == size);
    assert(sW2->nr == size);
    assert(sW->nc == 2 * size);
    assert(sW2->nc == size);

    ostate = remake_flappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    flappie_matrix tmp = make_flappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
    sCol1.data.v = ostate->data.v;
    sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
    gru_relu_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = X->data.v + index * X->nrq;
        sCol1.data.v = ostate->data.v + (index + 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + index * ostate->nrq;
        gru_relu_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    }

    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (ostate, 0.0, HUGE_VAL, 0.0, true, __FILE__, __LINE__));
    return ostate;
}


void gru_relu_step(const_flappie_matrix x, const_flappie_matrix istate,
                   const_flappie_matrix sW, const_flappie_matrix sW2,
                   flappie_matrix xF, flappie_matrix ostate) {
    /* Perform a single GRU step
     * x      is [isize]
     * istate is [size]
     * xW     is [isize, 3 * size]
     * sW     is [size, 2 * size]
     * sW2    is [size, size]
     * bias   is [3 * size]
     * xF     is [3 * size]
     * ostate is [size]
     */
    assert(NULL != x);
    assert(NULL != sW);
    assert(NULL != sW2);
    const size_t size = istate->nr;
    assert(x->nr == 3 * size);
    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    assert(size == sW->nr);
    assert(2 * size == sW->nc);
    assert(size == sW2->nr);
    assert(size == sW2->nc);
    assert(3 * size == xF->nr);
    assert(size == ostate->nr);


    // Copy input vector = iW x + b to temporary vector
    memcpy(xF->data.v, x->data.v, x->nrq * sizeof(__m128));
    /*  Add sW * istate to first 2 * size elts of xF
     *  then apply gate function to get r and z
     */ 
    //printf("7: Number of M=%d N=%d\n",sW->nr,sW->nc); 
    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
                sW->stride, istate->data.f, 1, 1.0, xF->data.f, 1);
    for (size_t i = 0; i < (sizeq +sizeq); i++) {
        xF->data.v[i] = LOGISTICFV(xF->data.v[i]);
    }

    const __m128 *z = xF->data.v;
    __m128 *r = xF->data.v + sizeq;
    __m128 *hbar = xF->data.v + sizeq + sizeq;
    for (size_t i = 0; i < sizeq; i++) {
        r[i] *= istate->data.v[i];
    }
    //printf("8: Number of M=%d N=%d\n",sW2->nr,sW2->nc); 
    cblas_sgemv(CblasColMajor, CblasTrans, sW2->nr, sW2->nc, 1.0, sW2->data.f,
                sW2->stride, (float *)r, 1, 1.0, (float *)hbar, 1);
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = relufv(hbar[i]);
    }

    const __m128 ones = _mm_set1_ps(1.0f);
    for (size_t i = 0; i < sizeq ; i++) {
        ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
    }
}


flappie_matrix lstm_forward(const_flappie_matrix Xaffine,
                             const_flappie_matrix sW, const_flappie_matrix p,
                             flappie_matrix output) {
    RETURN_NULL_IF(NULL == Xaffine, NULL);
    assert(NULL != sW);
    assert(NULL != p);

    const size_t size = sW->nr;
    const size_t bsize = Xaffine->nc;
    assert(Xaffine->nr == 4 * size);
    assert(p->nr == 3 * size);
    assert(sW->nc == 4 * size);

    output = remake_flappie_matrix(output, size, bsize);
    RETURN_NULL_IF(NULL == output, NULL);

    flappie_matrix tmp = make_flappie_matrix(4 * size, 1);
    flappie_matrix state = make_flappie_matrix(size, 1);
    if(NULL == tmp || NULL == state){
        //  Memory allocation falled, clean-up and return
        free(state);
        free(tmp);
        free(output);
        return NULL;
    }

    /* First step state & output are zero.  Set second column of output to zero and use that */
    memset(output->data.v + output->nrq, 0, output->nrq * sizeof(__m128));
    _Mat xCol, sCol1, sCol2;
    xCol = *Xaffine;
    sCol1 = *output;
    sCol2 = *output;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = output->data.v + output->nrq;
    sCol2.data.v = output->data.v;
    lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = Xaffine->data.v + i * Xaffine->nrq;
        sCol1.data.v = output->data.v + (i - 1) * output->nrq;
        sCol2.data.v = output->data.v + i * output->nrq;
        lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    }

    state = free_flappie_matrix(state);
    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (output, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return output;
}


flappie_matrix lstm_backward(const_flappie_matrix Xaffine,
                              const_flappie_matrix sW, const_flappie_matrix p,
                              flappie_matrix output) {
    RETURN_NULL_IF(NULL == Xaffine, NULL);
    assert(NULL != sW);
    assert(NULL != p);

    const size_t size = sW->nr;
    const size_t bsize = Xaffine->nc;
    assert(Xaffine->nr == 4 * size);
    assert(sW->nc == 4 * size);
    assert(p->nr == 3 * size);

    output = remake_flappie_matrix(output, size, bsize);
    RETURN_NULL_IF(NULL == output, NULL);

    flappie_matrix tmp = make_flappie_matrix(4 * size, 1);
    flappie_matrix state = make_flappie_matrix(size, 1);
    if(NULL == tmp || NULL == state){
        //  Memory allocation falled, clean-up and return
        free(state);
        free(tmp);
        free(output);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    memset(output->data.v, 0, output->nrq * sizeof(__m128));
    _Mat xCol, sCol1, sCol2;
    xCol = *Xaffine;
    sCol1 = *output;
    sCol2 = *output;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = Xaffine->data.v + (bsize - 1) * Xaffine->nrq;
    sCol1.data.v = output->data.v;
    sCol2.data.v = output->data.v + (bsize - 1) * output->nrq;
    lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = Xaffine->data.v + index * Xaffine->nrq;
        sCol1.data.v = output->data.v + (index + 1) * output->nrq;
        sCol2.data.v = output->data.v + index * output->nrq;
        lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    }

    state = free_flappie_matrix(state);
    tmp = free_flappie_matrix(tmp);

    assert(validate_flappie_matrix
           (output, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return output;
}


void lstm_step(const_flappie_matrix xAffine, const_flappie_matrix out_prev,
               const_flappie_matrix sW, const_flappie_matrix peep,
               flappie_matrix xF, flappie_matrix state,
               flappie_matrix output) {
    /* Perform a single LSTM step
     * xAffine  is [isize] (== iW x + b, where x is the input to the LSTM layer)
     * out_prev is [size]
     * sW       is [size, 4 * size]
     * peep     is [4 * size]
     * xF       is [4 * size]
     * state    is [size]
     * output   is [size]
     */
    assert(NULL != xAffine);
    assert(NULL != out_prev);
    assert(NULL != sW);
    assert(NULL != peep);
    assert(NULL != xF);
    assert(NULL != state);
    assert(NULL != output);
    const size_t size = state->nr;
    assert(xAffine->nr == 4 * size);
    assert(size == out_prev->nr);
    assert(size == sW->nr);
    assert(4 * size == sW->nc);
    assert(3 * size == peep->nr);
    assert(4 * size == xF->nr);
    assert(size == output->nr);

    // Copy input vector = iW x + b to temporary vector
    memcpy(xF->data.v, xAffine->data.v, xAffine->nrq * sizeof(__m128));
    //  + sW' * xprev
    //printf("9: Number of M=%d N=%d\n",sW->nr,sW->nc); 
    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
                sW->stride, out_prev->data.f, 1, 1.0, xF->data.f, 1);

    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    for (size_t i = 0; i < sizeq; i++) {
        // Forget gate
        __m128 forget = LOGISTICFV(xF->data.v[2 * sizeq + i]
                                   + state->data.v[i] * peep->data.v[sizeq + i])
            * state->data.v[i];
        // Update gate
        __m128 update = LOGISTICFV(xF->data.v[sizeq + i]
                                   + state->data.v[i] * peep->data.v[i])
            * TANHFV(xF->data.v[i]);
        state->data.v[i] = _mm_add_ps(forget, update);
        // Output gate
        output->data.v[i] = LOGISTICFV(xF->data.v[3 * sizeq + i]
                                       +
                                       state->data.v[i] * peep->data.v[2 *
                                                                       sizeq +
                                                                       i])
            * TANHFV(state->data.v[i]);
    }
}


size_t nbase_from_flipflop_nparam(size_t nparam){
    return roundf((-1.0f + sqrtf(1 + 2 * nparam)) / 2.0f);
}


double crf_manystay_partition_function(const_flappie_matrix C){
    RETURN_NULL_IF(NULL == C, NAN);
    const size_t nbase = nbase_from_flipflop_nparam(C->nr);
    const size_t nstate = nbase + nbase;
    assert(nstate * (nbase + 1) == C->nr);

    double * mem = calloc(2 * nstate, sizeof(double));
    RETURN_NULL_IF(NULL==mem, NAN);

    double * curr = mem;
    double * prev = mem + nstate;

    for(size_t c=0 ; c < C->nc ; c++){
        const size_t offset = c * C->stride;
        const size_t offset_stay = offset + nstate * nbase;
        //  Swap
        {
            double * tmp = curr;
            curr = prev;
            prev = tmp;
        }

        for(size_t stay=nbase ; stay < nstate ; stay++){
           const size_t from_base = stay - nbase;
           curr[stay] = logsumexp(prev[stay] + C->data.f[offset_stay + stay],
                                   prev[from_base] + C->data.f[offset_stay + from_base]);
        }
        for(size_t to_state=0 ; to_state < nbase ; to_state++){
            const size_t offsetS = offset + to_state * nstate;
            curr[to_state] = C->data.f[offsetS + 0] + prev[0];
            for(size_t from_state=1 ; from_state < nstate ; from_state++){
                curr[to_state] = logsumexp(curr[to_state], C->data.f[offsetS + from_state] + prev[from_state]);
            }
        }
    }

    double logZ = curr[0];
    for(size_t st=1 ; st < nstate ; st++){
        logZ = logsumexp(logZ, curr[st]);
    }

    free(mem);

    return logZ;
}


flappie_matrix globalnorm_manystay(const_flappie_matrix X, const_flappie_matrix W,
                                    const_flappie_matrix b, float temperature, flappie_matrix C) {
    C = affine_map(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);
    tanh_activation_inplace(C);
    shift_scale_matrix_inplace(C, 0.0f, temperature / 5.0f);

    float logZ = crf_manystay_partition_function(C) / (double)C->nc;

    for(size_t c=0 ; c < C->nc ; c++){
        const size_t offset = c * C->stride;
        for(size_t r=0 ; r < C->nr ; r++){
            C->data.f[offset + r] -= logZ;
        }
    }


    return C;
}


flappie_matrix globalnorm_flipflop(const_flappie_matrix X, const_flappie_matrix W,
                                    const_flappie_matrix b, float temperature, flappie_matrix C) {
    return globalnorm_manystay(X, W, b, temperature, C);
}
