#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    struct matrix *temp = malloc(sizeof(struct matrix));
    if (temp == NULL){
        return -1;
    }
    if (rows<=0 || cols<=0){
        return -1;
    }
    temp->rows = rows;
    temp->cols = cols;

    temp->data = calloc(rows*cols, sizeof(double));
    if (temp->data == NULL){  // allocate fail
        return -1;
    }
    temp->parent = NULL;
    temp->ref_cnt = 1;
    *mat = temp;
    return 0;
// int allocate_matrix(matrix **mat, int rows, int cols) {
//    struct matrix *temp;
//    struct matrix m;
//    temp = &m;
//    mat = &temp;
//    (*mat)->rows = rows;
//    (*mat)->cols = cols;
//    if (rows<=0 || cols<=0){
//        return -1;
//    }
//    (*mat)->data = calloc(rows*cols, sizeof(double));
//    if ((*mat)->data == NULL){  // allocate fail
//        return -1;
//    }
//    (*mat)->parent = NULL;
//    (*mat)->ref_cnt = 1;
//    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails.
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
//    struct matrix *temp;
//    temp = *mat;
//    struct matrix m;
//    temp = &m;
//    mat = &temp;
//    (*mat)->rows = rows;
//    (*mat)->cols = cols;
//    if (rows<=0 || cols<=0){
//        return -1;
//    }
//    (*mat)->data = from->data + offset;
//    if ((*mat)->data == NULL){  // allocate fail
//        return -1;
//    }
//    (*mat)->parent = from;
//    from->ref_cnt += 1;
//    (*mat)->ref_cnt = from->ref_cnt;
//    return 0;
    struct matrix *temp = malloc(sizeof(struct matrix));
    if (temp == NULL){
        return -1;
    }
    if (rows<=0 || cols<=0){
        return -1;
    }
    temp->rows = rows;
    temp->cols = cols;
    temp->data = from->data + offset;
    temp->parent = from;
    from->ref_cnt += 1;
    *mat = temp;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat == NULL){
        return;
    }
    if ((mat->parent == NULL && mat->ref_cnt == 1) ||
    (mat->parent != NULL && mat->parent->ref_cnt == 1 && mat->ref_cnt == 1) ){
        free(mat->data);
    }
    mat = NULL;

/*  if `mat` is not a slice || and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references*/
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    return mat->data[(mat->cols)*row + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
//    printf("%d, %d, %f, %p", row, col, val, mat);
    mat->data[(mat->cols)*row + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    for (int i = 0; i < (mat->rows)*(mat->cols); i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols){
        return -1;
    }
//    for (int i = 0; i < (mat1->rows)*(mat1->cols); i++) {
//        result->data[i] = mat1->data[i]+mat2->data[i];
//    }
//    return 0;
    #pragma omp parallel
    {
        #pragma omp for
        for(unsigned int i = 0; i < (mat1->rows)*(mat1->cols) / 8 * 8; i += 8) {
            __m256d m1 =  _mm256_loadu_pd(mat1->data + i);
            __m256d m2 =  _mm256_loadu_pd(mat2->data + i);
            __m256d m3 =  _mm256_loadu_pd(mat1->data + i + 4);
            __m256d m4 =  _mm256_loadu_pd(mat2->data + i + 4);
            __m256d m5 =  _mm256_add_pd(m1, m2);
            __m256d m6 =  _mm256_add_pd(m3, m4);
            _mm256_storeu_pd(result->data + i, m5);
            _mm256_storeu_pd(result->data + i + 4, m6);
        }
    }
    #pragma omp for
    for (int i = (mat1->rows)*(mat1->cols) / 8 * 8; i < (mat1->rows)*(mat1->cols); i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
//    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols){
//        return -1;
//    }
//    for (int i = 0; i < (mat1->rows)*(mat1->cols); i++) {
//        result->data[i] = mat1->data[i]-mat2->data[i];
//    }
//    return 0;
#pragma omp parallel
    {
#pragma omp for
        for(unsigned int i = 0; i < (mat1->rows)*(mat1->cols) / 8 * 8; i += 8) {
            __m256d m1 =  _mm256_loadu_pd(mat1->data + i);
            __m256d m2 =  _mm256_loadu_pd(mat2->data + i);
            __m256d m3 =  _mm256_loadu_pd(mat1->data + i + 4);
            __m256d m4 =  _mm256_loadu_pd(mat2->data + i + 4);
            __m256d m5 =  _mm256_sub_pd(m1, m2);
            __m256d m6 =  _mm256_sub_pd(m3, m4);
            _mm256_storeu_pd(result->data + i, m5);
            _mm256_storeu_pd(result->data + i + 4, m6);
        }
    }
#pragma omp for
    for (int i = (mat1->rows)*(mat1->cols) / 8 * 8; i < (mat1->rows)*(mat1->cols); i++) {
        result->data[i] = mat1->data[i] - mat2->data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->cols != mat2->rows){
        return -1;
    }
//// non parallel solution
//    matrix *temp = NULL;
//    if (result == mat1 || result == mat2){
//        allocate_matrix(&temp, result->rows, result->cols);
//    }else{
//        temp = result;
//    } TA says no need to consider this scenario.

//    for (int i = 0; i < (mat1->rows); i++) {
//        for (int j = 0; j < (mat2->cols); j++) {
//            for (int v = 0; v < (mat1->cols); v++) {
//                result->data[i * (mat2->cols) + j] += mat1->data[i*(mat1->cols)+v]*mat2->data[v*(mat2->cols)+j];
//            }
//        }
//    }
//    return 0;

//// debug:
    for (int i = 0; i < mat1->rows; i++) {
        printf("\n");
        for (int j = 0; j < mat1->cols; j++) {
            printf("%lf,", mat1->data[(i * mat1->cols) + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < mat2->rows; i++) {
        printf("\n");
        for (int j = 0; j < mat2->cols; j++) {
            printf("%lf,", mat2->data[(i * mat2->cols) + j]);
        }
        printf("\n");
    }

//// parallel solution:
//  make m2 trans matrix solution:
    int m2t_rows = mat2->cols;
    int m2t_cols = mat2->rows;
    int result_rows = mat1->rows;
    int result_cols = mat2->cols;
    double* m2trans = calloc(m2t_rows * m2t_cols , sizeof(double));
//  swap position block by block
#pragma omp for
    for (int i = 0; i < mat2->rows / 4 * 4; i += 4){
        for (int j = 0; j < mat2->cols / 4 * 4; j += 4) {
            __m256d block0;
            __m256d block1;
            __m256d block2;
            __m256d block3;

            double* data3 = mat2->data + ((i + 0) * mat2->cols) + j;
            double* data2 = mat2->data + ((i + 1) * mat2->cols) + j;
            double* data1 = mat2->data + ((i + 2) * mat2->cols) + j;
            double* data0 = mat2->data + ((i + 3) * mat2->cols) + j;

            block0 = _mm256_set_pd(*(data0 + 0), *(data1 + 0), *(data2 + 0), *(data3 + 0));
            block1 = _mm256_set_pd(*(data0 + 1), *(data1 + 1), *(data2 + 1), *(data3 + 1));
            block2 = _mm256_set_pd(*(data0 + 2), *(data1 + 2), *(data2 + 2), *(data3 + 2));
            block3 = _mm256_set_pd(*(data0 + 3), *(data1 + 3), *(data2 + 3), *(data3 + 3));

            _mm256_storeu_pd(m2trans + ((j + 0) * m2t_cols) + i, block0);
            _mm256_storeu_pd(m2trans + ((j + 1) * m2t_cols) + i, block1);
            _mm256_storeu_pd(m2trans + ((j + 2) * m2t_cols) + i, block2);
            _mm256_storeu_pd(m2trans + ((j + 3) * m2t_cols) + i, block3);
        }
    }

////debug:
//    for (int i = 0; i < mat2->cols; i++) {
//        printf("\n");
//        for (int j = 0; j < mat2->rows; j++) {
//            printf("%lf,", m2trans[(i * m2t_cols) + j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
//    printf("\n");

    // tail case
    for (int i = 0; i < mat2->rows / 4 * 4; i++) {
        for (int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
            *(m2trans + (m2t_cols * j) + i) = *(mat2->data + (mat2->cols * i) + j);
        }
    }
    for (int i = mat2->rows / 4 * 4; i < mat2->rows; i++) {
        for (int j = 0; j < mat2->cols; j++) {
            *(m2trans + (m2t_cols * j) + i) = *(mat2->data + (mat2->cols * i) + j);
        }
    }
////debug:
    for (int i = 0; i < mat2->cols; i++) {
        printf("\n");
        for (int j = 0; j < mat2->rows; j++) {
    		printf("%lf,", m2trans[(i * m2t_cols) + j]);
    	}
    	printf("\n");
    }


// computation:
#pragma omp parallel for
    for (int i = 0; i < result_rows; i++)
    {
        for (int j = 0; j < m2t_rows; j++)
        {
            __m256d m1m2t = _mm256_set1_pd(0);
            double block[4] = {0, 0, 0, 0};
            for (int k = 0; k < m2t_cols / 4 * 4; k+=4)
            {
////              non simd solution:
//            for (int k = 0; k < m2t_cols; k++) {
//                result->data[i * result_cols + j] +=
//                        mat1->data[i * (mat1->cols) + k] * m2trans[j * (mat2->rows) + k];
////              debug:
//                printf("result[%d] += mat1[%d] * m2trans[%d]\n",
//                       i * result_cols + j, i * (mat1->cols) + k, j * (mat2->rows) + k);
                __m256d m1 = _mm256_loadu_pd(mat1->data + i * mat1->cols + k);
                double b1[4] = {0, 0, 0, 0};
                _mm256_storeu_pd(b1,m1);
                printf("m1 block: %lf, %lf, %lf, %lf, \n", b1[0], b1[1], b1[2], b1[3]);
                __m256d m2t = _mm256_loadu_pd(m2trans + i * mat1->cols + k);
                double b2[4] = {0, 0, 0, 0};
                _mm256_storeu_pd(b2,m2t);
                printf("m2t block: %lf, %lf, %lf, %lf, \n", b2[0], b2[1], b2[2], b2[3]);
                m1m2t = _mm256_add_pd(m1m2t, _mm256_mul_pd(m1, m2t));
                _mm256_storeu_pd(block,m1m2t);
                printf("block sum: %lf, %lf, %lf, %lf, k: %d\n", block[0], block[1], block[2], block[3], k);

            }
//            _mm256_storeu_pd(block,m1m2t);
            double sum = block[0] + block[1] + block[2] + block[3];
//            printf("block sum: %lf, %lf, %lf, %lf, ", block[0], block[1], block[2], block[3]);

////debug:
            // debug:
            for (int i = 0; i < mat1->rows; i++) {
                printf("\n");
                for (int j = 0; j < mat2->cols; j++) {
                    printf("%lf,", result->data[(i * mat2->cols) + j]);
                }
                printf("\n");
            }


            // tail case for  [][][][]..  *   [][][][]..
            for (int k = m2t_cols / 4 * 4; k < m2t_cols; k++)
            {
                sum += (*(mat1->data + i * mat1->cols + k)) * m2trans[j * m2t_cols + k];
            }
            *(result->data + i * result_cols + j) = sum;



        }
    }
// debug:
    for (int i = 0; i < mat1->rows; i++) {
        printf("\n");
        for (int j = 0; j < mat2->cols; j++) {
            printf("%lf,", result->data[(i * mat2->cols) + j]);
        }
        printf("\n");
    }

    return 0;

////  no transpose solution:
//    for (int i = 0; i < mat2->rows / 4 * 4; i += 4){
//        for (int j = 0; j < mat1->rows; j += 1) {
//            for (int k = 0; k < mat2->cols; k += 1) {
//            double* data0 = *(mat2->data) + (i + 0) * mat2->cols;
//            double* data1 = *(mat2->data) + (i + 1) * mat2->cols;
//            double* data2 = *(mat2->data) + (i + 2) * mat2->cols;
//            double* data3 = *(mat2->data) + (i + 3) * mat2->cols;
//            block0 = _mm256_set_pd(*(data0 + k), *(data1 + k), *(data2 + k), *(data3 + k));
//            _mm256_storeu_pd(result + (j + 0) * result->cols + i, block1);
//            }
//        }
//    }

}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */

//solution 1:
struct matrix* helper(matrix *mat, int pow){ // recursion will have performance issue with large power.
    if (pow == 1){
        return mat;
    }
    struct matrix* temp = NULL;
    allocate_matrix(&temp, mat->rows, mat->cols);
    struct matrix* preTemp;
    preTemp = helper(mat, pow-1);
    mul_matrix(temp, mat, preTemp);
    if (pow > 2){
        deallocate_matrix(preTemp);
    }
    return temp;
}

int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
//    solution 1:
    if (mat->cols!=mat->rows){
        return -1;
    }
    result->data = helper(mat, pow)->data;
    return 0;


//    int currPow = 1;
//    struct matrix* temp = NULL;
//    allocate_matrix(&temp, mat->rows, mat->cols);
//    for (int i = 0; i < mat->rows; i++) {
//        for (int j = 0; j < mat->cols; j++) {
//            set(temp, i, j, mat->data[(mat->cols)*i + j]);
//        }
//    }
//    while(currPow < pow){
//
//        mul_matrix(result, temp, temp);
//        for (int i = 0; i < mat->rows; i++) {
//            for (int j = 0; j < mat->cols; j++) {
//                set(temp, i, j, result->data[(mat->cols)*i + j]);
//            }
//        }
//        currPow *= 2;
//    }
//    deallocate_matrix(temp);
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat == NULL){
        return -1;
    }
    for (int i = 0; i < (mat->rows)*(mat->cols); i++) {
        result->data[i] = -mat->data[i];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat == NULL){
        return -1;
    }
    for (int i = 0; i < (mat->rows)*(mat->cols); i++) {
        if (mat->data[i]<0){
            result->data[i] = -mat->data[i];
        }else{
            result->data[i] = mat->data[i];
        }
    }
    return 0;
}

