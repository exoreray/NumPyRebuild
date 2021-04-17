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
        for(unsigned int i = 0; i < (mat1->rows)*(mat1->cols) / 4 * 4; i += 4) {
            __m256d m1 =  _mm256_loadu_pd(mat1->data + i);
            __m256d m2 =  _mm256_loadu_pd(mat2->data + i);
            __m256d m3 =  _mm256_add_pd(m1, m2);
            _mm256_storeu_pd(result->data + i, m3);
        }
    }
    #pragma omp for
    for (int i = (mat1->rows)*(mat1->cols) / 4 * 4; i < (mat1->rows)*(mat1->cols); i++) {
        result->data[i] = mat1->data[i]+mat2->data[i];
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
        for(unsigned int i = 0; i < (mat1->rows)*(mat1->cols) / 4 * 4; i += 4) {
            __m256d m1 =  _mm256_loadu_pd(mat1->data + i);
            __m256d m2 =  _mm256_loadu_pd(mat2->data + i);
            __m256d m3 =  _mm256_sub_pd(m1, m2);
            _mm256_storeu_pd(result->data + i, m3);
        }
    }
    #pragma omp for
    for (int i = (mat1->rows)*(mat1->cols) / 4 * 4; i < (mat1->rows)*(mat1->cols); i++) {
        result->data[i] = mat1->data[i]-mat2->data[i];
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
//    matrix *temp = NULL;
//    if (result == mat1 || result == mat2){
//        allocate_matrix(&temp, result->rows, result->cols);
//    }else{
//        temp = result;
//    }

//    for (int i = 0; i < (mat1->rows); i++) {
//        for (int j = 0; j < (mat2->cols); j++) {
//            double value = 0;
//            for (int v = 0; v < (mat1->cols); v++) {
//                value += mat1->data[i*(mat1->cols)+v]*mat2->data[v*(mat2->cols)+j];
//            }
//            result->data[i * (mat1->rows) + j] = value;
//        }
//    }
//
//    return 0;

    #pragma omp parallel
    {
    #pragma omp for
        for (int j = 0; j < (mat2->cols); j++) {
            for (int i = 0; i < (mat1->rows); i++) {
//                double *m2col = malloc(sizeof(double)*mat2->rows);
//                for (int w = 0; w < (mat2->rows); w++) {
//                    m2col[w] = mat2->data[w*mat2->cols + j];
//                }
                for (int v = 0; v < (mat1->cols); v++) {
                    result->data[i * (mat1->rows) + j] +=
                            mat1->data[i*(mat1->cols)+v]*mat2->data[v*(mat2->cols)+j];
                }
            }
        }
    }
    return 0;

}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
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
    if (mat->cols!=mat->rows){
        return -1;
    }
    result->data = helper(mat, pow)->data;
    return 0;
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

