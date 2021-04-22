# numc
  numc.c:
  Info: Python/C API Reference
Here is the link to the full reference manual: https://docs.python.org/3.6/c-api/index.html. If you ever find anything confusing in the skeleton code or are at a lost on how to implement src/numc.c, this is a great resource.
We define the Matrix61c struct in numc.h. It is of type PyObject (this means you can always cast Matrix61c to PyObject, but not vice versa), which according to the official documentation, “contains the information Python needs to treat a pointer to an object as an object”. Our Matrix61c has the matrix struct we defined in src/matrix.h.

Then we define a struct PyTypeObject named Matrix61cType to specify the intended behaviors of our Python object Matrix61c. This struct will then be initialized to be our numc.Matrix objects.
![image](https://user-images.githubusercontent.com/46427258/115670131-ec4fa400-a2fd-11eb-9fa6-f46633784ba5.png)
For example, .tp_dealloc tells Python which function to call to destroy a numc.Matrix object when its reference count becomes 0, and .tp_members tells Python what instance attributes numc.Matrix objects have. You can take a look at the official documentation if you are curious.
https://docs.python.org/3.6/c-api/typeobj.html

Here's what I did in project 4:
-
simple:
  add:
  #pragma omp for
  unrolling : 8 - 1
  neg:
              _mm256_storeu_pd(result->data + i, _mm256_sub_pd(zero, m1));
  abs:
              _mm256_storeu_pd(result->data + i, _mm256_max_pd(_mm256_sub_pd(zero, m1), m1));

multiply:
![image](https://user-images.githubusercontent.com/46427258/115668750-56ffe000-a2fc-11eb-90fa-3cf95b48411d.png)

power:
    Basically if we are given a power n and a matrix A where we want A^n we can do it a few different ways.  With repeated squaring we break down n into its binary form so if n = 10, lets say that its form is 1010.  By the way these 1s are positioned we want the value of A^8 * A^2 = A^10.  Using this representation there is a much faster way of computing the power as we just need to keep track of A, A^2, A^4, A^8, ..., A^(2^k).  This means that we only need O(log(n)) number of multiplications.  So theoretically we can keep track of these intermediate powers and then construct A^n by multiplying the ones together where there is a 1 in that respective binary spot.  In practice we dont want to have all this space wasted so we can instead keep some sort of running square of A and our current result. 
    So for our toy example we start off with the result as the identity (and do nothing as n%2 = 0) and then we can also calculate A^2.  Now we divide n/2 (shift left) and now we have a 1 in the rightmost binary spot.  This means that we need to use this square, which we just computed above and multiply it with our current result (identity).  But we also have to square again so now we have a result of A^2 and a current square of A^4.  Dividing n by 2 again we see another 0, so we dont touch the result but we square again giving us a result of A^2 and square of A^8.  We divide n by 2 again and this time we see a 1 so we multiply our running result with the square giving us A^10 and our loop can stop.  

    while (pow != 0){
        if (pow & 1){
            mul_matrix(result, temp, result);
        }
        pow = pow >> 1;
        if (!pow){
            break;
        }
        square(temp);
    }

    deallocate_matrix(temp);
    return 0;
}


custom helper function for pow:
  //// pow helper:
  int square(struct matrix *mat){
      struct matrix* temp;
      allocate_matrix(&temp, mat->rows, mat->cols);
      mul_matrix(temp, mat, mat);
      free(mat->data);
      mat->data = temp->data;
  //    deallocate_matrix(temp);
      temp = NULL;
      return 0;
  }
