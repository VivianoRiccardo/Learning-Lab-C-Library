/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files ((the "LICENSE")), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "llab.h"

char* get_full_path(char* directory, char* filename){
    char* temp = (char*)malloc(sizeof(char)*256);// temp = (256)
    temp[0] = '\0';
    strcat(temp,directory);
    strcat(temp,filename);
    return temp;
}

/* This function set the output from a given mask already set
 * 
 * Input:
 *         @ int size:= the size of input and output and mask
 *                                 
 *         @ float* mask:= the vector of the mask (0s and 1s)
 *                            dimensions: size
 *                           
 *         @ float* input:= the vector of the input before the dropout
 *                          dimensions: size
 *         @ float* output:= the vector of the input after the dropout
 *                           dimensions: size
 * */
void get_dropout_array(int size, float* mask, float* input, float* output){
    
    int i;
    for(i = 0; i < size; i++){    
        output[i] = mask[i]*input[i];
    }
}

/* This function set the mask for dropout for a layer
 * 
 * Input:
 *             @ int size:= the size of input,mask
 *             @ float* mask:= the mask that must be set
 *             @ float threshold:= the dropout threshold
 * 
 * */
void set_dropout_mask(int size, float* mask, float threshold){
    
    int i;
    for(i = 0; i < size; i++){
            if(r2() < threshold)
                mask[i] = 0;
    }
}

/* This function add the l2regularization noise to a single weight derivative
 * 
 * Input:
 *             @ float* dw:= the derivative of the weight
 *             @ float w:= the weight
 *             @ float lambda:= an hyperparameter
 *             @ int n:= the number of total weights in the network
 * */
void ridge_regression(float *dw, float w, float lambda, int n){
    (*dw) = (*dw) + (float)((((double)(lambda))/((double)(n)))*((double)(w)));
}

/* Function used to read all the files in a directory
 * Input:
 *             @char** name:= a matrix of char with
 *                            dimensione: n_files*longest_length__of_files
 *             @char* directory:= the name of the directory where the files are
 * */
int read_files(char** name, char* directory){
  DIR           *d;
  struct dirent *dir;
  int count = 0;
  int index = 0;
  char* temp = ".";
  char* temp2 = "..";
  char* temp3 = (char*)malloc(sizeof(char*)*256);
  temp3[0] = '\0';
  strcat(temp3,directory);
  d = opendir(directory);
  if(d == NULL)
    return 1;
  if (d)
  {
    while ((dir = readdir(d)) != NULL)
    {
      if((strcmp(dir->d_name, temp) && strcmp(dir->d_name, temp2))){
          strcat(temp3,dir->d_name);
          strcpy(name[count],temp3);
          temp3[0] = '\0';
          strcat(temp3,directory);
          fprintf(stderr,"%s\n", name[count]);
          count++;
      }
    }

    closedir(d);
  }
  
  free(temp3);
  return(count);
}


/* Function used to convert a number in a string
 * 
 * Input:
 *             @int i:= the number that want to be converted in string
 *             @char b[]:= an array where the string will be stored
 * */
char* itoa(int i, char b[]){
    char const digit[] = "0123456789";
    char* p = b;
    if(i<0){
        *p++ = '-';
        i *= -1;
    }
    int shifter = i;
    do{
        ++p;
        shifter = shifter/10;
    }while(shifter);
    *p = '\0';
    do{
        *--p = digit[i%10];
        i = i/10;
    }while(i);
    return b;
}

/* Function used to shuffle randomly the pointers of the matrix m
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_char_matrix(char** m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          char* t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *                @char** m1:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_char_matrices(char** m,char** m1,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          char* t = m[j];
          char* t1 = m1[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1 and 2 vectors, float and int
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *                @char** m1:= a matrix
 *                         dimensions: n*k
 *               @float* f:= the float vector
 *                @int* v:= the int vector
 *             @int n:= number of pointers char* of m
 * */
int shuffle_char_matrices_float_int_vectors(char** m,char** m1,float* f, int* v,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          char* t = m[j];
          char* t1 = m1[j];
          float t2 = f[j];
          int t3 = v[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
          f[j] = f[i];
          f[i] = t2;
          v[i] = v[i];
          v[i] = t3;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1 and 2 vectors, float and int
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *                @char** m1:= a matrix
 *                         dimensions: n*k
 *               @float* f:= the float vector
 *                @int* v:= the int vector
 *             @int n:= number of pointers char* of m
 * */
int shuffle_float_matrices_float_int_vectors(float** m,float** m1,float* f, int* v,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          float* t = m[j];
          float* t1 = m1[j];
          float t2 = f[j];
          int t3 = v[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
          f[j] = f[i];
          f[i] = t2;
          v[i] = v[i];
          v[i] = t3;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1 and 2 vectors, float and int
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *             @char** m1:= a matrix
 *                         dimensions: n*k
 *             @float* f:= the float vector
 *             @int* v:= the int vector
 *             @int* v2:= the int vector
 *             @int n:= number of pointers char* of m
 * */
int shuffle_char_matrices_float_int_int_vectors(char** m,char** m1,float* f, int* v, int* v2, int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          char* t = m[j];
          char* t1 = m1[j];
          float t2 = f[j];
          int t3 = v[j];
          int t4 = v2[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
          f[j] = f[i];
          f[i] = t2;
          v[i] = v[i];
          v[i] = t3;
          v2[i] = v2[i];
          v2[i] = t4;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1 and 2 vectors, float and int
 * 
 * Input:
 *             @float** m:= a matrix
 *                         dimensions: n*k
 *             @float** m1:= a matrix
 *                         dimensions: n*k
 *             @float* f:= the float vector
 *             @int* v:= the int vector
 *             @int* v2:= the int vector
 *             @int n:= number of pointers char* of m
 * */
int shuffle_float_matrices_float_int_int_vectors(float** m,float** m1,float* f, int* v, int* v2, int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          float* t = m[j];
          float* t1 = m1[j];
          float t2 = f[j];
          int t3 = v[j];
          int t4 = v2[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
          f[j] = f[i];
          f[i] = t2;
          v[i] = v[i];
          v[i] = t3;
          v2[i] = v2[i];
          v2[i] = t4;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the matrix m
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_float_matrix(float** m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          float* t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1
 * 
 * Input:
 *             @float** m:= a matrix
 *                         dimensions: n*k
 *                @float** m1:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_float_matrices(float** m,float** m1,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          float* t = m[j];
          float* t1 = m1[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
        }
    
    }
    return 0;
}


int shuffle_float_matrix_float_tensor(float** m,float*** t,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          float* temp = m[j];
          float** temp2 = t[j];
          m[j] = m[i];
          m[i] = temp;
          t[j] = t[i];
          t[i] = temp2;
        }
    
    }
    return 0;
}
/* Function used to shuffle randomly the pointers of the matrix m
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_int_matrix(int** m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int* t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}

/* Function used to shuffle randomly the pointers of the matrix m
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_int_array(int* m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}


/* Function used to shuffle randomly the pointers of the 2 matrices m and m1
 * 
 * Input:
 *             @int** m:= a matrix
 *                         dimensions: n*k
 *                @int** m1:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_int_matrices(int** m,int** m1,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int* t = m[j];
          int* t1 = m1[j];
          m[j] = m[i];
          m[i] = t;
          m1[j] = m1[i];
          m1[i] = t1;
        }
    
    }
    return 0;
}


/* function to check if number is a nan. If the input is not real then it returns false else returns true*/
int bool_is_real(float d){
    return !(d != d);
}

/* this function read a file and store the content in a char* vector
 * 
 * Input:
 *             @char** ksource:= the space where must be stored the file
 *             @char* fname:= the name of the file
 *             @int* size:= the size of the file that will be filled
 * 
 * */
int read_file_in_char_vector(char** ksource, char* fname, int* size){
    int i;
    FILE *kfile;
    size_t kfilesize;
  
    kfile = fopen(fname, "r" );
    
    
    
    if(kfile == NULL){
        fprintf(stderr,"Error opening file %s\n",fname);
        return 1;
    }
    
    
    
    
    fseek( kfile, 0, SEEK_END );
    kfilesize = ((size_t)ftell(kfile));
    rewind( kfile );
    
    (*ksource) = (char*)malloc(kfilesize*sizeof(char));
    i = fread((*ksource), 1, kfilesize, kfile );
    fclose( kfile );
    (*size) = kfilesize;
    return 0;
}



/* given a float* input array this function copies it in float* output array
 * 
 * Input:
 *             
 *             @ float* input:= the array that must be copied
 *             @ float* output:= the copied array
 *             @ int size:= the dimensions of input and output
 * 
 * */
void copy_array(float* input, float* output, int size){
    memcpy(output,input,(sizeof(float)*size));
}

/* given a int* input array this function copies it in float* output array
 * 
 * Input:
 *             
 *             @ int* input:= the array that must be copied
 *             @ int* output:= the copied array
 *             @ int size:= the dimensions of input and output
 * 
 * */
void copy_int_array(int* input, int* output, int size){
    memcpy(output,input,(sizeof(int)*size));
}

/* given a char* input array this function copies it in char* output array
 * 
 * Input:
 *             
 *             @ char* input:= the array that must be copied
 *             @ char* output:= the copied array
 *             @ int size:= the dimensions of input and output
 * 
 * */
void copy_char_array(char* input, char* output, int size){
    memcpy(output,input,(sizeof(char)*size));
}











/* This function frees a space allocated by a matrix
 * 
 * Input:
 * 
 *             @ void** m:= the matrix m
 *             @ int n:= number of rows of m
 * 
 * */
void free_matrix(float** m, int n){
    int i;
    for(i = 0; i < n; i++){
        if(m[i] != NULL)
            free(m[i]);
    }
    
    free(m);
}

/* This function returns a matrix that can be associated with the confusion matrix
 * where the rows rapresent the model output*2 with real yes and no and the cols rapresent predicted model yes and predicted model no
 * label i:
 * 
 *  label i:         true positive    true negative
 * 
 * 
 * model positive    tp(correct)      fp(incorrect)
 * 
 * 
 * model negative    fn(incorrect)    tn(correct)
 * 
 * Inputs:
 * 
 *                 @ float* model_output:= the output from the model
 *                 @ float* real_output:= the real output from data
 *                 @ long long unsigned int** cm:= a confusion matrix already computed for others output arrays, in this case
 *                                                 the correct/incorrect responses will be summed with the previous computed, dimensions = size*2xsize*2
 *                 @ int size:= the length of model_output and real output arrays
 *                 @ float threshold:= the arbitrary threshold chosen to classify as output 1, for example:
 *                                     if threshold = 0.5 and model_output[i] >= threshold then model_output[i] is classified as 1
*/

long long unsigned int** confusion_matrix(float* model_output, float* real_output, long long unsigned int** cm, int size, float threshold){
    long long unsigned int** conf_mat;
    int i;
    if(cm == NULL){
        conf_mat = (long long unsigned int**)malloc(sizeof(long long unsigned int*)*size*2);
        for(i = 0; i < 2*size; i++){
            conf_mat[i] = (long long unsigned int*)calloc(size*2,sizeof(long long unsigned int));
        }
    }
    
    else
        conf_mat = cm;
    
    for(i = 0; i < size; i++){
        if(real_output[i] >= threshold && model_output[i] >= threshold)
            conf_mat[i*2+1][i*2+1]++;
        else if(real_output[i] < threshold && model_output[i] < threshold)
            conf_mat[i*2][i*2]++;
        else if(real_output[i] >= threshold && model_output[i] < threshold)
            conf_mat[i*2+1][i*2]++;
        else if(real_output[i] < threshold && model_output[i] >= threshold)
            conf_mat[i*2][i*2+1]++;
    }
    
    return conf_mat;
}

/* this function returns an array with the accuracy for each label i
 * 
 * Inputs:
 * 
 *                 @ long long unsigned int** cm:= a confusion matrix, dimensions = size*2xsize*2
 *                 @ int size:= confusion materix dimensions
 * */
double* accuracy_array(long long unsigned int** cm, int size){
    double* accuracy_arr = (double*)calloc(size,sizeof(double));
    int i;
    for(i = 0; i < size; i++){
        accuracy_arr[i] = (double)100*((double)cm[i*2][i*2]+cm[i*2+1][i*2+1])/((double)((cm[i*2][i*2]+cm[i*2+1][i*2+1]+cm[i*2+1][i*2]+cm[i*2][i*2+1])));
        if(accuracy_arr[i]!=accuracy_arr[i])
            accuracy_arr[i] = 0;
        
    }
    
    return accuracy_arr;
}

/* this function returns an array with the precision for each label i
 * 
 * Inputs:
 * 
 *                 @ long long unsigned int** cm:= a confusion matrix, dimensions = size*2xsize*2
 *                 @ int size:= confusion materix dimensions
 * */
double* precision_array(long long unsigned int** cm, int size){
    double* accuracy_arr = (double*)calloc(size,sizeof(double));
    int i;
    for(i = 0; i < size; i++){
        accuracy_arr[i] = (double)100*((double)cm[i*2+1][i*2+1])/((double)((cm[i*2+1][i*2+1]+cm[i*2][i*2+1])));
        if(accuracy_arr[i]!=accuracy_arr[i])
            accuracy_arr[i] = 0;
    }
    
    return accuracy_arr;
}

/* this function returns an array with the sensitivity for each label i
 * 
 * Inputs:
 * 
 *                 @ long long unsigned int** cm:= a confusion matrix, dimensions = size*2xsize*2
 *                 @ int size:= confusion materix dimensions
 * */
double* sensitivity_array(long long unsigned int** cm, int size){
    double* accuracy_arr = (double*)calloc(size,sizeof(double));
    int i;
    for(i = 0; i < size; i++){
        accuracy_arr[i] = (double)100*((double)cm[i*2+1][i*2+1])/((double)((cm[i*2+1][i*2+1]+cm[i*2+1][i*2])));
        if(accuracy_arr[i]!=accuracy_arr[i])
            accuracy_arr[i] = 0;
    }
    
    return accuracy_arr;
}

/* this function returns an array with the specificity for each label i
 * 
 * Inputs:
 * 
 *                 @ long long unsigned int** cm:= a confusion matrix, dimensions = size*2xsize*2
 *                 @ int size:= confusion materix dimensions
 * */
double* specificity_array(long long unsigned int** cm, int size){
    double* accuracy_arr = (double*)calloc(size,sizeof(double));
    int i;
    for(i = 0; i < size; i++){
        accuracy_arr[i] = (double)100*((double)cm[i*2][i*2])/((double)((cm[i*2][i*2]+cm[i*2][i*2+1])));
        if(accuracy_arr[i]!=accuracy_arr[i])
            accuracy_arr[i] = 0;
    }
    
    return accuracy_arr;
}


void print_accuracy(long long unsigned int** cm, int size){
    int i;
    double* aa = accuracy_array(cm,size);
    for(i = 0; i < size; i++){
        printf("%lf    ",aa[i]);
    } 
    printf("\n");
    free(aa);
}

void print_precision(long long unsigned int** cm, int size){
    int i;
    double* aa = precision_array(cm,size);
    for(i = 0; i < size; i++){
        printf("%lf    ",aa[i]);
    } 
    printf("\n");
    free(aa);
}

void print_sensitivity(long long unsigned int** cm, int size){
    int i;
    double* aa = sensitivity_array(cm,size);
    for(i = 0; i < size; i++){
        printf("%lf    ",aa[i]);
    } 
    printf("\n");
    free(aa);
}

void print_specificity(long long unsigned int** cm, int size){
    int i;
    double* aa = specificity_array(cm,size);
    for(i = 0; i < size; i++){
        printf("%lf    ",aa[i]);
    } 
    printf("\n");
    free(aa);
}

/* this function given a float array and a int array of indices from 0 to hi 
 * already sorted will sort the array of indices based on the float a
 * 
 * input:
 * 
 *                 @ float A[]:= the array of values
 *                 @ int I[]:= the array of indices
 *                 @ int lo:= 0
 *                 int hi:= len-1
 * */
void quick_sort(float A[], int I[], int lo, int hi){
    if (lo < hi)
    {
        float pivot = A[I[lo + (hi - lo) / 2]];
        int t;
        int i = lo - 1;
        int j = hi + 1;
        while (1)
        {
            while (A[I[++i]] < pivot);
            while (A[I[--j]] > pivot);
            if (i >= j)
                break;
            t = I[i];
            I[i] = I[j];
            I[j] = t;
        }
        quick_sort(A, I, lo, j);
        quick_sort(A, I, j + 1, hi);
    }
}

char** get_files(int index1, int n_files){
    char** files = (char**)malloc(sizeof(char*)*n_files);
    int i;
    char* temp = ".bin";
    for(i = 0; i < n_files; i++){
        files[i] = (char*)malloc(sizeof(char*)*256);
        files[i][0] = '.';
        files[i][1] = '/';
        files[i][2] = '\0';
        char* b = (char*)malloc(sizeof(char)*256);
        b = itoa((i+index1),b);
        strcat(files[i],b);
        strcat(files[i],temp);
        free(b);
    }
    
    return files;
}

/* this function checks if there is some nan in a matrix, returns 1 if there is at least 1 nan, 0 otherwise
 * 
 * Inputs:
 * 
 *             @ float** m:= the matrix that must be checked
 *             @ int rows:= the rows of matrix that must be checked
 *             @ int cols:= the cols matrix that must be checked
 * */
 int check_nans_matrix(float** m, int rows, int cols){
     int i,j;
     for(i = 0; i < rows; i++){
         for(j = 0; j < cols; j++){
             if(!bool_is_real(m[i][j]))
                return 1;
         }
     }
     return 0;
 }
 
 
void merge(float* values, int* indices, int temp[], int from, int mid, int to, int length){
    int k = from, i = from, j = mid + 1;
 
    while (i <= mid && j <= to){
        if (values[indices[i]] < values[indices[j]]) {
            temp[k++] = indices[i++];
        }
        else {
            temp[k++] = indices[j++];
        }
    }
 
    while (i < length && i <= mid) {
        temp[k++] = indices[i++];
    }
 
    for (i = from; i <= to; i++) {
        indices[i] = temp[i];
    }
}
 
void mergesort(float* values, int* indices, int low, int high){
    int i,m,from,mid,to,length = high-low + 1;
    int* temp = (int*)calloc(length,sizeof(int));
    for(i = 0; i < length; i++){
        temp[i] = indices[i];
    }
    for (m = 1; m <= high - low; m = 2*m){
        for (i = low; i < high; i += 2*m){
            from = i;
            mid = i + m - 1;
            to = min(i + 2*m - 1, high);
            merge(values,indices, temp, from, mid, to,length);
        }
    }
    free(temp);
}


void sort(float* values, int* indices, int low, int high){
    if(high-low > SORT_SWITCH_THRESHOLD)
        mergesort(values,indices,low,high);
    else
        quick_sort(values,indices,low,high);
}


void free_tensor(float*** t, int dim1, int dim2){
    int i,j;
    for(i = 0; i < dim1; i++){
        for(j = 0; j < dim2; j++){
            free(t[i][j]);
        }
        free(t[i]);
    }
    free(t);
}

void free_4D_tensor(float**** t, int dim1, int dim2, int dim3){
    int i,j,k;
    for(i = 0; i < dim1; i++){
        for(j = 0; j < dim2; j++){
            for(k = 0; k < dim3; k++){
                free(t[i][j][k]);
            }
            free(t[i][j]);
        }
        free(t[i]);
    }
    free(t);
}
