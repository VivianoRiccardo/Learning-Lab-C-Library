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

/* this function initializes all the values of an array with a gaussian
 * 
 * Inputs:
 * 
 *         @float* array:= the array that will be filled
 *         @int size:= the size of the array
 * */
void set_array_random_normal(float* array, int size){
    int i;
    for(i = 0; i < size; i++){
        array[i] = random_normal();
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
void ridge_regression(float *dw, float w, float lambda_value, int n){
    (*dw) = (*dw) + (float)((((double)(lambda_value))/((double)(n)))*((double)(w)));
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
char* itoa_n(int i, char b[]){
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

/* Function used to shuffle randomly the pointers of the 2 matrices m and m1
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *                @char** m1:= a matrix
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

/* Function used to shuffle randomly the pointers of the matrix m
 * 
 * Input:
 *             @char** m:= a matrix
 *                         dimensions: n*k
 *             @int n:= number of pointers char* of m
 * */
int shuffle_int_array_until_length(int* m,int n, int length){
    if (n > 1) {
        size_t i;
        for (i = 0; i < length - 1; i++) 
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
/* this function read a file and store the content in a char* vector
 * 
 * Input:
 *             @char** ksource:= the space where must be stored the file
 *             @char* fname:= the name of the file
 *             @int* size:= the size of the file that will be filled
 * 
 * */
uint64_t uint64t_read_file_in_char_vector(char** ksource, char* fname, uint64_t* size){
    int i;
    FILE *kfile;
    uint64_t kfilesize;
  
    kfile = fopen(fname, "r" );
    
    
    
    if(kfile == NULL){
        fprintf(stderr,"Error opening file %s\n",fname);
        return 1;
    }
    
    
    
    
    fseek( kfile, 0, SEEK_END );
    kfilesize = ((uint64_t)ftell(kfile));
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
    if(input == NULL || output == NULL || !size) return;
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
    if(input == NULL || output == NULL || !size) return;
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
    if(input == NULL || output == NULL || !size) return;
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
void free_matrix(void** m, int n){
    if(m == NULL)
        return;
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
void quick_sort_int(int A[], int I[], int lo, int hi){
    if (lo < hi)
    {
        int pivot = A[I[lo + (hi - lo) / 2]];
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
        quick_sort_int(A, I, lo, j);
        quick_sort_int(A, I, j + 1, hi);
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
        b = itoa_n((i+index1),b);
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
 
 
void merge(float* values, int* indices, int temp[], int from_index, int mid, int to, int length){
    int k = from_index, i = from_index, j = mid + 1;
 
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
 
    for (i = from_index; i <= to; i++) {
        indices[i] = temp[i];
    }
}
 
void merge_sort(float* values, int* indices, int low, int high){
    int i,m,from,mid,to,length = high-low + 1;
    int* temp = (int*)calloc(length,sizeof(int));
    for(i = 0; i < length; i++){
        temp[i] = indices[i];
    }
    for (m = 1; m <= high - low; m = 2*m){
        for (i = low; i < high; i += 2*m){
            from = i;
            mid = i + m - 1;
            to = min_int(i + 2*m - 1, high);
            merge(values,indices, temp, from, mid, to,length);
        }
    }
    free(temp);
}


void sort(float* values, int* indices, int low, int high){
    if(high-low > SORT_SWITCH_THRESHOLD)
        merge_sort(values,indices,low,high);
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


void set_vector_with_value(float value, float* v, int dimension){
    int i;
    for(i = 0; i < dimension; i++){
        v[i] = value;
    }
}

void set_int_vector_with_value(int value, int* v, int dimension){
    int i;
    for(i = 0; i < dimension; i++){
        v[i] = value;
    }
}


// this function gives me the file of all the data files. each line in the file is a filename
// that filename contains all the filenames of the data the client must work with
// furthermore each line at the end of the file name has a ; to tell: that filename is currently being used or not
// this function checks this file and return an array of package size length within the first free filename it meets
// then it sets the ; to that filename
char* read_files_from_file(char* file, int package_size){
    // checking files of the subset
    int size = 0,count,i;
    char* ksource;
    read_file_in_char_vector(&ksource,file,&size);// reading the files
    char* temp = (char*)calloc(package_size,sizeof(char));
    char* temp2 = NULL;
    FILE* f = fopen(file,"w");
    for(i = 0,count = 0; i < size; i++){
        if(ksource[i] != '\n' && temp2 == NULL){
            fprintf(f,"%c",ksource[i]);
            temp[count] = ksource[i];
            count++;
        }
        else if (ksource[i] == '\n' && temp2 == NULL){
            if(temp[count-1] != ';'){
                fprintf(f,";");
                temp[count] = '\0';
                temp2 = temp;
            }
            else{
                free(temp);
                temp = (char*)calloc(package_size,sizeof(char));
            }
            fprintf(f,"\n");
            count = 0;
        }
        else{
            fprintf(f,"%c",ksource[i]);
        }
    }
    fclose(f);
    if(temp2 == NULL)
        free(temp);
    free(ksource);
    return temp2;
}

// read above. this function open that file and remove the ; from the line of file_to_free
void set_files_free_from_file(char* file_to_free, char* file){
    // checking files of the subset
    int size = 0,count,i;
    char* ksource;
    read_file_in_char_vector(&ksource,file,&size);// reading the files
    char* temp = (char*)calloc(1024,sizeof(char));
    FILE* f = fopen(file,"w");
    for(i = 0,count = 0; i < size; i++){
        if(ksource[i] != '\n' && ksource[i] != ';'){
            fprintf(f,"%c",ksource[i]);
            temp[count] = ksource[i];
            count++;
        }
        else{
            if(ksource[i] == ';'){
                temp[count] = '\0';
                printf("%s\n%s\n",temp,file_to_free);
                printf("%d\n",strcmp(temp,file_to_free));
                if(strcmp(temp,file_to_free))
                    fprintf(f,";");
                
                
            }
            else
                fprintf(f,"\n");
            free(temp);
            temp = (char*)calloc(1024,sizeof(char));
            count = 0;
        }
    }
    fclose(f);
    free(temp);
    free(ksource);
}

// this function removes all the ; from file
void remove_occupied_sets(char* file){
    // checking files of the subset
    int size = 0,i;
    char* ksource;
    read_file_in_char_vector(&ksource,file,&size);// reading the files
    FILE* f = fopen(file,"w");
                
    for(i = 0; i < size; i++){
        if(ksource[i] != ';')
            fprintf(f,"%c",ksource[i]);
    }
    fclose(f);
    free(ksource);
}


/* msleep(): Sleep for the requested number of milliseconds. */
int msleep(long msec)
{
    struct timespec ts;
    int res;

    if (msec < 0)
    {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

/* get a copy of the input array in a new allocated array
 * */
int* get_new_copy_int_array(int* array, int size){
    if(array == NULL)
        return NULL;
    int* new = (int*)malloc(sizeof(int)*size);
    copy_int_array(array,new,size);
    return new;
}

int argmax(float* vector, int dimension){
    if(vector == NULL || dimension <= 0)
        return -1;
    int i,index = 0;
    float max = vector[0];
    for(i = 0; i < dimension; i++){
        if(vector[i] > max){
            max = vector[i];
            index = i;
        }
    }
    return index;
}

// flaot are the values, indeces are the indices of the values
// the indeces array is sorted in this way the first is the greatest
// reverse_indices[i] tells us where the values[i] is in the array indices
void max_heapify(float* values, uint* indices,uint* reverse_indices,  uint n, uint i) {
  // Find largest among root, left child and right child
  uint largest = i;
  uint left = 2 * i + 1;
  uint right = 2 * i + 2;

  if (left < n && values[indices[left]] >= values[indices[largest]])
    largest = left;

  if (right < n && values[indices[right]] >= values[indices[largest]])
    largest = right;

    // Swap and continue heapifying if root is not largest
    if (largest != i) {
      uint x = indices[i];
      indices[i] = indices[largest];
      reverse_indices[indices[largest]] = i;
      reverse_indices[x] = largest;
      indices[largest] = x;
      max_heapify(values,indices, reverse_indices, n,largest);
  }
}
// flaot are the values, indeces are the indices of the values
// the indeces array is sorted in this way the first is the greatest
// reverse_indices[i] tells us where the values[i] is in the array indices
void max_heapify_up(float* values, uint* indices,uint* reverse_indices,  uint n, uint i) {
  // Find largest among root, left child and right child
  if(i == 0)
    return;
  uint smallest = i;
  uint parent;
  if((i%2))
    parent = (i-1)/2;
  else
    parent = (i-2)/2;
  if (values[indices[parent]] <= values[indices[smallest]])
    smallest = parent;
    
  if (smallest != i) {
      uint x = indices[i];
      indices[i] = indices[smallest];
      reverse_indices[indices[smallest]] = i;
      reverse_indices[x] = smallest;
      indices[smallest] = x;
      max_heapify_up(values,indices, reverse_indices, n,smallest);
  }
}

void remove_ith_element_from_max_heap(float* values, uint* indices,uint* reverse_indices,  uint n, uint i){
    if(i >= n)
        return;
    
    float value1 = values[indices[i]];
    float value2 = values[indices[n-1]];
    reverse_indices[indices[i]] = n-1;
    reverse_indices[indices[n-1]] = i;
    uint x = indices[i];
    indices[i] = indices[n-1];
    indices[n-1] = x;
    if(value2 < value1)
        max_heapify(values,indices, reverse_indices, n-1,i);
    else if(value2>value1)
        max_heapify_up(values,indices, reverse_indices, n-1,i);
}

void update_recursive_cumulative_heap_up(float* values, uint index, uint started_index, uint n, float value){
    uint parent;
    if(index){
        if(!(index%2))
            parent = (index-2)/2;
        else
            parent = (index-1)/2;
        if(!started_index){
            values[index]+=value;        
        }
    }
    else{
        if(!started_index){
            values[index]+=value;        
        }
        return;
    }
    update_recursive_cumulative_heap_up(values,parent,0,n,value);
}

int index_is_inside_buffer(uint* buffer, uint length, uint index){
    uint i;
    for(i = 0; i < length; i++){
        if(buffer[i] == index){
            return 1;
        }
    }
    return 0;
}

int value_is_child(uint child, uint parent){
    if(child <= parent)
        return 0;
    do{
        if(child%2)
            child = (child-1)/2;
        else
            child = (child-2)/2;
    }while(child > parent);
    if(child == parent)
        return 1;
    return 0;
}

float subtracted_value(uint index, float* current_values, uint* taken_values, uint taken_values_length){
    float ret = 0;
    uint i;
    for(i = 0; i < taken_values_length; i++){
        if(!index_is_inside_buffer(taken_values,i,taken_values[i])){
            if(value_is_child(taken_values[i],index))
                ret+=current_values[taken_values[i]];
        }
    }
    return ret;
}

float subtracted_value_rewards(uint index, int* current_values, uint* taken_values, uint taken_values_length,float alpha){
    float ret = 0;
    uint i;
    for(i = 0; i < taken_values_length; i++){
        if(!index_is_inside_buffer(taken_values,i,taken_values[i]) && current_values[taken_values[i]] > 0){
            if(value_is_child(taken_values[i],index))
                ret+=pow(((double)1/((double)current_values[taken_values[i]])),alpha);
            //printf("i, ret: %d, %f\n",taken_values[i],ret);
        }
    }
    return ret;
}

// log(n) sample
uint weighted_random_sample(float* cumulative_values, float* current_values, uint index, uint size, float random_value, double sum, uint* taken_values, uint taken_values_length){
    if(index >= size){
        //printf("M ");
        return index-1;
     }
    float v = 0, v_left = 0;
    if(!index_is_inside_buffer(taken_values,taken_values_length,index)){
        v = current_values[index]/sum;
        if(random_value <= v){
            //printf("K: %d, %d. ", index, index_is_inside_buffer(taken_values,taken_values_length,index));
            return index;
        }
    }
        
    uint left = index*2+1;
    uint right = index*2+2;
    if(left >= size){
        //printf("L ");
        uint i;
        for(i = index+1; i < size; i++){
            if(!index_is_inside_buffer(taken_values,taken_values_length,i))
                return i;
        }
        return index;
    }
    
    if(right >= size){
        uint i;
        //printf("R");
        if(!index_is_inside_buffer(taken_values,taken_values_length,left))
            return left;
        for(i = index+1; i < size; i++){
            if(!index_is_inside_buffer(taken_values,taken_values_length,i))
                return i;
        }
        return left;
    }
    if(!index_is_inside_buffer(taken_values,taken_values_length,left)){
        v_left = current_values[left];
    }
    random_value-=v;
    float sub = subtracted_value(left,current_values,taken_values,taken_values_length);
    if(random_value <= ((v_left+cumulative_values[left]-sub)/sum))
        return weighted_random_sample(cumulative_values, current_values, left, size, random_value, sum,taken_values,taken_values_length);
    else
        return weighted_random_sample(cumulative_values, current_values, right, size, random_value-((v_left+cumulative_values[left]-sub)/sum), sum,taken_values,taken_values_length);

    
}
uint weighted_random_sample_rewards(float* cumulative_values, int* current_values, uint index, uint size, float random_value, double sum, uint* taken_values, uint taken_values_length, float alpha){
    if(index >= size){
        //printf("M ");
        return index-1;
     }
    float v = 0;
    float v_left = 0;
    if(current_values[index] > 0 && !index_is_inside_buffer(taken_values,taken_values_length,index)){
        v = pow(((double)1/((double)current_values[index])),alpha)/sum;
        if(random_value <= v){
            //printf("random, v, sum, index: %f, %f, %lf, %d\n",random_value,v,sum, index);
            //printf("K: %d, %d. ", index, index_is_inside_buffer(taken_values,taken_values_length,index));
            return index;
        }
    } 
    //printf("random, v, sum, index: %f, %f, %lf, %d\n",random_value,v,sum, index);
    uint left = index*2+1;
    uint right = index*2+2;
    random_value-=v;
    if(left >= size){
        if(current_values[index]>0)
            return index;
        else
            return size;
    }
    
    if(right >= size){
        uint i;
        if(current_values[left] > 0)
            return left;
        else
            return size;
            
    }
    if(current_values[left] > 0 && !index_is_inside_buffer(taken_values,taken_values_length,left)){
        v_left = pow(((double)1/((double)current_values[left])),alpha);
    } 
    
    float sub = subtracted_value_rewards(left,current_values,taken_values,taken_values_length,alpha);
    //printf("sub, rv: %f %f\n",sub,random_value);
    uint returned_index = size;
    if(random_value <= ((v_left+cumulative_values[left]-sub)/sum)){
        returned_index = weighted_random_sample_rewards(cumulative_values, current_values, left, size, random_value, sum,taken_values,taken_values_length,alpha);
        if(returned_index == size)
            returned_index = weighted_random_sample_rewards(cumulative_values, current_values, right, size, random_value-((v_left+cumulative_values[left]-sub)/sum), sum,taken_values,taken_values_length,alpha);
        if(current_values[index] > 0 && returned_index == size)
            return index;
        return returned_index;
    }
    else{
        returned_index = weighted_random_sample_rewards(cumulative_values, current_values, right, size, random_value-((v_left+cumulative_values[left]-sub)/sum), sum,taken_values,taken_values_length,alpha);
        if(returned_index == size)
            returned_index = weighted_random_sample_rewards(cumulative_values, current_values, left, size, random_value, sum,taken_values,taken_values_length,alpha);
        if(current_values[index] > 0 && returned_index == size)
            return index;
        return returned_index;
    }
    
}


int is_little_endian(){
    unsigned int x = 1;
    return ((int) (((char *)&x)[0]) == 1);
}


void reverse_ptr(void* ptr, uint64_t size){
    if(size <= 1)
        return;
   char* array = (char*) ptr;
   uint64_t i, len = size/2;
   for(i = 0; i < len; i++){
       char temp = array[i];
       array[i] = array[size-1-i];
       array[size-1-i] = temp;
   }
   return;
}


void swap_array_bytes_order(void* ptr, uint64_t size, uint64_t len){
    if(size <= 1 || !len)
        return;
    char* array = (char*) &ptr;
    uint64_t i;
    for(i = 0; i < len; i++){
        reverse_ptr(array + i*size,size);
    }
}

void convert_data(void* ptr, uint64_t size, uint64_t len){
    if(!is_little_endian())
        swap_array_bytes_order(ptr,size,len);
}

void convert_communication_data(void* ptr, uint64_t size, uint64_t len){
    if(!is_little_endian())
        swap_array_bytes_order(ptr,size,len);
}


void merge_for_probabilities(float* p, int* index, int left, int middle, int right, int n) {
    int i, j, k;
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // Copia i dati negli array temporanei
    float L[n1], R[n2];
    int L_index[n1], R_index[n2];
    for (i = 0; i < n1; i++) {
        L[i] = p[(left + i) * n + middle + i];
        L_index[i] = index[left + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = p[(middle + 1 + j) * n + right + j];
        R_index[j] = index[middle + 1 + j];
    }

    // Unione delle due metà ordinate
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i] < R[j]) {
            p[k * n + k + middle + j + 1] = L[i];
            index[k] = L_index[i];
            i++;
        } else {
            p[(k + j + 1) * n + right] = R[j];
            index[k] = R_index[j];
            j++;
        }
        k++;
    }

    // Copia gli elementi rimanenti della metà sinistra
    while (i < n1) {
        p[k * n + k + middle + j + 1] = L[i];
        index[k] = L_index[i];
        i++;
        k++;
    }

    // Copia gli elementi rimanenti della metà destra
    while (j < n2) {
        p[(k + j + 1) * n + right] = R[j];
        index[k] = R_index[j];
        j++;
        k++;
    }
}

void merge_sort_for_probabilities(float* p, int* index, int left, int right, int n) {
    if (left < right) {
        int middle = left + (right - left) / 2;
        merge_sort_for_probabilities(p, index, left, middle, n);
        merge_sort_for_probabilities(p, index, middle + 1, right, n);
        merge_for_probabilities(p, index, left, middle, right, n);
    }
}

void readapt_scores(int** indices, int* index, int size_1, int size_2, dueling_categorical_dqn* dqn){
    int i,j;
    float* means = (float*)calloc(size_2,sizeof(float));
    float* std = (float*)calloc(size_2,sizeof(float));
    uint64_t last_sum, sum = 0;
    // go with means
    for(i = 0; i < size_2; i++){
        for(j = 0; j < size_1; j++){
            means[i]+=((float)indices[j][i]);
        }
    }
    
    do{
        last_sum = sum;
        sum=get_array_size_scores_index_dueling_categorical_dqn(dqn, sum);
        for(i = last_sum; i < sum; i++){
            means[i]/=(double)(sum-last_sum);
        }
    }while(last_sum != sum);
    
    // go with std
    for(i = 0; i < size_2; i++){
        for(j = 0; j < size_1; j++){
            float temp = (((float)indices[j][i])-means[i]);
            temp*=temp;
            std[i]+=temp;
        }
    }
    sum = 0;
    do{
        last_sum = sum;
        sum=get_array_size_scores_index_dueling_categorical_dqn(dqn, sum);
        for(i = last_sum; i < sum; i++){
            index[i] = i-last_sum;
            std[i]/=(double)(sum-last_sum);
        }
        if(last_sum != sum){
            get_sorted_probability_vector(means+last_sum, std+last_sum, sum-last_sum, index+last_sum);
        }
    }while(last_sum != sum);
    
    
    
    free(means);
    free(std);
}

void readapt_scores_float(float** indices, int* index, int size_1, int size_2, dueling_categorical_dqn* dqn){
    int i,j;
    float* means = (float*)calloc(size_2,sizeof(float));
    float* std = (float*)calloc(size_2,sizeof(float));
    uint64_t last_sum, sum = 0;
    // go with means
    for(i = 0; i < size_2; i++){
        for(j = 0; j < size_1; j++){
            means[i]+=((float)indices[j][i]);
        }
    }
    
    do{
        last_sum = sum;
        sum=get_array_size_scores_index_dueling_categorical_dqn(dqn, sum);
        for(i = last_sum; i < sum; i++){
            means[i]/=(double)(sum-last_sum);
        }
    }while(last_sum != sum);
        
    // go with std
    for(i = 0; i < size_2; i++){
        for(j = 0; j < size_1; j++){
            float temp = (((float)indices[j][i])-means[i]);
            temp*=temp;
            std[i]+=temp;
        }
    }
    sum = 0;
    do{
        last_sum = sum;
        sum=get_array_size_scores_index_dueling_categorical_dqn(dqn, sum);
        for(i = last_sum; i < sum; i++){
            index[i] = i-last_sum;
            std[i]/=(double)(sum-last_sum);
        }
        if(last_sum != sum){
            get_sorted_probability_vector(means+last_sum, std+last_sum, sum-last_sum, index+last_sum);
        }
    }while(last_sum != sum);
    
    
    
    free(means);
    free(std);
}

// increasing order
void set_float_vector_to_int_vector(float* v1, int* v2, int size, float step){
    int i;
    for(i = 0; i < size; i++){
        v1[v2[i]] = (float)((i)*step);
    }
}

void scores_and_indices_readapter(int** indices, float** scores, int size_1, int size_2, float step, dueling_categorical_dqn* dqn){
    readapt_scores(indices, indices[0], size_1, size_2, dqn);
    uint64_t last_sum, sum = 0;
    do{
        last_sum = sum;
        sum=get_array_size_scores_index_dueling_categorical_dqn(dqn, sum);
        set_float_vector_to_int_vector(scores[0]+last_sum, indices[0]+last_sum, sum-last_sum,step);
    }while(last_sum != sum);
    int i;
    for(i = 1; i < size_1; i++){
        copy_array(scores[0], scores[i],size_2);
        copy_int_array(indices[0], indices[i],size_2);
    }
    return;


}

void scores_and_indices_readapter_float(int** indices, float** scores, int size_1, int size_2, float step, dueling_categorical_dqn* dqn){
    readapt_scores_float(scores, indices[0], size_1, size_2, dqn);
    uint64_t last_sum, sum = 0;
    do{
        last_sum = sum;
        sum=get_array_size_scores_index_dueling_categorical_dqn(dqn, sum);
        set_float_vector_to_int_vector(scores[0]+last_sum, indices[0]+last_sum, sum-last_sum,step);
    }while(last_sum != sum);
    int i;
    for(i = 1; i < size_1; i++){
        copy_array(scores[0], scores[i],size_2);
        copy_int_array(indices[0], indices[i],size_2);
    }
    return;


}


