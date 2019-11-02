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
/*random number between 0 and 1*/
float r2(){
    return (float)rand() / (float)RAND_MAX ;
}

float drand (){
  return (rand () + 1.0) / (RAND_MAX + 1.0);
}

/* a random number from a gaussian distribution with mean 0 and std 1*/
float random_normal (){
  return sqrtf(-2 * log (drand ())) * cos (2 * M_PI * drand ());
  }

/* a random number from a gaussian distribution with mean 0 and std = sqrtf(2/n)
 * where n is the number of neuron of layer l-1*/
float random_general_gaussian(float mean, float n){
    return mean + sqrtf(2/(n))*random_normal();
}



/* a random number from a gaussian distribution with mean 0 and std = sqrtf(1/n)
 * where n is the number of neuron of layer l-1*/
float random_general_gaussian_xavier_init(float mean, float n){
    return mean + sqrtf(1/(n))*random_normal();
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
    (*dw) = (*dw) + (lambda/(float)n)*w;
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
  d = opendir(directory);
  if(d == NULL)
    return 1;
  if (d)
  {
    while ((dir = readdir(d)) != NULL)
    {
      if((strcmp(dir->d_name, temp) && strcmp(dir->d_name, temp2))){
          strcpy(name[count],dir->d_name);
          fprintf(stderr,"%s\n", name[count]);
          count++;
      }
    }

    closedir(d);
  }

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


/* This function computes the dot product between 2 array, input and input2
 * with the same length, and store the result in the output array
 * 
 * Input:
 * 
 *             @ float* input1:= the first input array
 *             @ float* input2:= the second input array
 *             @ float* output:= the output array
 *             @ int size:= the size of input1, input2, input3
 * */
void dot1D(float* input1, float* input2, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]*input2[i];
    }
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
    int i;
    for(i = 0; i < size; i++){
        output[i] = input[i];
    }
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
    int i;
    for(i = 0; i < size; i++){
        output[i] = input[i];
    }
}

/* This function computes the sum between 2 array, input and input2
 * with the same length, and store the result in the output array
 * 
 * Input:
 * 
 *             @ float* input1:= the first input array
 *             @ float* input2:= the second input array
 *             @ float* output:= the output array
 *             @ int size:= the size of input1, input2, input3
 * */
void sum1D(float* input1, float* input2, float* output, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]+input2[i];
    }
}

/* This function computes a dot product between an array and a float value: value
 * 
 * Input
 * 
 *             @ float* input:= the imput used to compute the output
 *             @ float value:= the float value that must be multiplied for the inputs
 *             @ float* output:= the array where you need to store the output
 *             @ int dimension:= the dimension of input and output
 * 
 * */
void mul_value(float* input, float value, float* output, int dimension){
    int i;
    for(i = 0; i < dimension; i++){
        output[i] = input[i]*value;
    }
}


/* Given a model, this function update the params of the residual layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                nesterov_momentum(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w]);
                            }
                        }
                    }
                    nesterov_momentum(&m->rls[i]->cls[j]->biases[k],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_biases[k],&m->rls[i]->cls[j]->d1_biases[k]);
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                        bm->n_bn = m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels;
                        bm->bns = m->rls[i]->cls[j]->group_norm;
                        update_batch_normalized_layer_nesterov_bmodel(bm,lr,momentum,mini_batch_size);
                        free(bm);
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the residual layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_residual_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                nesterov_momentum(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w]);
                            }
                        }
                    }
                    nesterov_momentum(&m->rls[i]->cls[j]->biases[k],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_biases[k],&m->rls[i]->cls[j]->d1_biases[k]);
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                        bm->n_bn = m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels;
                        bm->bns = m->rls[i]->cls[j]->group_norm;
                        update_batch_normalized_layer_nesterov_bmodel(bm,lr,momentum,mini_batch_size);
                        free(bm);
                    }
                }
            }
        }
    }
}



/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                adam_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                            }
                        }
                    }
                    adam_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                        bm->n_bn = m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels;
                        bm->bns = m->rls[i]->cls[j]->group_norm;
                        update_batch_normalized_layer_adam_bmodel(bm,lr,mini_batch_size,b1,b2);
                        free(bm);
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * 
 * */
void update_residual_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long t){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                radam_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                            }
                        }
                    }
                    radam_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                        bm->n_bn = m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels;
                        bm->bns = m->rls[i]->cls[j]->group_norm;
                        update_batch_normalized_layer_radam_bmodel(bm,lr,mini_batch_size,b1,b2,t);
                        free(bm);
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_residual_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                adam_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                            }
                        }
                    }
                    adam_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                        bm->n_bn = m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels;
                        bm->bns = m->rls[i]->cls[j]->group_norm;
                        update_batch_normalized_layer_adam_bmodel(bm,lr,mini_batch_size,b1,b2);
                        free(bm);
                    }
                }
            }
        }
    }
}


/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_residual_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                radam_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                            }
                        }
                    }
                    radam_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                        bm->n_bn = m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels;
                        bm->bns = m->rls[i]->cls[j]->group_norm;
                        update_batch_normalized_layer_radam_bmodel(bm,lr,mini_batch_size,b1,b2,t);
                        free(bm);
                    }
                }
            }
        }
    }
}

/* This function sum the partial derivatives of the residual layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ model* m:= the input model
 *             @ model* m2:= another input model
 *             @ model* m3:= the output model
 * 
 * */
void sum_residual_layers_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    sum1D(m->rls[i]->cls[j]->d_kernels[k],m2->rls[i]->cls[j]->d_kernels[k],m3->rls[i]->cls[j]->d_kernels[k],m3->rls[i]->cls[j]->channels*m3->rls[i]->cls[j]->kernel_rows*m3->rls[i]->cls[j]->kernel_cols);
                }
                
                sum1D(m->rls[i]->cls[j]->d_biases,m2->rls[i]->cls[j]->d_biases,m3->rls[i]->cls[j]->d_biases,m3->rls[i]->cls[j]->n_kernels);

                if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels; k++){
                        sum1D(m->rls[i]->cls[j]->group_norm[k]->d_beta,m2->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->vector_dim);
                        sum1D(m->rls[i]->cls[j]->group_norm[k]->d_beta,m2->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->vector_dim);
                    }
                }
            }
        }
    }
}

/* This function sum the partial derivatives of the residual layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ bmodel* m:= the input model
 *             @ bmodel* m2:= another input model
 *             @ bmodel* m3:= the output model
 * 
 * */
void sum_residual_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    sum1D(m->rls[i]->cls[j]->d_kernels[k],m2->rls[i]->cls[j]->d_kernels[k],m3->rls[i]->cls[j]->d_kernels[k],m3->rls[i]->cls[j]->channels*m3->rls[i]->cls[j]->kernel_rows*m3->rls[i]->cls[j]->kernel_cols);
                }
                
                sum1D(m->rls[i]->cls[j]->d_biases,m2->rls[i]->cls[j]->d_biases,m3->rls[i]->cls[j]->d_biases,m3->rls[i]->cls[j]->n_kernels);

                if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels; k++){
                        sum1D(m->rls[i]->cls[j]->group_norm[k]->d_beta,m2->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->vector_dim);
                        sum1D(m->rls[i]->cls[j]->group_norm[k]->d_beta,m2->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->d_beta,m3->rls[i]->cls[j]->group_norm[k]->vector_dim);
                    }
                }
            }
        }
    }
}
/* Given a model, this function update the params of the convolutional layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_convolutional_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            nesterov_momentum(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr,momentum,mini_batch_size, m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w]);
                        }
                            
                    }
                }
                nesterov_momentum(&m->cls[j]->biases[k],lr,momentum,mini_batch_size, m->cls[j]->d_biases[k],&m->cls[j]->d1_biases[k]);
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                    bm->n_bn = m->cls[j]->n_kernels/m->cls[j]->group_norm_channels;
                    bm->bns = m->cls[j]->group_norm;
                    update_batch_normalized_layer_nesterov_bmodel(bm,lr,momentum,mini_batch_size);
                    free(bm);
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_convolutional_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            nesterov_momentum(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr,momentum,mini_batch_size, m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w]);
                        }
                            
                    }
                }
                nesterov_momentum(&m->cls[j]->biases[k],lr,momentum,mini_batch_size, m->cls[j]->d_biases[k],&m->cls[j]->d1_biases[k]);
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                    bm->n_bn = m->cls[j]->n_kernels/m->cls[j]->group_norm_channels;
                    bm->bns = m->cls[j]->group_norm;
                    update_batch_normalized_layer_nesterov_bmodel(bm,lr,momentum,mini_batch_size);
                    free(bm);
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_convolutional_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            adam_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                        }
                            
                    }
                }
                adam_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                    bm->n_bn = m->cls[j]->n_kernels/m->cls[j]->group_norm_channels;
                    bm->bns = m->cls[j]->group_norm;
                    update_batch_normalized_layer_adam_bmodel(bm,lr,mini_batch_size,b1,b2);
                    free(bm);
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                   @ unsigned long long int t:= the number of time radam has been used
 * */
void update_convolutional_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            radam_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                        }
                            
                    }
                }
                radam_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                    bm->n_bn = m->cls[j]->n_kernels/m->cls[j]->group_norm_channels;
                    bm->bns = m->cls[j]->group_norm;
                    update_batch_normalized_layer_radam_bmodel(bm,lr,mini_batch_size,b1,b2,t);
                    free(bm);
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_convolutional_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            adam_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                        }
                            
                    }
                }
                adam_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                    bm->n_bn = m->cls[j]->n_kernels/m->cls[j]->group_norm_channels;
                    bm->bns = m->cls[j]->group_norm;
                    update_batch_normalized_layer_adam_bmodel(bm,lr,mini_batch_size,b1,b2);
                    free(bm);
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                   @ unsigned long long int t:= the number of time radam has been used
 * * */
void update_convolutional_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            radam_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                        }
                            
                    }
                }
                radam_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    bmodel* bm = (bmodel*)malloc(sizeof(bmodel));
                    bm->n_bn = m->cls[j]->n_kernels/m->cls[j]->group_norm_channels;
                    bm->bns = m->cls[j]->group_norm;
                    update_batch_normalized_layer_radam_bmodel(bm,lr,mini_batch_size,b1,b2,t);
                    free(bm);
                }
            }
        }
    }
}

/* This function sum the partial derivatives of the convolutional layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ model* m:= the input model
 *             @ model* m2:= another input model
 *             @ model* m3:= the output model
 * 
 * */
void sum_convolutional_layers_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                sum1D(m->cls[j]->d_kernels[k],m2->cls[j]->d_kernels[k],m3->cls[j]->d_kernels[k],m3->cls[j]->channels*m3->cls[j]->kernel_rows*m3->cls[j]->kernel_cols);
            }
            
            sum1D(m->cls[j]->d_biases,m2->cls[j]->d_biases,m3->cls[j]->d_biases,m3->cls[j]->n_kernels);

            if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                for(k = 0; k < m->cls[j]->n_kernels/m->cls[j]->group_norm_channels; k++){
                    sum1D(m->cls[j]->group_norm[k]->d_beta,m2->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->vector_dim);
                    sum1D(m->cls[j]->group_norm[k]->d_beta,m2->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->vector_dim);
                }
            }
        }
    }

}

/* This function sum the partial derivatives of the convolutional layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ bmodel* m:= the input model
 *             @ bmodel* m2:= another input model
 *             @ bmodel* m3:= the output model
 * 
 * */
void sum_convolutional_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                sum1D(m->cls[j]->d_kernels[k],m2->cls[j]->d_kernels[k],m3->cls[j]->d_kernels[k],m3->cls[j]->channels*m3->cls[j]->kernel_rows*m3->cls[j]->kernel_cols);
            }
            
            sum1D(m->cls[j]->d_biases,m2->cls[j]->d_biases,m3->cls[j]->d_biases,m3->cls[j]->n_kernels);

            if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                for(k = 0; k < m->cls[j]->n_kernels/m->cls[j]->group_norm_channels; k++){
                    sum1D(m->cls[j]->group_norm[k]->d_beta,m2->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->vector_dim);
                    sum1D(m->cls[j]->group_norm[k]->d_beta,m2->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->d_beta,m3->cls[j]->group_norm[k]->vector_dim);
                }
            }
        }
    }

}
/* Given a model, this function update the params of the fully-connected layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_fully_connected_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                nesterov_momentum(&m->fcls[i]->weights[j*m->fcls[i]->input+k], lr, momentum, mini_batch_size, m->fcls[i]->d_weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k]);
            }
            nesterov_momentum(&m->fcls[i]->biases[j], lr, momentum, mini_batch_size, m->fcls[i]->d_biases[j],&m->fcls[i]->d1_biases[j]);
        }
    }
}

/* Given a model, this function update the params of the fully-connected layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_fully_connected_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                nesterov_momentum(&m->fcls[i]->weights[j*m->fcls[i]->input+k], lr, momentum, mini_batch_size, m->fcls[i]->d_weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k]);
            }
            nesterov_momentum(&m->fcls[i]->biases[j], lr, momentum, mini_batch_size, m->fcls[i]->d_biases[j],&m->fcls[i]->d1_biases[j]);
        }
    }
}

/* Given a bmodel, this function update the params of the batch-normalized layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_batch_normalized_layer_nesterov_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < m->n_bn; i++){
        for(j = 0; j < m->bns[i]->vector_dim; j++){
            nesterov_momentum(&m->bns[i]->gamma[j], lr, momentum, 1, m->bns[i]->d_gamma[j],&m->bns[i]->d1_gamma[j]);
            nesterov_momentum(&m->bns[i]->beta[j], lr, momentum, 1, m->bns[i]->d_beta[j],&m->bns[i]->d1_beta[j]);
        }
    }
}


/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_fully_connected_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                adam_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
            }
            adam_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
        }
    }
}

/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_fully_connected_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                radam_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
            }
            radam_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
        }
    }
}

/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_fully_connected_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                adam_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
            }
            adam_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
        }
    }
}


/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_fully_connected_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2,unsigned long long int t){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                radam_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
            }
            radam_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
        }
    }
}

/* Given a bmodel, this function update the params of the batch-normalized layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_batch_normalized_layer_adam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2){
    int i,j,k;
    for(i = 0; i < m->n_bn; i++){
        for(j = 0; j < m->bns[i]->vector_dim; j++){
            adam_algorithm(&m->bns[i]->gamma[j],&m->bns[i]->d1_gamma[j], &m->bns[i]->d2_gamma[j], m->bns[i]->d_gamma[j], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,1);
            adam_algorithm(&m->bns[i]->beta[j],&m->bns[i]->d1_beta[j], &m->bns[i]->d2_beta[j], m->bns[i]->d_beta[j], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,1);  
        }
    }
}

/* Given a bmodel, this function update the params of the batch-normalized layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_batch_normalized_layer_radam_bmodel(bmodel* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t){
    int i,j,k;
    for(i = 0; i < m->n_bn; i++){
        for(j = 0; j < m->bns[i]->vector_dim; j++){
            radam_algorithm(&m->bns[i]->gamma[j],&m->bns[i]->d1_gamma[j], &m->bns[i]->d2_gamma[j], m->bns[i]->d_gamma[j], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
            radam_algorithm(&m->bns[i]->beta[j],&m->bns[i]->d1_beta[j], &m->bns[i]->d2_beta[j], m->bns[i]->d_beta[j], lr, BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);  
        }
    }
}



/* This function sum the partial derivatives of the fully-connected layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ model* m:= the input model
 *             @ model* m2:= another input model
 *             @ model* m3:= the output model
 * 
 * */
void sum_fully_connected_layers_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        sum1D(m->fcls[i]->d_weights,m2->fcls[i]->d_weights,m3->fcls[i]->d_weights,m->fcls[i]->input*m->fcls[i]->output);    
        sum1D(m->fcls[i]->d_biases,m2->fcls[i]->d_biases,m3->fcls[i]->d_biases,m->fcls[i]->output);    
    }
    
        
}

/* This function sum the partial derivatives of the fully-connected layers of a model m and a second model m2 in a third model m3
 * 
 * Input:
 * 
 *             @ bmodel* m:= the input model
 *             @ bmodel* m2:= another input model
 *             @ bmodel* m3:= the output model
 * 
 * */
void sum_fully_connected_layers_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        sum1D(m->fcls[i]->d_weights,m2->fcls[i]->d_weights,m3->fcls[i]->d_weights,m->fcls[i]->input*m->fcls[i]->output);    
        sum1D(m->fcls[i]->d_biases,m2->fcls[i]->d_biases,m3->fcls[i]->d_biases,m->fcls[i]->output);    
    }
    
        
}



/* This function add the l2 regularization to the partial derivative of the weights for residual layers of m
 * 
 * 
 * Input:
 *         
 *             @ model* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_residual_layer(model* m,int total_number_weights,float lambda){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                ridge_regression(&m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lambda, total_number_weights);
                            }
                        }
                    }
                }
                if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels; k++){
                        for(u = 0; u < m->rls[i]->cls[j]->group_norm[k]->vector_dim; u++){
                            ridge_regression(&(m->rls[i]->cls[j]->group_norm[k]->d_gamma[u]),m->rls[i]->cls[j]->group_norm[k]->gamma[u],lambda,total_number_weights);
                        }
                    }
                }
            }
        }
    }
}

/* This function add the l2 regularization to the partial derivative of the weights for residual layers of m
 * 
 * 
 * Input:
 *         
 *             @ bmodel* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_residual_layer_bmodel(bmodel* m,int total_number_weights,float lambda){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION){
                for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                        for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                ridge_regression(&m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lambda, total_number_weights);
                            }
                        }
                    }
                }
                if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels; k++){
                        for(u = 0; u < m->rls[i]->cls[j]->group_norm[k]->vector_dim; u++){
                            ridge_regression(&(m->rls[i]->cls[j]->group_norm[k]->d_gamma[u]),m->rls[i]->cls[j]->group_norm[k]->gamma[u],lambda,total_number_weights);
                        }
                    }
                }
            }
        }
    }
}


/* This function add the l2 regularization to the partial derivative of the weights for convolutional layers of m
 * 
 * 
 * Input:
 *         
 *             @ model* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_convolutional_layer(model* m,int total_number_weights,float lambda){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            ridge_regression(&m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lambda, total_number_weights);

                        }
                            
                    }
                }
            }
            if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                for(k = 0; k < m->cls[j]->n_kernels/m->cls[j]->group_norm_channels; k++){
                    for(u = 0; u < m->cls[j]->group_norm[k]->vector_dim; u++){
                        ridge_regression(&(m->cls[j]->group_norm[k]->d_gamma[u]),m->cls[j]->group_norm[k]->gamma[u],lambda,total_number_weights);
                    }
                }
            }
        }
    }
}

/* This function add the l2 regularization to the partial derivative of the weights for convolutional layers of m
 * 
 * 
 * Input:
 *         
 *             @ bmodel* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_convolutional_layer_bmodel(bmodel* m,int total_number_weights,float lambda){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION){
            for(k = 0; k < m->cls[j]->n_kernels; k++){
                for(u = 0; u < m->cls[j]->channels; u++){
                    for(z = 0; z < m->cls[j]->kernel_rows; z++){
                        for(w = 0; w < m->cls[j]->kernel_cols; w++){
                            ridge_regression(&m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lambda, total_number_weights);

                        }
                            
                    }
                }
            }
            if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                for(k = 0; k < m->cls[j]->n_kernels/m->cls[j]->group_norm_channels; k++){
                    for(u = 0; u < m->cls[j]->group_norm[k]->vector_dim; u++){
                        ridge_regression(&(m->cls[j]->group_norm[k]->d_gamma[u]),m->cls[j]->group_norm[k]->gamma[u],lambda,total_number_weights);
                    }
                }
            }
        }
    }
}


/* This function add the l2 regularization to the partial derivative of the weights for fully-connected layers of m
 * 
 * 
 * Input:
 *         
 *             @ model* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_fully_connected_layer(model* m,int total_number_weights,float lambda){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                ridge_regression(&m->fcls[i]->d_weights[j*m->fcls[i]->input+k],m->fcls[i]->weights[j*m->fcls[i]->input+k],lambda, total_number_weights);

            }
        }
    }
}

/* This function add the l2 regularization to the partial derivative of the weights for fully-connected layers of m
 * 
 * 
 * Input:
 *         
 *             @ bmodel* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_fully_connected_layer_bmodel(bmodel* m,int total_number_weights,float lambda){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        for(j = 0; j < m->fcls[i]->output; j++){
            for(k = 0; k < m->fcls[i]->input; k++){
                ridge_regression(&m->fcls[i]->d_weights[j*m->fcls[i]->input+k],m->fcls[i]->weights[j*m->fcls[i]->input+k],lambda, total_number_weights);

            }
        }
    }
}

/* This function add the l2 regularization to the partial derivative of the weights for lstm layers of m
 * 
 * 
 * Input:
 *         
 *             @ rmodel* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_lstm_layer(rmodel* m,int total_number_weights,float lambda){
    int j,k,u,z,w;
    for(j = 0; j < m->n_lstm; j++){
        for(k = 0; k < 4; k++){
            for(u = 0; u < m->lstms[j]->size*m->lstms[j]->size; u++){
                ridge_regression(&m->lstms[j]->d_w[k][u],m->lstms[j]->w[k][u],lambda,total_number_weights);
                ridge_regression(&m->lstms[j]->d_u[k][u],m->lstms[j]->u[k][u],lambda,total_number_weights);
            }
        }
    }
}


/* Given a rmodel, this function update the params of the lstm layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 * 
 * */
void update_lstm_layer_nesterov(rmodel* m, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->size*m->lstms[i]->size; k++){
                nesterov_momentum(&m->lstms[i]->w[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_w[j][k],&m->lstms[i]->d1_w[j][k]);
                nesterov_momentum(&m->lstms[i]->u[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_u[j][k],&m->lstms[i]->d1_u[j][k]);
                if(k < m->lstms[i]->size)
                    nesterov_momentum(&m->lstms[i]->biases[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_biases[j][k],&m->lstms[i]->d1_biases[j][k]);
            }
        }
    }
}

/* Given a rmodel, this function update the params of the lstm layers of the model with the adam algorithm
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 * 
 * */
void update_lstm_layer_adam(rmodel* m,float lr,int mini_batch_size,float b1, float b2){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->size*m->lstms[i]->size; k++){
                adam_algorithm(&m->lstms[i]->w[j][k],&m->lstms[i]->d1_w[j][k],&m->lstms[i]->d2_w[j][k],m->lstms[i]->d_w[j][k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                adam_algorithm(&m->lstms[i]->u[j][k],&m->lstms[i]->d1_u[j][k],&m->lstms[i]->d2_u[j][k],m->lstms[i]->d_u[j][k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
                if(k < m->lstms[i]->size)
                    adam_algorithm(&m->lstms[i]->biases[j][k],&m->lstms[i]->d1_biases[j][k],&m->lstms[i]->d2_biases[j][k],m->lstms[i]->d_biases[j][k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size);
            }
        }
    }
}


/* Given a rmodel, this function update the params of the lstm layers of the model with the adam algorithm
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_lstm_layer_radam(rmodel* m,float lr,int mini_batch_size,float b1, float b2, unsigned long long int t){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->size*m->lstms[i]->size; k++){
                radam_algorithm(&m->lstms[i]->w[j][k],&m->lstms[i]->d1_w[j][k],&m->lstms[i]->d2_w[j][k],m->lstms[i]->d_w[j][k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                radam_algorithm(&m->lstms[i]->u[j][k],&m->lstms[i]->d1_u[j][k],&m->lstms[i]->d2_u[j][k],m->lstms[i]->d_u[j][k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                if(k < m->lstms[i]->size)
                    radam_algorithm(&m->lstms[i]->biases[j][k],&m->lstms[i]->d1_biases[j][k],&m->lstms[i]->d2_biases[j][k],m->lstms[i]->d_biases[j][k],lr,BETA1_ADAM,BETA2_ADAM,b1,b2,EPSILON_ADAM,mini_batch_size,t);
            }
        }
    }
}


/* This function sum the partial derivatives of the lstm layers of a rmodel m and a second rmodel m2 in a third rmodel m3
 * 
 * Input:
 * 
 *             @ rmodel* m:= the input rmodel
 *             @ rmodel* m2:= another input rmodel
 *             @ rmodel* m3:= the output rmodel
 * 
 * */
void sum_lstm_layers_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: you passed a NULL pointer as argument\n");
        exit(1);
    }
    int i,j;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            sum1D(m->lstms[i]->d_w[j],m2->lstms[i]->d_w[j],m3->lstms[i]->d_w[j],m->lstms[i]->size*m->lstms[i]->size);
            sum1D(m->lstms[i]->d_u[j],m2->lstms[i]->d_u[j],m3->lstms[i]->d_u[j],m->lstms[i]->size*m->lstms[i]->size);
            sum1D(m->lstms[i]->d_biases[j],m2->lstms[i]->d_biases[j],m3->lstms[i]->d_biases[j],m->lstms[i]->size);
        }
    
        if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
            for(j = 0; j < m->lstms[i]->window/m->lstms[i]->n_grouped_cell; j++){
                sum1D(m->lstms[i]->bns[j]->d_gamma,m2->lstms[i]->bns[j]->d_gamma,m3->lstms[i]->bns[j]->d_gamma,m->lstms[i]->bns[j]->vector_dim);
                sum1D(m->lstms[i]->bns[j]->d_beta,m2->lstms[i]->bns[j]->d_beta,m3->lstms[i]->bns[j]->d_beta,m->lstms[i]->bns[j]->vector_dim);
            }
        }
    }
    

}


/* This function frees a space allocated by a matrix
 * 
 * Input:
 * 
 *             @ float** m:= the matrix m
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

