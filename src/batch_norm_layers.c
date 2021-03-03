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

/* this function builds a batch normalization layer
 * 
 * Input:
 * 
 *             @ int batch_size:= the batch size used
 *             @ int vector_input_dimension:= the dimension of the input of this layer, or the output dimension of the previous layer
 *                @ int layer:= the layer
 *                @ int activation_flag:= for the moment is useless
 * 
 * */
bn* batch_normalization(int batch_size, int vector_input_dimension, int layer, int activation_flag){
    if(batch_size <= 1 || vector_input_dimension < 1){
        fprintf(stderr,"Warning: remember if you are using online learning (batch_size = 1) batch normalization is useless, and remember also that vector input dimension must be >= 1\n");
    }
    int i;
    bn* b = (bn*)malloc(sizeof(bn));
    b->layer = layer;
    b->batch_size = batch_size; 
    b->vector_dim = vector_input_dimension;
    b->activation_flag = activation_flag;
    
    b->input_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->error2 = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp1 = (float**)malloc(sizeof(float*)*batch_size); 
    b->outputs = (float**)malloc(sizeof(float*)*batch_size);
    b->post_activation = (float**)malloc(sizeof(float*)*batch_size);
    
    b->gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->ex_d_gamma_diff_grad = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d3_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->ex_d_beta_diff_grad = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d3_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->mean = (float*)calloc(vector_input_dimension,sizeof(float));
    b->var = (float*)calloc(vector_input_dimension,sizeof(float));
    b->temp2 = (float*)calloc(vector_input_dimension,sizeof(float));
    b->final_mean = (float*)calloc(vector_input_dimension,sizeof(float));
    b->final_var = (float*)calloc(vector_input_dimension,sizeof(float));
    b->mode_flag = BATCH_NORMALIZATION_TRAINING_MODE;
    
    for(i = 0; i < batch_size; i++){
        b->input_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->error2[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp1[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->outputs[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->post_activation[i] = (float*)calloc(vector_input_dimension,sizeof(float));
    }
    
    for(i = 0; i < vector_input_dimension; i++){
        b->gamma[i] = 1;
    }
    
    b->epsilon = EPSILON;
    
    return b;
}

/* This functions deallocates the space allocated by a bn structure
 * 
 * Input:
 * 
 *             @ bn* b:= the structure
 * 
 * */
void free_batch_normalization(bn* b){
    if(b == NULL)
        return;
    int i;
    for(i = 0; i < b->batch_size; i++){
        free(b->input_vectors[i]);
        free(b->temp_vectors[i]);
        free(b->error2[i]);
        free(b->temp1[i]);
        free(b->outputs[i]);
        free(b->post_activation[i]);
    }
    free(b->input_vectors);
    free(b->temp_vectors);
    free(b->error2);
    free(b->temp1);
    free(b->outputs);
    free(b->post_activation);
    
    free(b->gamma);
    free(b->d_gamma);
    free(b->ex_d_gamma_diff_grad);
    free(b->d1_gamma);
    free(b->d2_gamma);
    free(b->d3_gamma);
    free(b->beta);
    free(b->d_beta);
    free(b->ex_d_beta_diff_grad);
    free(b->d1_beta);
    free(b->d2_beta);
    free(b->d3_beta);
    free(b->temp2);
    free(b->final_mean);
    free(b->final_var);
    free(b->mean);
    free(b->var);
    free(b);
}

/* This function saves a batch normalized layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ bn* b:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_bn(bn* b, int n){
    if(b == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&b->layer,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->batch_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->vector_dim,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->activation_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->mode_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->gamma,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    
    i = fwrite(b->beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->final_mean,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->final_var,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    free(s);
    
    
}

/* This function saves a batch normalized layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ bn* b:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void heavy_save_bn(bn* b, int n){
    if(b == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&b->layer,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->batch_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->vector_dim,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->activation_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->mode_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->gamma,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->d1_gamma,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->d2_gamma,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->d3_gamma,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->ex_d_gamma_diff_grad,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->d1_beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    i = fwrite(b->d2_beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    i = fwrite(b->d3_beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    i = fwrite(b->d_beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    i = fwrite(b->final_mean,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->final_var,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    free(s);
    
    
}

/* This function loads a batch_normalized layer from a file
 * 
 * Inputs:
 * 
 *                 @ FILE* fr:= the file from where the batch normalized layer must been loaded
 * 
 * */
bn* load_bn(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int batch_size = 0,vector_dim = 0, layer = 0, activation_flag,mode_flag;
    float* gamma;
    float* beta;
    float* final_mean;
    float* final_var;
    
    i = fread(&layer,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&batch_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&vector_dim,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&activation_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&mode_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    gamma = (float*)malloc(sizeof(float)*vector_dim);
    beta = (float*)malloc(sizeof(float)*vector_dim);
    final_mean = (float*)malloc(sizeof(float)*vector_dim);
    final_var = (float*)malloc(sizeof(float)*vector_dim);
    
    i = fread(gamma,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    i = fread(beta,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(final_mean,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    i = fread(final_var,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    
    bn* b = batch_normalization(batch_size,vector_dim, layer, activation_flag);
    
    copy_array(gamma,b->gamma,vector_dim);
    copy_array(beta,b->beta,vector_dim);
    copy_array(final_mean,b->final_mean,vector_dim);
    copy_array(final_var,b->final_var,vector_dim);
    b->mode_flag = mode_flag;
    free(gamma);
    free(beta);
    free(final_mean);
    free(final_var);
    
    return b;
    
}


/* This function loads a batch_normalized layer from a file
 * 
 * Inputs:
 * 
 *                 @ FILE* fr:= the file from where the batch normalized layer must been loaded
 * 
 * */
bn* heavy_load_bn(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int batch_size = 0,vector_dim = 0, layer = 0, activation_flag,mode_flag;
    float* gamma;
    float* d1_gamma;
    float* d2_gamma;
    float* d3_gamma;
    float* ex_d_gamma_diff_grad;
    float* beta;
    float* d1_beta;
    float* d2_beta;
    float* d3_beta;
    float* ex_d_beta_diff_grad;
    float* final_mean;
    float* final_var;
    
    i = fread(&layer,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&batch_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&vector_dim,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&activation_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&mode_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    gamma = (float*)malloc(sizeof(float)*vector_dim);
    d1_gamma = (float*)malloc(sizeof(float)*vector_dim);
    d2_gamma = (float*)malloc(sizeof(float)*vector_dim);
    d3_gamma = (float*)malloc(sizeof(float)*vector_dim);
    ex_d_gamma_diff_grad = (float*)malloc(sizeof(float)*vector_dim);
    beta = (float*)malloc(sizeof(float)*vector_dim);
    d1_beta = (float*)malloc(sizeof(float)*vector_dim);
    d2_beta = (float*)malloc(sizeof(float)*vector_dim);
    d3_beta = (float*)malloc(sizeof(float)*vector_dim);
    ex_d_beta_diff_grad = (float*)malloc(sizeof(float)*vector_dim);
    final_mean = (float*)malloc(sizeof(float)*vector_dim);
    final_var = (float*)malloc(sizeof(float)*vector_dim);
    
    i = fread(gamma,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(d1_gamma,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(d2_gamma,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(d3_gamma,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(ex_d_gamma_diff_grad,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    i = fread(beta,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(d1_beta,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    i = fread(d2_beta,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    i = fread(d3_beta,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    i = fread(ex_d_beta_diff_grad,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    i = fread(final_mean,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    i = fread(final_var,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    
    bn* b = batch_normalization(batch_size,vector_dim, layer, activation_flag);
    
    copy_array(gamma,b->gamma,vector_dim);
    copy_array(d1_gamma,b->d1_gamma,vector_dim);
    copy_array(d2_gamma,b->d2_gamma,vector_dim);
    copy_array(d3_gamma,b->d3_gamma,vector_dim);
    copy_array(ex_d_gamma_diff_grad,b->ex_d_gamma_diff_grad,vector_dim);
    copy_array(beta,b->beta,vector_dim);
    copy_array(d1_beta,b->d1_beta,vector_dim);
    copy_array(d2_beta,b->d2_beta,vector_dim);
    copy_array(d3_beta,b->d3_beta,vector_dim);
    copy_array(ex_d_beta_diff_grad,b->ex_d_beta_diff_grad,vector_dim);
    copy_array(final_mean,b->final_mean,vector_dim);
    copy_array(final_var,b->final_var,vector_dim);
    b->mode_flag = mode_flag;
    free(gamma);
    free(d1_gamma);
    free(d2_gamma);
    free(d3_gamma);
    free(ex_d_gamma_diff_grad);
    free(beta);
    free(d1_beta);
    free(d2_beta);
    free(d3_beta);
    free(ex_d_beta_diff_grad);
    free(final_mean);
    free(final_var);
    
    return b;
    
}

/* This function returns a bn* layer that is the same copy of the input b.
 * 
 * Input:
 * 
 *             @ bn* b:= the batch normalized layer that must be copied
 * 
 * */ 
bn* copy_bn(bn* b){
    if(b == NULL)
        return NULL;
    bn* copy = batch_normalization(b->batch_size,b->vector_dim, b->layer, b->activation_flag);
    copy_array(b->gamma,copy->gamma,b->vector_dim);
    copy_array(b->d_gamma,copy->d_gamma,b->vector_dim);
    copy_array(b->ex_d_gamma_diff_grad,copy->ex_d_gamma_diff_grad,b->vector_dim);
    copy_array(b->d1_gamma,copy->d1_gamma,b->vector_dim);
    copy_array(b->d2_gamma,copy->d2_gamma,b->vector_dim);
    copy_array(b->d3_gamma,copy->d3_gamma,b->vector_dim);
    copy_array(b->beta,copy->beta,b->vector_dim);
    copy_array(b->d_beta,copy->d_beta,b->vector_dim);
    copy_array(b->ex_d_beta_diff_grad,copy->ex_d_beta_diff_grad,b->vector_dim);
    copy_array(b->d1_beta,copy->d1_beta,b->vector_dim);
    copy_array(b->d2_beta,copy->d2_beta,b->vector_dim);
    copy_array(b->d3_beta,copy->d3_beta,b->vector_dim);
    copy_array(b->final_mean,copy->final_mean,b->vector_dim);
    copy_array(b->final_var,copy->final_var,b->vector_dim);
    copy->mode_flag = b->mode_flag;
    
    return copy;
}


/* this function resets all the arrays of a batch normalized layer (used by feed forward and back propagation) but it keeps the weights andbiases.
 * 
 * 
 * Input:
 * 
 *             @ bn* b:= a bn* b layer
 * 
 * */
bn* reset_bn(bn* b){
    if(b == NULL)
        return NULL;
    int i,j;
    for(i = 0; i < b->vector_dim; i++){
        for(j = 0; j < b->batch_size; j++){
            b->input_vectors[j][i] = 0;
            b->temp_vectors[j][i] = 0;
            b->outputs[j][i] = 0;
            b->post_activation[j][i] = 0;
            b->error2[j][i] = 0;
            b->temp1[j][i] = 0;
        }
        
        b->d_gamma[i] = 0; 
        b->d_beta[i] = 0; 
        b->temp2[i] = 0; 
        b->mean[i] = 0; 
        b->var[i] = 0; 
    } 
    
    return b;
}
/* this function resets all the arrays of a batch normalized layer (used by feed forward and back propagation) but it keeps the weights andbiases.
 * 
 * 
 * Input:
 * 
 *             @ bn* b:= a bn* b layer
 * 
 * */
bn* reset_bn_except_partial_derivatives(bn* b){
    if(b == NULL)
        return NULL;
    int i,j;
    for(i = 0; i < b->vector_dim; i++){
        for(j = 0; j < b->batch_size; j++){
            b->input_vectors[j][i] = 0;
            b->temp_vectors[j][i] = 0;
            b->outputs[j][i] = 0;
            b->post_activation[j][i] = 0;
            b->error2[j][i] = 0;
            b->temp1[j][i] = 0;
        }
        
        b->temp2[i] = 0; 
        b->mean[i] = 0; 
        b->var[i] = 0; 
    } 
    
    return b;
}

/* this function computes the size of the space allocated by the arrays of a batch normalized layer (more or less)
 * just to give an idea of the size occupied by this structure
 * 
 * Input:
 * 
 *             bn* b:= the batch normalized layer b
 * 
 * */
unsigned long long int size_of_bn(bn* b){
    unsigned long long int sum = 0;
    sum+= (b->batch_size*b->vector_dim*6);
    sum+= (b->vector_dim*17);
    return sum;
}

/* This function returns a bn* layer that is the same copy of the input b1
 * except for temp arrays used for feed forward and backprop 
 * Input:
 * 
 *             @ bn* b1:= the batch normalized layer that must be copied
 *             @ bn* b2:= the batch normalized layer where b1 is copied
 * 
 * */
void paste_bn(bn* b1, bn* b2){
    if(b1 == NULL || b2 == NULL)
        return;
    
    copy_array(b1->gamma,b2->gamma,b1->vector_dim);
    copy_array(b1->d_gamma,b2->d_gamma,b1->vector_dim);
    copy_array(b1->ex_d_gamma_diff_grad,b2->ex_d_gamma_diff_grad,b1->vector_dim);
    copy_array(b1->d1_gamma,b2->d1_gamma,b1->vector_dim);
    copy_array(b1->d2_gamma,b2->d2_gamma,b1->vector_dim);
    copy_array(b1->d3_gamma,b2->d3_gamma,b1->vector_dim);
    copy_array(b1->beta,b2->beta,b1->vector_dim);
    copy_array(b1->d_beta,b2->d_beta,b1->vector_dim);
    copy_array(b1->ex_d_beta_diff_grad,b2->ex_d_beta_diff_grad,b1->vector_dim);
    copy_array(b1->d1_beta,b2->d1_beta,b1->vector_dim);
    copy_array(b1->d2_beta,b2->d2_beta,b1->vector_dim);
    copy_array(b1->d3_beta,b2->d3_beta,b1->vector_dim);
    copy_array(b1->d3_beta,b2->d3_beta,b1->vector_dim);
    copy_array(b1->final_mean,b2->final_mean,b1->vector_dim);
    copy_array(b1->final_var,b2->final_var,b1->vector_dim);
    
    return;
}

/* This function returns a bn* layer that is the same copy of the input b1
 * except for temp arrays used for feed forward and backprop 
 * Input:
 * 
 *             @ bn* b1:= the batch normalized layer that must be copied
 *             @ bn* b2:= the batch normalized layer where b1 is copied
 * 
 * */
void paste_w_bn(bn* b1, bn* b2){
    if(b1 == NULL || b2 == NULL)
        return;
    
    copy_array(b1->gamma,b2->gamma,b1->vector_dim);
    copy_array(b1->beta,b2->beta,b1->vector_dim);
    copy_array(b1->final_mean,b2->final_mean,b1->vector_dim);
    copy_array(b1->final_var,b2->final_var,b1->vector_dim);
    
    return;
}
/* This function returns a bn* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ bn* f:= the batch normalized layer that must be copied
 *             @ bn* copy:= the batch normalized layer where f is copied
 *             @ float tau:= the tau param
 * */
void slow_paste_bn(bn* f, bn* copy,float tau){
    if(f == NULL)
        return;
    
    int i,j;
    for(i = 0; i < f->vector_dim; i++){
        copy->gamma[i] = tau*f->gamma[i] + (1-tau)*copy->gamma[i];
        copy->d1_gamma[i] = tau*f->d1_gamma[i] + (1-tau)*copy->d1_gamma[i];
        copy->d2_gamma[i] = tau*f->d2_gamma[i] + (1-tau)*copy->d2_gamma[i];
        copy->d3_gamma[i] = tau*f->d3_gamma[i] + (1-tau)*copy->d3_gamma[i];
        copy->ex_d_gamma_diff_grad[i] = tau*f->ex_d_gamma_diff_grad[i] + (1-tau)*copy->ex_d_gamma_diff_grad[i];
        copy->beta[i] = tau*f->beta[i] + (1-tau)*copy->beta[i];
        copy->d1_beta[i] = tau*f->d1_beta[i] + (1-tau)*copy->d1_beta[i];
        copy->d2_beta[i] = tau*f->d2_beta[i] + (1-tau)*copy->d2_beta[i];
        copy->d3_beta[i] = tau*f->d3_beta[i] + (1-tau)*copy->d3_beta[i];
        copy->ex_d_beta_diff_grad[i] = tau*f->ex_d_beta_diff_grad[i] + (1-tau)*copy->ex_d_beta_diff_grad[i];
    }
    
    return;
}
