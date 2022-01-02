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

/* this function builds a normalization layer that can be used for batch normalization or group normalization or layer normalization
 * 
 * Input:
 * 
 *             @ int batch_size:= the batch size used
 *             @ int vector_input_dimension:= the dimension of the input of this layer, or the output dimension of the previous layer
 * 
 * */
bn* batch_normalization(int batch_size, int vector_input_dimension){
    
    if(batch_size <= 0 || vector_input_dimension < 1){
        fprintf(stderr,"Error: batch size <= 0 and vector_input:dimension < 1 are not admissible!\n");
        exit(1);
    }
    
    int i;
    bn* b = (bn*)malloc(sizeof(bn));
    b->batch_size = batch_size; 
    b->vector_dim = vector_input_dimension;
    
    b->input_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->error2 = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp1 = (float**)malloc(sizeof(float*)*batch_size); 
    b->outputs = (float**)malloc(sizeof(float*)*batch_size);
    
    b->gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d3_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d3_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->mean = (float*)calloc(vector_input_dimension,sizeof(float));
    b->var = (float*)calloc(vector_input_dimension,sizeof(float));
    b->temp2 = (float*)calloc(vector_input_dimension,sizeof(float));
    
    for(i = 0; i < batch_size; i++){
        b->input_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->error2[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp1[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->outputs[i] = (float*)calloc(vector_input_dimension,sizeof(float));
    }
    
    for(i = 0; i < vector_input_dimension; i++){
        b->gamma[i] = 1;
    }
    
    b->epsilon = EPSILON;
    b->training_mode = GRADIENT_DESCENT;
    
    return b;
}

/* this function builds a normalization layer without however any array initialized
 * 
 * Input:
 * 
 *             @ int batch_size:= the batch size used
 *             @ int vector_input_dimension:= the dimension of the input of this layer, or the output dimension of the previous layer
 * 
 * */
bn* batch_normalization_without_arrays(int batch_size, int vector_input_dimension){
    if(batch_size <= 0 || vector_input_dimension < 1){
        fprintf(stderr,"Error: batch size <= 0 and vector_input:dimension < 1 are not admissible!\n");
        exit(1);
    }
    
    
    int i;
    bn* b = (bn*)malloc(sizeof(bn));
    b->batch_size = batch_size; 
    b->vector_dim = vector_input_dimension;
    b->epsilon = EPSILON;
    b->training_mode = GRADIENT_DESCENT;
    
    return b;
}


/* this function builds a normalization layer that can be used for batch normalization or group normalization or layer normalization without weights and biases and d1, d2, d3 (learning parameters)
 * 
 * Input:
 * 
 *             @ int batch_size:= the batch size used
 *             @ int vector_input_dimension:= the dimension of the input of this layer, or the output dimension of the previous layer
 * 
 * */
bn* batch_normalization_without_learning_parameters(int batch_size, int vector_input_dimension){
    if(batch_size <= 0 || vector_input_dimension < 1){
        fprintf(stderr,"Error: batch size <= 0 and vector_input:dimension < 1 are not admissible!\n");
        exit(1);
    }
    int i;
    bn* b = (bn*)malloc(sizeof(bn));
    b->batch_size = batch_size; 
    b->vector_dim = vector_input_dimension;
    
    b->input_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->error2 = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp1 = (float**)malloc(sizeof(float*)*batch_size); 
    b->outputs = (float**)malloc(sizeof(float*)*batch_size);
    b->gamma = NULL;
    b->d_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_gamma = NULL;
    b->d2_gamma = NULL;
    b->d3_gamma = NULL;
    b->beta = NULL;
    b->d_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_beta = NULL;
    b->d2_beta = NULL;
    b->d3_beta = NULL;
    b->mean = (float*)calloc(vector_input_dimension,sizeof(float));
    b->var = (float*)calloc(vector_input_dimension,sizeof(float));
    b->temp2 = (float*)calloc(vector_input_dimension,sizeof(float));
    
    for(i = 0; i < batch_size; i++){
        b->input_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->error2[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp1[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->outputs[i] = (float*)calloc(vector_input_dimension,sizeof(float));
    }
    
    
    b->epsilon = EPSILON;
    b->training_mode = GRADIENT_DESCENT;
    
    return b;
}

/* This function deallocates the space allocated by the arrays not used during the feed forward for the bn layer
 * 
 * Input:
 * 
 *             @ bn* b:= the structure
 * 
 * */
void make_the_bn_only_for_ff(bn* b){
    if(b == NULL)
        return;
    free_matrix((void**)b->error2,b->batch_size);
    free_matrix((void**)b->temp1,b->batch_size);
    
    
    free(b->d_gamma);
    free(b->d1_gamma);
    free(b->d2_gamma);
    free(b->d3_gamma);
    free(b->d_beta);
    free(b->d1_beta);
    free(b->d2_beta);
    free(b->d3_beta);
    free(b->temp2);
    b->error2 = NULL;
    b->temp1 = NULL;
    b->d_gamma = NULL;
    b->d1_gamma = NULL;
    b->d2_gamma = NULL;
    b->d3_gamma = NULL;
    b->d_beta = NULL;
    b->d1_beta = NULL;
    b->d2_beta = NULL;
    b->d3_beta = NULL;
    b->temp2 = NULL;
}

/* This function deallocates the remaining space occupied by b (which has been partially deallocated by make_the_bn_only_for_ff)
 * 
 * Input:
 * 
 *             @ bn* b:= the structure
 * 
 * */
void free_the_bn_only_for_ff(bn* b){
    if(b == NULL)
        return;
    
    free_matrix((void**)b->input_vectors,b->batch_size);
    free_matrix((void**)b->temp_vectors,b->batch_size);
    free_matrix((void**)b->outputs,b->batch_size);

    free(b->gamma);
    free(b->beta);
    free(b->mean);
    free(b->var);

    free(b);
}


/* This function deallocates the space allocated by a bn structure
 * 
 * Input:
 * 
 *             @ bn* b:= the structure
 * 
 * */
void free_batch_normalization(bn* b){
    if(b == NULL)
        return;
    
    free_matrix((void**)b->input_vectors,b->batch_size);
    free_matrix((void**)b->temp_vectors,b->batch_size);
    free_matrix((void**)b->error2,b->batch_size);
    free_matrix((void**)b->temp1,b->batch_size);
    free_matrix((void**)b->outputs,b->batch_size);

    
    free(b->gamma);
    free(b->d_gamma);
    free(b->d1_gamma);
    free(b->d2_gamma);
    free(b->d3_gamma);
    free(b->beta);
    free(b->d_beta);
    free(b->d1_beta);
    free(b->d2_beta);
    free(b->d3_beta);
    free(b->temp2);
    free(b->mean);
    free(b->var);
    free(b);
}

/* This function saves a batch normalized layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ bn* b:= the actual layer that must be saved
 *             @ int n:= the name (number) of the bin file where the layer is saved (n.bin)
 * 
 * 
 * */
void save_bn(bn* b, int n){
    if(b == NULL || n < 0)
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
 *                 @ FILE* fr:= the file fwhere the batch normalized layer must been loaded from
 * 
 * */
bn* load_bn(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int batch_size = 0,vector_dim = 0;
    float* gamma;
    float* beta;
    
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

    
    gamma = (float*)malloc(sizeof(float)*vector_dim);
    beta = (float*)malloc(sizeof(float)*vector_dim);
    
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

    
    
    
    bn* b = batch_normalization(batch_size,vector_dim);
    
    copy_array(gamma,b->gamma,vector_dim);
    copy_array(beta,b->beta,vector_dim);
    free(gamma);
    free(beta);
    
    return b;
    
}

/* This function loads a batch_normalized layer from a file, and makes it only for feed forward
 * 
 * Inputs:
 * 
 *                 @ FILE* fr:= the file where the batch normalized layer must been loaded from
 * 
 * */
bn* load_bn_only_for_ff(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int batch_size = 0,vector_dim = 0;
    float* gamma;
    float* beta;

    
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

    
    gamma = (float*)malloc(sizeof(float)*vector_dim);
    beta = (float*)malloc(sizeof(float)*vector_dim);
    
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

    
    
    
    bn* b = batch_normalization(batch_size,vector_dim);
    
    copy_array(gamma,b->gamma,vector_dim);
    copy_array(beta,b->beta,vector_dim);
    free(gamma);
    free(beta);
    make_the_bn_only_for_ff(b);
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
    bn* copy = batch_normalization(b->batch_size,b->vector_dim);
    copy_array(b->gamma,copy->gamma,b->vector_dim);
    copy_array(b->d_gamma,copy->d_gamma,b->vector_dim);
    copy_array(b->d1_gamma,copy->d1_gamma,b->vector_dim);
    copy_array(b->d2_gamma,copy->d2_gamma,b->vector_dim);
    copy_array(b->d3_gamma,copy->d3_gamma,b->vector_dim);
    copy_array(b->beta,copy->beta,b->vector_dim);
    copy_array(b->d_beta,copy->d_beta,b->vector_dim);
    copy_array(b->d1_beta,copy->d1_beta,b->vector_dim);
    copy_array(b->d2_beta,copy->d2_beta,b->vector_dim);
    copy_array(b->d3_beta,copy->d3_beta,b->vector_dim);
    copy->training_mode = b->training_mode;
    copy->epsilon = b->epsilon;
    return copy;
}
/* This function returns a bn* layer that is the same copy of the input b, but without learning parameters (\ {d1, d2, ...}).
 * 
 * Input:
 * 
 *             @ bn* b:= the batch normalized layer that must be copied
 * 
 * */ 
bn* copy_bn_without_learning_parameters(bn* b){
    if(b == NULL)
        return NULL;
    bn* copy = batch_normalization_without_learning_parameters(b->batch_size,b->vector_dim);
    copy_array(b->d_gamma,copy->d_gamma,b->vector_dim);
    copy_array(b->d_beta,copy->d_beta,b->vector_dim);
    copy->training_mode = b->training_mode;
    copy->epsilon = b->epsilon;
    return copy;
}


/* this function resets all the arrays of a batch normalized layer (used by feed forward and back propagation).
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
    for(i = 0; i < b->batch_size; i++){
        set_vector_with_value(0,b->input_vectors[i],b->vector_dim);
        set_vector_with_value(0,b->temp_vectors[i],b->vector_dim);
        set_vector_with_value(0,b->outputs[i],b->vector_dim);
        if(b->error2 != NULL)
        set_vector_with_value(0,b->error2[i],b->vector_dim);
        if(b->temp1 != NULL)
        set_vector_with_value(0,b->temp1[i],b->vector_dim);
    }
    if(b->d_gamma != NULL)
    set_vector_with_value(0,b->d_gamma,b->vector_dim);
    if(b->d_beta != NULL)
    set_vector_with_value(0,b->d_beta,b->vector_dim);
    if(b->temp2 != NULL)
    set_vector_with_value(0,b->temp2,b->vector_dim);
    set_vector_with_value(0,b->mean,b->vector_dim);
    set_vector_with_value(0,b->var,b->vector_dim);
    
    
    return b;
}

/* this function resets all the arrays of a batch normalized layer (used by feed forward and back propagation) except from partial derivatives.
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
    for(i = 0; i < b->batch_size; i++){
        set_vector_with_value(0,b->input_vectors[i],b->vector_dim);
        set_vector_with_value(0,b->temp_vectors[i],b->vector_dim);
        set_vector_with_value(0,b->outputs[i],b->vector_dim);
        set_vector_with_value(0,b->error2[i],b->vector_dim);
        set_vector_with_value(0,b->temp1[i],b->vector_dim);
    }
    
    set_vector_with_value(0,b->temp2,b->vector_dim);
    set_vector_with_value(0,b->mean,b->vector_dim);
    set_vector_with_value(0,b->var,b->vector_dim);
    
    
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
uint64_t size_of_bn(bn* b){
	if (b == NULL)
		return 0;
    uint64_t sum = 0;
    sum+= (b->batch_size*b->vector_dim*5);
    sum+= (b->vector_dim*13);
    return sum;
}
/* this function computes the size of the space allocated by the arrays of a batch normalized layer without learning parameters (more or less)
 * just to give an idea of the size occupied by this structure
 * 
 * Input:
 * 
 *             bn* b:= the batch normalized layer b
 * 
 * */
uint64_t size_of_bn_without_learning_parameters(bn* b){
	if(b == NULL)
		return 0;
    uint64_t sum = 0;
    sum+= (b->batch_size*b->vector_dim*5);
    sum+= (b->vector_dim*5);
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
    copy_array(b1->d1_gamma,b2->d1_gamma,b1->vector_dim);
    copy_array(b1->d2_gamma,b2->d2_gamma,b1->vector_dim);
    copy_array(b1->d3_gamma,b2->d3_gamma,b1->vector_dim);
    copy_array(b1->beta,b2->beta,b1->vector_dim);
    copy_array(b1->d_beta,b2->d_beta,b1->vector_dim);
    copy_array(b1->d1_beta,b2->d1_beta,b1->vector_dim);
    copy_array(b1->d2_beta,b2->d2_beta,b1->vector_dim);
    copy_array(b1->d3_beta,b2->d3_beta,b1->vector_dim);
    copy_array(b1->d3_beta,b2->d3_beta,b1->vector_dim);
    
    return;
}
/* This function returns a bn* layer that is the same copy of the input b1
 * except for temp arrays used for feed forward and backprop, both the structures are "without learning parameters" structures
 * Input:
 * 
 *             @ bn* b1:= the batch normalized layer that must be copied
 *             @ bn* b2:= the batch normalized layer where b1 is copied
 * 
 * */
void paste_bn_without_learning_parameters(bn* b1, bn* b2){
    if(b1 == NULL || b2 == NULL)
        return;
    
    copy_array(b1->d_gamma,b2->d_gamma,b1->vector_dim);
    copy_array(b1->d_beta,b2->d_beta,b1->vector_dim);
    
    return;
}


/* this function copies the gamma and beta parameters from a bn structure to another one
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
    
    return;
}

/* This function returns a bn* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i, the copy still works also 
 * for d1, d2, ... parameters
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
        copy->beta[i] = tau*f->beta[i] + (1-tau)*copy->beta[i];
        copy->d1_beta[i] = tau*f->d1_beta[i] + (1-tau)*copy->d1_beta[i];
        copy->d2_beta[i] = tau*f->d2_beta[i] + (1-tau)*copy->d2_beta[i];
        copy->d3_beta[i] = tau*f->d3_beta[i] + (1-tau)*copy->d3_beta[i];
    }
    
    return;
}
