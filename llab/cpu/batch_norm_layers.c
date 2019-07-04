#include "llab.h"

/* this functions build a batch normalization layer
 * 
 * Input:
 * 
 *             @ int batch_size:= the batch size used
 *             @ int vector_input_dimension:= the dimension of the input of this layer, or the output dimension of the previous layer
 * 			   @ int layer:= the layer
 * 			   @ int activation_flag:= for the moment is useless
 * 
 * */
bn* batch_normalization(int batch_size, int vector_input_dimension, int layer, int activation_flag){
    if(batch_size <= 1 || vector_input_dimension < 1){
        fprintf(stderr,"Error: remember if you are using online learning (batch_size = 1) batch normalization is useless, and remember also that vector input dimension must be >= 1\n");
        exit(1);
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
    b->d1_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_beta = (float*)calloc(vector_input_dimension,sizeof(float));
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
    free(b->d1_gamma);
    free(b->d2_gamma);
    free(b->beta);
    free(b->d_beta);
    free(b->d1_beta);
    free(b->d2_beta);
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


/* This function load a batch_normalized layer from a file
 * 
 * Inputs:
 * 
 * 				@ FILE* fr:= the file where the batch normalized layer must been loaded
 * 
 * */
bn* load_bn(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int batch_size = 0,vector_dim = 0, layer = 0, activation_flag;
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
    
    free(gamma);
    free(beta);
    free(final_mean);
    free(final_var);
    
    return b;
    
}

/* This function returns a bn* layer that is the same copy of the input f
 * arrays used during the feed forward and backpropagation
 * 
 * Input:
 * 
 *             @ bn* f:= the batch normalized layer that must be copied
 * 
 * */ 
bn* copy_bn(bn* b){
    if(b == NULL)
        return NULL;
    bn* copy = batch_normalization(b->batch_size,b->vector_dim, b->layer, b->activation_flag);
    copy_array(b->gamma,copy->gamma,b->vector_dim);
    copy_array(b->d_gamma,copy->d_gamma,b->vector_dim);
    copy_array(b->d1_gamma,copy->d1_gamma,b->vector_dim);
    copy_array(b->d2_gamma,copy->d2_gamma,b->vector_dim);
    copy_array(b->beta,copy->beta,b->vector_dim);
    copy_array(b->d_beta,copy->d_beta,b->vector_dim);
    copy_array(b->d1_beta,copy->d1_beta,b->vector_dim);
    copy_array(b->d2_beta,copy->d2_beta,b->vector_dim);
    copy_array(b->final_mean,copy->final_mean,b->vector_dim);
    copy_array(b->final_var,copy->final_var,b->vector_dim);
    
    return copy;
}

/* this function reset all the arrays of a batch normalized layer
 * 
 * 
 * Input:
 * 
 *             @ bn* b:= a bn* f layer
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
}

/* this function computes the size of the space allocated by the arrays of a batch normalized layer
 * 
 * Input:
 * 
 *             bn* b:= the batch normalized layer b
 * 
 * */
unsigned long long int size_of_bn(bn* b){
    unsigned long long int sum = 0;
    sum+= (b->batch_size*b->vector_dim*6);
    sum+= (b->vector_dim*13);
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
    copy_array(b1->beta,b2->beta,b1->vector_dim);
    copy_array(b1->d_beta,b2->d_beta,b1->vector_dim);
    copy_array(b1->d1_beta,b2->d1_beta,b1->vector_dim);
    copy_array(b1->d2_beta,b2->d2_beta,b1->vector_dim);
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
        copy->beta[i] = tau*f->beta[i] + (1-tau)*copy->beta[i];
    }
    
    return;
}
