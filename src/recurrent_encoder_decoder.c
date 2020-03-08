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

/* This function initialize a recurrent encoder - decoder network with attention mechanism
 * 
 * Inputs:
 * 
 * 
 *                 @ rmodel* encoder:= the encoder
 *                 @ rmodel* decoder:= the decoder
 * */
recurrent_enc_dec* recurrent_enc_dec_network(rmodel* encoder, rmodel* decoder){
    recurrent_enc_dec* r = (recurrent_enc_dec*)malloc(sizeof(recurrent_enc_dec));
    r->encoder = encoder;
    r->decoder = decoder;
    r->attention_weights = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->d_attention_weights = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->d1_attention_weights = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->d2_attention_weights = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->ex_d_attention_weights_diff_grad = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->score = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->post_activation = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    r->context_vector = (float**)malloc(sizeof(float*)*encoder->lstms[0]->size);
    int i;
    for(i = 0; i < encoder->lstms[0]->size; i++){
        r->attention_weights[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
        r->d1_attention_weights[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
        r->d2_attention_weights[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
        r->ex_d_attention_weights_diff_grad[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
        r->score[i] = (float*)calloc(encoder->lstms[0]->window,sizeof(float));
        r->post_activation[i] = (float*)calloc(encoder->lstms[0]->window,sizeof(float));
        r->context_vector[i] = (float*)calloc(encoder->lstms[0]->window,sizeof(float));
    }
    
    return r;
}

/* This function deallocates the space allocated by a recurrent_enc_dec struct
 * 
 * Inputs:
 * 
 * 
 *             @ recurrent_enc_dec* r
 * */
void free_recurrent_enc_dec(recurrent_enc_dec* r){
    int size = r->encoder->lstms[0]->size,i;
    free_rmodel(r->encoder);
    free_rmodel(r->decoder);
    for(i = 0; i < size; i++){
        free(r->attention_weights[i]);
        free(r->d_attention_weights[i]);
        free(r->d1_attention_weights[i]);
        free(r->d2_attention_weights[i]);
        free(r->ex_d_attention_weights_diff_grad[i]);
        free(r->score[i]);
        free(r->post_activation[i]);
        free(r->context_vector[i]);
    }
    free(r->attention_weights);
    free(r->d_attention_weights);
    free(r->d1_attention_weights);
    free(r->d2_attention_weights);
    free(r->ex_d_attention_weights_diff_grad);
    free(r->score);
    free(r->post_activation);
    free(r->context_vector);
    free(r);
}


/* This function create a new recurrent_enc_dec struct that is the same of the input
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ recurrent_enc_dec* r:= the encoder decoder struct
 * 
 * */
recurrent_enc_dec* copy_recurrent_enc_dec(recurrent_enc_dec* r){
    rmodel* encoder = copy_rmodel(r->encoder);
    rmodel* decoder = copy_rmodel(r->encoder);
    recurrent_enc_dec* r2 = recurrent_enc_dec_network(encoder,decoder);
    int i;
    for(i = 0; i < r->encoder->lstms[0]->size; i++){
        copy_array(r->attention_weights[i],r2->attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->d_attention_weights[i],r2->d_attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->d1_attention_weights[i],r2->d1_attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->d2_attention_weights[i],r2->d2_attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->ex_d_attention_weights_diff_grad[i],r2->ex_d_attention_weights_diff_grad[i],r->encoder->lstms[0]->size);
    }
    
    return r2;
}


/* Given 2 recurrent_enc_dec struct that have the same structure
 * in the second input is pasted the weights/biases of the first model
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ recurrent_enc_dec* r:= the first model
 *             @ recurrent_enc_dec* copy:= where is copied
 * */
void paste_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy){
    paste_rmodel(r->encoder,copy->encoder);
    paste_rmodel(r->decoder,copy->decoder);
    int i;
    for(i = 0; i < r->encoder->lstms[0]->size; i++){
        copy_array(r->attention_weights[i],copy->attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->d_attention_weights[i],copy->d_attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->d1_attention_weights[i],copy->d1_attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->d2_attention_weights[i],copy->d2_attention_weights[i],r->encoder->lstms[0]->size);
        copy_array(r->ex_d_attention_weights_diff_grad[i],copy->ex_d_attention_weights_diff_grad[i],r->encoder->lstms[0]->size);
    }
}


/* This function does the same of the paste_recurrent_enc_dec function but is slowed by a factor 1-tau
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ recurrent_enc_dec* r:= the first model
 *             @ recurrent_enc_dec* copy:= where is copied
 *             @ float tau:= the slowing factor
 * */
void slow_paste_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy, float tau){
    paste_rmodel(r->encoder,copy->encoder);
    paste_rmodel(r->decoder,copy->decoder);
    int i,j;
    for(i = 0; i < r->encoder->lstms[0]->size; i++){
        for(j = 0; j < r->encoder->lstms[0]->size; j++){
            copy->attention_weights[i][j] = r->attention_weights[i][j]*tau + (1-tau)*copy->attention_weights[i][j];
            copy->d_attention_weights[i][j] = tau*r->d_attention_weights[i][j] + (1-tau)*copy->d_attention_weights[i][j];
            copy->d1_attention_weights[i][j] = r->d1_attention_weights[i][j]*tau + (1-tau)*copy->d1_attention_weights[i][j];
            copy->d2_attention_weights[i][j] = tau*r->d2_attention_weights[i][j] + (1-tau)*copy->d2_attention_weights[i][j];
            copy->ex_d_attention_weights_diff_grad[i][j] = tau*r->ex_d_attention_weights_diff_grad[i][j] + (1-tau)*copy->ex_d_attention_weights_diff_grad[i][j];
        }
    }
}

/* this function resets the arrays needed for the feedforward and backpropagation of the model
 * 
 * 
 * Inputs:
 * 
 *                 @ recurrent_enc_dec* r := the recurrent encoder decoder structure
 * */
void reset_recurrent_enc_dec(recurrent_enc_dec* r){
    reset_rmodel(r->encoder);
    reset_rmodel(r->decoder);
    int i,j;
    for(i = 0; i < r->encoder->lstms[0]->size; i++){
        for(j = 0; j < r->encoder->lstms[0]->size; j++){
            r->d_attention_weights[i][j] = 0;
        }
        for(j = 0; j < r->encoder->lstms[0]->window; j++){
            r->score[i][j] = 0;
            r->post_activation[i][j] = 0;
            r->context_vector[i][j] = 0;
        }
    }
    
}


/* this function save in 3 files the recurrent enc dec structure
 * 
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ recurrent_enc_dec* r:= the model that must be saved
 *             @ int n1:= where the encoder of r is saved
 *             @ int n2:= where the decoder of r is saved
 *             @ int n3:= where the weights of r are saved
 * */
void save_recurrent_enc_dec(recurrent_enc_dec* r, int n1, int n2, int n3){
    if(r == NULL)
        return;
    save_rmodel(r->encoder,n1);
    save_rmodel(r->encoder,n2);
    int i,j;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa(n3,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    for(j = 0; j < r->encoder->lstms[0]->size; j++){
        i = fwrite(&r->attention_weights[i],sizeof(float)*r->encoder->lstms[0]->size,1,fw);
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the recurrent_enc_dec model\n");
            exit(1);
        }
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    free(s);
}

/* This function loads a recurrent_enc_dec structure given 3 files where it has been saved
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ char* file1:= where the encoder of the recurrent enc_dec structure has been saved
 *             @ char* file2:= where the decoder of the recurrent enc_dec structure has been saved
 *             @ char* file3:= where the weights of the recurrent enc_dec structure have been saved
 * */
recurrent_enc_dec* load_recurrent_enc_dec(char* file1, char* file2, char* file3){
    if(file3 == NULL)
        return NULL;
    rmodel* r1 = load_rmodel(file1);
    rmodel* r2 = load_rmodel(file2);
    int i,j;
    FILE* fr = fopen(file3,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",file3);
        exit(1);
    }
    
    float** d_w = (float**)malloc(sizeof(float*)*r1->lstms[0]->size);
    for(i = 0; i < r1->lstms[0]->size; i++){
        d_w[i] = (float*)calloc(r1->lstms[0]->size,sizeof(float));
    }

    
    for(j = 0; j < r1->lstms[0]->size; j++){
        i = fread(&d_w[i],sizeof(float)*r1->lstms[0]->size,1,fr);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading the recurrent_enc_dec model\n");
            exit(1);
        }
    }
    
    recurrent_enc_dec* r = recurrent_enc_dec_network(r1,r2);
    for(j = 0; j < r->encoder->lstms[0]->size; j++){
        copy_array(d_w[i],r->d_attention_weights[i],r->encoder->lstms[0]->size);
        free(d_w[i]);
    }
    free(d_w);
    
    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file3);
        exit(1);
    }
    
    return r;
    
}


