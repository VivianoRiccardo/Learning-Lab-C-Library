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
    int i;
    recurrent_enc_dec* r = (recurrent_enc_dec*)malloc(sizeof(recurrent_enc_dec));
    r->encoder = encoder;
    r->decoder = decoder;
    fcl** fcls = (fcl**)malloc(sizeof(fcl*));
    fcls[0] = fully_connected(encoder->lstms[0]->size*(encoder->window+1),encoder->lstms[0]->size*encoder->window,0,NO_DROPOUT,TANH,0);
    model** m = (model**)malloc(sizeof(model*)*decoder->window);
    m[0] = network(1,0,0,1,NULL,NULL,fcls);
    for(i = 1; i < decoder->window; i++){
		m[i] = copy_model(m[0]);
	}
    
    r->m = m;
    r->output_encoder = (float**)malloc(sizeof(float*)*encoder->window);
    r->output_error_encoder = (float**)malloc(sizeof(float)*encoder->window);
    r->softmax_array = (float*)calloc(encoder->window*encoder->lstms[0]->size,sizeof(float));
    r->context_array = (float*)calloc(encoder->window*encoder->lstms[0]->size,sizeof(float));
    for(i = 0; i < encoder->window; i++){
		r->output_encoder[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
		r->output_error_encoder[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
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
    int window1 = r->encoder->window;
    int window2 = r->decoder->window;
    free_rmodel(r->encoder);
    free_rmodel(r->decoder);
    int i;
    for(i = 0; i < window2; i++){
		free_model(r->m[i]);
	}
	free(r->m);
	free_matrix(r->output_encoder,window1);
	free_matrix(r->output_error_encoder,window1);
	free(r->softmax_array);
	free(r->context_array);
	free(r);
	return;
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
    for(i = 0; i < r->decoder->window; i++){
        paste_w_model(r->m[i],r2->m[i]);
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
    for(i = 0; i < r->decoder->window; i++){
        paste_model(r->m[i],copy->m[i]);
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
    slow_paste_rmodel(r->encoder,copy->encoder,tau);
    slow_paste_rmodel(r->decoder,copy->decoder,tau);
    int i,j;
    for(i = 0; i < r->decoder->window; i++){
        slow_paste_model(r->m[i],copy->m[i],tau);
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
    for(i = 0; i < r->decoder->window; i++){
        reset_model(r->m[i]);
    }
    for(i = 0; i < r->encoder->window; i++){
		for(j = 0; j < r->encoder->lstms[0]->size; j++){
			r->softmax_array[i*r->encoder->lstms[0]->size+j] = 0;
			r->context_array[i*r->encoder->lstms[0]->size+j] = 0;
			r->output_encoder[i][j] = 0;
			r->output_error_encoder[i][j] = 0;
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
    save_rmodel(r->decoder,n2);
    save_model(r->m[0],n3);
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
    model* m = load_model(file3);
    recurrent_enc_dec* r = recurrent_enc_dec_network(r1,r2);
    int i;
    for(i = 0; i < r->decoder->window; i++){
		paste_w_model(m,r->m[i]);
	}
	free_model(m);
	return r;
}


