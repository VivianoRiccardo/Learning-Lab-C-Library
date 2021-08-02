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

/* this function returns a transformer structure
 * 
 * Inputs:
 * 
 *             @ int n_te:= number of encoder transformers structures
 *             @ int n_td:= number of decoder transoformers structures
 *             @ transformer_encoder** te:= the transformer encoder structures
 *             @ transformer_decoder** td:= the transformer decoder structures
 *             @ int** encoder_decoder_connections:= a matrix of n_te*n_td size
 *                                                   it indicates where the output of each encoder must be
 *                                                   for example, let's imagine we have 3 encoder and 3 decoder
 *                                                   we want the first encoder output feeding the first decoder and the third one
 *                                                   the second encoder feeding the first one and the second one,
 *                                                   the last encoder to feed noone, then the matrix will be
 *                                                   [1,0,1,
 *                                                   [1,0,0,
 *                                                   [0,0,0]
 * 
 * remember, all the decoders must have a connection with at least 1 encoder
 * */
transformer* transf(int n_te, int n_td, transformer_encoder** te, transformer_decoder** td, int** encoder_decoder_connections){
    if(!n_te){
        fprintf(stderr,"Error, there must be at least 1 encoder!\n");
        exit(1);
    }
    if(n_te && te == NULL || n_td && td == NULL){
        fprintf(stderr,"Error, can't be n_te > 0 and te = NULL or n_td > 0 and td = NULL!\n");
        exit(1);
    }
    
    
    transformer* t = (transformer*)malloc(sizeof(transformer));
    t->n_te = n_te;
    t->n_td = n_td;
    t->te = te;
    t->td = td;
    t->encoder_decoder_connections = encoder_decoder_connections;
    t->beta1_adam = BETA1_ADAM;
    t->beta2_adam = BETA2_ADAM;
    t->beta3_adamod = BETA3_ADAMOD;
    return t;
}

/* this function deallocates the space allocated by a transformer structure including the matrix given as input
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer* t:= the transformer structure that must be deallocated
 * */
void free_transf(transformer* t){
    int i;
    for(i = 0; i < t->n_te; i++){
        free_transformer_encoder_layer(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        free_transformer_decoder_layer(t->td[i]);
    }
    free(t->te);
    free(t->td);
    for(i = 0; i < t->n_te; i++){
        free(t->encoder_decoder_connections[i]);
    }
    free(t->encoder_decoder_connections);
    free(t);
    return;
} 
/* this function deallocates the space allocated by a transformer structure including the matrix given as input
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer* t:= the transformer structure that must be deallocated
 * */
void free_transf_without_learning_parameters(transformer* t){
    int i;
    for(i = 0; i < t->n_te; i++){
        free_transformer_encoder_layer_without_learning_parameters(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        free_transformer_decoder_layer_without_learning_parameters(t->td[i]);
    }
    free(t->te);
    free(t->td);
    for(i = 0; i < t->n_te; i++){
        free(t->encoder_decoder_connections[i]);
    }
    free(t->encoder_decoder_connections);
    free(t);
    return;
} 






/* name is self-explanatory
 * 
 * Inputs:
 * 
 *             @ transformer* t:= the transformer that must be copied
 * */
transformer* copy_transf(transformer* t){
    if (t == NULL) return NULL;
    int i,j;
    transformer_encoder** te = NULL;
    transformer_decoder** td = NULL;
    int** enc_dec_con = NULL;
    if(t->n_te)
        te = (transformer_encoder**)malloc(sizeof(transformer_encoder*)*t->n_te);
    if(t->n_td)
        td = (transformer_decoder**)malloc(sizeof(transformer_decoder*)*t->n_td);
    for(i = 0; i < t->n_te; i++){
        te[i] = copy_transformer_encoder(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        td[i] = copy_transformer_decoder(t->td[i]);
    }
    
    if(t->encoder_decoder_connections != NULL && t->n_td){
        enc_dec_con = (int**)malloc(sizeof(int*)*t->n_te);
        for(i = 0; i < t->n_te; i++){
            enc_dec_con[i] = (int*)calloc(t->n_td,sizeof(int));
            for(j = 0; j < t->n_td; j++){
                enc_dec_con[i][j] = t->encoder_decoder_connections[i][j];
            }
        }
    }
    
    transformer* t2 = transf(t->n_te,t->n_td,te,td,enc_dec_con);
    t2->beta1_adam = t->beta1_adam; 
    t2->beta2_adam = t->beta2_adam;
    t2->beta3_adamod = t->beta3_adamod;
    return t2;
}
/* name is self-explanatory
 * 
 * Inputs:
 * 
 *             @ transformer* t:= the transformer that must be copied
 * */
transformer* copy_transf_without_learning_parameters(transformer* t){
    if (t == NULL) return NULL;
    int i,j;
    transformer_encoder** te = NULL;
    transformer_decoder** td = NULL;
    int** enc_dec_con = NULL;
    if(t->n_te)
        te = (transformer_encoder**)malloc(sizeof(transformer_encoder*)*t->n_te);
    if(t->n_td)
        td = (transformer_decoder**)malloc(sizeof(transformer_decoder*)*t->n_td);
    for(i = 0; i < t->n_te; i++){
        te[i] = copy_transformer_encoder_without_learning_parameters(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        td[i] = copy_transformer_decoder_without_learning_parameters(t->td[i]);
    }
    
    if(t->encoder_decoder_connections != NULL && t->n_td){
        enc_dec_con = (int**)malloc(sizeof(int*)*t->n_te);
        for(i = 0; i < t->n_te; i++){
            enc_dec_con[i] = (int*)calloc(t->n_td,sizeof(int));
            for(j = 0; j < t->n_td; j++){
                enc_dec_con[i][j] = t->encoder_decoder_connections[i][j];
            }
        }
    }
    
    transformer* t2 = transf(t->n_te,t->n_td,te,td,enc_dec_con);
    t2->beta1_adam = t->beta1_adam; 
    t2->beta2_adam = t->beta2_adam;
    t2->beta3_adamod = t->beta3_adamod;
    return t2;
}


/* paste a transformer into another
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer* t:= the transformer that must bhe copied
 *                 @ transformer* copy:= where the previous one is copied
 * */
void paste_transformer(transformer* t, transformer* copy){
    if(t == NULL || copy == NULL)
        return;
    int i;
    for(i = 0; i < t->n_te; i++){
        paste_transformer_encoder(t->te[i],copy->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        paste_transformer_decoder(t->td[i],copy->td[i]);
    }
    copy->beta1_adam = t->beta1_adam;
    copy->beta2_adam = t->beta2_adam;
    copy->beta3_adamod = t->beta3_adamod;
    return;
}

/* paste a transformer into another
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer* t:= the transformer that must bhe copied
 *                 @ transformer* copy:= where the previous one is copied
 * */
void paste_transformer_without_learning_parameters(transformer* t, transformer* copy){
    if(t == NULL || copy == NULL)
        return;
    int i;
    for(i = 0; i < t->n_te; i++){
        paste_transformer_encoder_without_learning_parameters(t->te[i],copy->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        paste_transformer_decoder_without_learning_parameters(t->td[i],copy->td[i]);
    }
    copy->beta1_adam = t->beta1_adam;
    copy->beta2_adam = t->beta2_adam;
    copy->beta3_adamod = t->beta3_adamod;
    return;
}


/* paste a transformer into another with a tau parameter
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer* t:= the transformer that must bhe copied
 *                 @ transformer* copy:= where the previous one is copied
 * */
void slow_paste_transformer(transformer* t, transformer* copy, float tau){
    if(t == NULL || copy == NULL)
        return;
    int i;
    for(i = 0; i < t->n_te; i++){
        slow_paste_transformer_encoder(t->te[i],copy->te[i],tau);
    }
    for(i = 0; i < t->n_td; i++){
        slow_paste_transformer_decoder(t->td[i],copy->td[i],tau);
    }
    copy->beta1_adam = t->beta1_adam;
    copy->beta2_adam = t->beta2_adam;
    copy->beta3_adamod = t->beta3_adamod;
    return;
}

/* this functions aves a transformer structure in a .bin file
 * 
 * Inputs:
 * 
 *             @ transformer* t:= the structure that must be saved
 *             @ int n:= the file name without the .bin part (is an integer that will be converted in "n.bin"
 * */
void save_transf(transformer* t, int n){
    if (t == NULL) return;
    int i,j;
    
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* tt = ".bin";
    s = itoa(n,s);
    s = strcat(s,tt);
    
    fw = fopen(s,"a+");
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    
    i = fwrite(&t->n_te,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer\n");
        exit(1);
    }
    
    i = fwrite(&t->n_td,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer\n");
        exit(1);
    }
    
    if(t->n_td && t->encoder_decoder_connections != NULL){
        for(j = 0; j < t->n_te; j++){
            i = fwrite(&t->encoder_decoder_connections[j],sizeof(int),t->n_td,fw);
            
            if(i != 1){
                fprintf(stderr,"Error: an error occurred saving a transformer\n");
                exit(1);
            }
        }
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    free(s);
    
    for(i = 0; i < t->n_te; i++){
        save_transformer_encoder(t->te[i],n);
    }
    for(i = 0; i < t->n_td; i++){
        save_transformer_decoder(t->td[i],n);
    }
    
    return;
}


/* this function load a transformer structure from a file
 * 
 * Inputs:
 * 
 * 
 *                 @ FILE* fr:= the file where the transformer is saved
 * */
transformer* load_transf(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i,j;
    int n_te = 0,n_td = 0;
    int** enc_dec_conn = NULL;
    transformer_encoder** te = NULL;
    transformer_decoder** td = NULL;
    i = fread(&n_te,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer\n");
        exit(1);
    }
    i = fread(&n_td,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer\n");
        exit(1);
    }
    
    if(n_td){
        enc_dec_conn = (int**)malloc(sizeof(int*));
        for(j = 0; j < n_te; j++){
            enc_dec_conn[j] =(int*)malloc(sizeof(int)*n_td);
            i = fread(enc_dec_conn[j],sizeof(int),n_td,fr);
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a transformer\n");
                exit(1);
            }
        } 
    }
    if(n_te)
        te = (transformer_encoder**)malloc(sizeof(transformer_encoder*)*n_te);
    if(n_td)
        td = (transformer_decoder**)malloc(sizeof(transformer_decoder*)*n_td);
    for(i = 0; i < n_te; i++){
        te[i] = load_transformer_encoder(fr);
    }
    for(i = 0; i < n_td; i++){
        td[i] = load_transformer_decoder(fr);
    }
    return transf(n_te,n_td,te,td,enc_dec_conn);
}


/* this function resetes all the arrays inside the transformer structure
 * that are used during the ff and bp of the training
 * 
 * Inputs:
 *             
 *             @ transformer* t:= the transformer structure that must be reset
 * */
void reset_transf(transformer* t){
    if(t == NULL) return;
    int i;
    for(i = 0; i < t->n_te; i++){
        reset_transformer_encoder(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        reset_transformer_decoder(t->td[i]);
    }
    return;
}
/* this function resetes all the arrays inside the transformer structure
 * that are used during the ff and bp of the training
 * 
 * Inputs:
 *             
 *             @ transformer* t:= the transformer structure that must be reset
 * */
void reset_transf_without_learning_parameters(transformer* t){
    if(t == NULL) return;
    int i;
    for(i = 0; i < t->n_te; i++){
        reset_transformer_encoder_without_learning_parameters(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        reset_transformer_decoder_without_learning_parameters(t->td[i]);
    }
    return;
}
/* this function resetes all the arrays inside the transformer structure
 * that are used during the ff and bp of the training
 * 
 * Inputs:
 *             
 *             @ transformer* t:= the transformer structure that must be reset
 * */
void reset_transf_decoders(transformer* t){
    if(t == NULL) return;
    int i;
    for(i = 0; i < t->n_td; i++){
        reset_transformer_decoder_except_partial_derivatives_and_left_input(t->td[i]);
    }
    return;
}

/* resets only the arrays for the edge popup
 * 
 * Inpus:
 * 
 *             @ transformer* t:= the transformer that must be rest
 * */
void reset_transf_for_edge_popup(transformer* t){
    if(t == NULL) return;
    int i;
    for(i = 0; i < t->n_te; i++){
        reset_transformer_encoder_for_edge_popup(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        reset_transformer_decoder_for_edge_popup(t->td[i]);
    }
    return;
}

/* this function gives an approximation of the space allocated by a transformer structure
 * 
 * Inputs:
 * 
 * 
 *             @ transformer* t:= the transformer which space is calcolated
 * */
uint64_t size_of_transformer(transformer* t){
    uint64_t sum = 0;
    int i;
    for(i = 0; i < t->n_te; i++){
        sum+=size_of_transformer_encoder(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        sum+=size_of_transformer_decoder(t->td[i]);
    }
    
    return sum;
}
/* this function gives an approximation of the space allocated by a transformer structure
 * 
 * Inputs:
 * 
 * 
 *             @ transformer* t:= the transformer which space is calcolated
 * */
uint64_t size_of_transformer_without_learning_parameters(transformer* t){
    uint64_t sum = 0;
    int i;
    for(i = 0; i < t->n_te; i++){
        sum+=size_of_transformer_encoder_without_learning_parameters(t->te[i]);
    }
    for(i = 0; i < t->n_td; i++){
        sum+=size_of_transformer_decoder_without_learning_parameters(t->td[i]);
    }
    
    return sum;
}

/* this function returns the output array of an encoder layer
 * 
 * Inputs:
 * 
 * 
 *             @ transformer_encoder* t:= the encoder which output must be returned
 * */
float* get_output_layer_from_encoder_transf(transformer_encoder* t){
    if(t->normalization_flag2 != NO_NORMALIZATION)
        return t->l2[t->n_l2-1]->output;
    else if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
        return t->residual2_output;
    else
        return t->m->output_layer;
}

/* this function computes the feed forward of the transformer
 * 
 * Inputs:
 * 
 *             @ transformer* t:= the structure that must compute the feed forward
 *             @ float* inputs_encoder:= the inputs for the first encoder layer of t
 *             @ int_input_dimension1:= the dimension of inputs_encoder
 *             @ float* inputs_decoder:= the inputs for the first decoder layer
 *             @ int input_dimension2:= the dimension of inputs_decoder
 *                @ int flag:= if set to RUN_ONLY_DECODER it runs only the decoder, else if it is set to RUN_ALL_TRANSF both encoder and decoder executes the ff
 * */
void transf_ff(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, int flag){
    int i, in = input_dimension1,j,k;
    float* temp = inputs_encoder;
    if(flag == RUN_ALL_TRANSF){
        for(i = 0; i < t->n_te; i++){
            encoder_transformer_ff(temp,t->te[i],in);
            temp = get_output_layer_from_encoder_transf(t->te[i]);
            in = t->te[i]->m->output_dimension;
            
            for(j = 0; j < t->n_td; j++){
                int c = 0;
                if(t->encoder_decoder_connections[i][j]){
                    for(k = 0; k < i; k++){
                        if(t->encoder_decoder_connections[k][j])
                            c += t->te[k]->m->output_dimension;
                    }
                    memcpy(&t->td[j]->incoming_input[c],get_output_layer_from_encoder_transf(t->te[i]),sizeof(float)*t->te[i]->m->output_dimension);
                }
            }
        }
    }    
    in = input_dimension2;
    temp = inputs_decoder;
    for(i = 0; i < t->n_td; i++){
        int c = 0;
        for(j = 0; j < t->n_te; j++){
            if (t->encoder_decoder_connections[j][i]){
                c += t->te[j]->m->output_dimension;
            }
        }
        decoder_transformer_ff(temp,t->td[i]->incoming_input,t->td[i],in,c);
        temp = get_output_layer_from_encoder_transf(t->td[i]->e);
        in = t->td[i]->e->m->output_dimension;
    }
    
    return;
}
/* this function computes the feed forward of the transformer
 * 
 * Inputs:
 * 
 *             @ transformer* t:= the structure that must compute the feed forward
 *             @ float* inputs_encoder:= the inputs for the first encoder layer of t
 *             @ int_input_dimension1:= the dimension of inputs_encoder
 *             @ float* inputs_decoder:= the inputs for the first decoder layer
 *             @ int input_dimension2:= the dimension of inputs_decoder
 *                @ int flag:= if set to RUN_ONLY_DECODER it runs only the decoder, else if it is set to RUN_ALL_TRANSF both encoder and decoder executes the ff
 * */
void transf_ff_opt(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, int flag, transformer* t2){
    int i, in = input_dimension1,j,k;
    float* temp = inputs_encoder;
    if(flag == RUN_ALL_TRANSF){
        for(i = 0; i < t->n_te; i++){
            encoder_transformer_ff_opt(temp,t->te[i],in,t2->te[i]);
            temp = get_output_layer_from_encoder_transf(t->te[i]);
            in = t->te[i]->m->output_dimension;
            
            for(j = 0; j < t->n_td; j++){
                int c = 0;
                if(t->encoder_decoder_connections[i][j]){
                    for(k = 0; k < i; k++){
                        if(t->encoder_decoder_connections[k][j])
                            c += t->te[k]->m->output_dimension;
                    }
                    memcpy(&t->td[j]->incoming_input[c],get_output_layer_from_encoder_transf(t->te[i]),sizeof(float)*t->te[i]->m->output_dimension);
                }
            }
        }
    }    
    in = input_dimension2;
    temp = inputs_decoder;
    for(i = 0; i < t->n_td; i++){
        int c = 0;
        for(j = 0; j < t->n_te; j++){
            if (t->encoder_decoder_connections[j][i]){
                c += t->te[j]->m->output_dimension;
            }
        }
        decoder_transformer_ff_opt(temp,t->td[i]->incoming_input,t->td[i],in,c,t->td[i]);
        temp = get_output_layer_from_encoder_transf(t->td[i]->e);
        in = t->td[i]->e->m->output_dimension;
    }
    
    return;
}


/* this function computes the bp passage for the transformer
 * 
 * Inputs:
 * 
 * 
 *             @ transformer* t:= the transformer that must compute the bp passage
 *             @ float* inputs_encoder:= the inputs given to the first encoder, dimension: input_dimension1
 *             @ float* input_dimension1:= the dimension of inputs_encoder
 *             @ float* inputs_decoder:= the inputs given to the first decoder, dimension: input_dimension2
 *             @ int input_dimension2:= the dimension of inputs_decoder
 *             @ float* output_error:= the error of the last decoder (if you want to add error to the last encoder add it to t->te[t->n_te-1]->encoder_output_error
 *                @ int flag:= if set to RUN_ONLY_DECODER it runs only the decoder, else if it is set to RUN_ALL_TRANSF both encoder and decoder executes the bp
 * */
float* transf_bp(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, float* output_error, int flag){
    int i,j,k,in;
    float* temp1 = output_error;
    float* temp2 = inputs_decoder;
    
    if(flag != RUN_ONLY_ENCODER){
        for(i = t->n_td-1; i >-1; i--){
            int c = 0;
            if(i){
                temp2 = get_output_layer_from_encoder_transf(t->td[i-1]->e);
                in = t->td[i-1]->e->m->output_dimension;
            }
            
            else{
                temp2 = inputs_decoder;
                in = input_dimension2;
            }
            for(j = 0; j < t->n_te; j++){
                if (t->encoder_decoder_connections[j][i]){
                    c+=t->te[j]->m->output_dimension;
                }
            }
            temp1 = decoder_transformer_bp(temp2,t->td[i]->incoming_input,t->td[i],in,c,temp1,t->td[i]->incoming_input_error);
        }
        
        for(i = t->n_te-1; i > -1; i--){
            
            for(j = 0; j < t->n_td; j++){
                int c = 0;
                if(t->encoder_decoder_connections[i][j]){
                    for(k = 0; k < i; k++){
                        if(t->encoder_decoder_connections[k][j])
                            c += t->te[k]->m->output_dimension;
                    }
                    sum1D(&t->td[j]->incoming_input_error[c],t->te[i]->encoder_output_error,t->te[i]->encoder_output_error,t->te[i]->m->output_dimension);
                }
            }
        }
    }
    if(flag == RUN_ALL_TRANSF){
        for(i = t->n_te-1; i > -1; i--){    
            if(i){
                temp2 = get_output_layer_from_encoder_transf(t->te[i-1]);
                in = t->te[i-1]->m->output_dimension;
            }
            
            else{
                temp2 = inputs_encoder;
                in = input_dimension1;
            }
            
            if(i < t->n_te-1){
                sum1D(t->te[i]->encoder_output_error,temp1,t->te[i]->encoder_output_error,t->te[i]->m->output_dimension);
            }
            temp1 = encoder_transformer_bp(temp2,t->te[i],in,t->te[i]->encoder_output_error);
        }
    }
    
    return temp1;
    
}
/* this function computes the bp passage for the transformer
 * 
 * Inputs:
 * 
 * 
 *             @ transformer* t:= the transformer that must compute the bp passage
 *             @ float* inputs_encoder:= the inputs given to the first encoder, dimension: input_dimension1
 *             @ float* input_dimension1:= the dimension of inputs_encoder
 *             @ float* inputs_decoder:= the inputs given to the first decoder, dimension: input_dimension2
 *             @ int input_dimension2:= the dimension of inputs_decoder
 *             @ float* output_error:= the error of the last decoder (if you want to add error to the last encoder add it to t->te[t->n_te-1]->encoder_output_error
 *                @ int flag:= if set to RUN_ONLY_DECODER it runs only the decoder, else if it is set to RUN_ALL_TRANSF both encoder and decoder executes the bp
 * */
float* transf_bp_opt(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, float* output_error, int flag, transformer* t2){
    int i,j,k,in;
    float* temp1 = output_error;
    float* temp2 = inputs_decoder;
    
    if(flag != RUN_ONLY_ENCODER){
        for(i = t->n_td-1; i >-1; i--){
            int c = 0;
            if(i){
                temp2 = get_output_layer_from_encoder_transf(t->td[i-1]->e);
                in = t->td[i-1]->e->m->output_dimension;
            }
            
            else{
                temp2 = inputs_decoder;
                in = input_dimension2;
            }
            for(j = 0; j < t->n_te; j++){
                if (t->encoder_decoder_connections[j][i]){
                    c+=t->te[j]->m->output_dimension;
                }
            }
            temp1 = decoder_transformer_bp_opt(temp2,t->td[i]->incoming_input,t->td[i],in,c,temp1,t->td[i]->incoming_input_error,t2->td[i]);
        }
        
        for(i = t->n_te-1; i > -1; i--){
            
            for(j = 0; j < t->n_td; j++){
                int c = 0;
                if(t->encoder_decoder_connections[i][j]){
                    for(k = 0; k < i; k++){
                        if(t->encoder_decoder_connections[k][j])
                            c += t->te[k]->m->output_dimension;
                    }
                    sum1D(&t->td[j]->incoming_input_error[c],t->te[i]->encoder_output_error,t->te[i]->encoder_output_error,t->te[i]->m->output_dimension);
                }
            }
        }
    }
    if(flag == RUN_ALL_TRANSF){
        for(i = t->n_te-1; i > -1; i--){    
            if(i){
                temp2 = get_output_layer_from_encoder_transf(t->te[i-1]);
                in = t->te[i-1]->m->output_dimension;
            }
            
            else{
                temp2 = inputs_encoder;
                in = input_dimension1;
            }
            
            if(i < t->n_te-1){
                sum1D(t->te[i]->encoder_output_error,temp1,t->te[i]->encoder_output_error,t->te[i]->m->output_dimension);
            }
            temp1 = encoder_transformer_bp_opt(temp2,t->te[i],in,t->te[i]->encoder_output_error,t2->te[i]);
        }
    }
    
    return temp1;
    
}



