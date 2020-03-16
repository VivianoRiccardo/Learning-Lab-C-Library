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
 * 
 * Pay attention if your input size of the decoder is [input_size] then your decoder
 * size should be encoder size + decoder size
 * */
recurrent_enc_dec* recurrent_enc_dec_network(rmodel* encoder, rmodel* decoder){
    int i;
    if(encoder == NULL || decoder == NULL)
        return NULL;
    if(decoder->lstms[decoder->n_lstm-1]->dropout_flag_right || decoder->lstms[decoder->n_lstm-1]->dropout_flag_up || encoder->lstms[encoder->n_lstm-1]->dropout_flag_right || encoder->lstms[encoder->n_lstm-1]->dropout_flag_up){
        fprintf(stderr,"Error: you have to avoid right and up dropout for last lstm for decoder and encoder!\n");
        exit(1);
    }
    
    if(encoder->lstms[encoder->n_lstm-1]->residual_flag == LSTM_RESIDUAL || decoder->lstms[decoder->n_lstm-1]->residual_flag == LSTM_RESIDUAL){
        fprintf(stderr,"Error: is useless the residual flag for last lstm of the encoder, and to simplify functions no residual for last lstm for decoder!\n");
        exit(1);
    }
    
    if(encoder->lstms[encoder->n_lstm-1]->residual_flag == GROUP_NORMALIZATION){
        fprintf(stderr,"Error: is useless the normalization for last lstm of the encoder!\n");
        exit(1);
    }
    
    if(decoder->n_lstm != encoder->n_lstm){
        fprintf(stderr,"Error: your encoder and your decoder must have same number of lstm!\n");
        exit(1);
    }
    
    if(decoder->lstms[0]->size < encoder->lstms[0]->size){
        fprintf(stderr,"Error: the decoder size must be >= of encoder size\n");
        exit(1);
    }
    recurrent_enc_dec* r = (recurrent_enc_dec*)malloc(sizeof(recurrent_enc_dec));
    r->encoder = encoder;
    r->decoder = decoder;
    fcl** fcls = (fcl**)malloc(sizeof(fcl*));
    fcls[0] = fully_connected(encoder->lstms[0]->size*(encoder->window+1),encoder->window,0,NO_DROPOUT,TANH,0);
    model** m = (model**)malloc(sizeof(model*)*decoder->window);
    m[0] = network(1,0,0,1,NULL,NULL,fcls);
    for(i = 1; i < decoder->window; i++){
        m[i] = copy_model(m[0]);
    }
    
    
    r->m = m;
    r->flatten_fcl_input = (float*)malloc(sizeof(float)*(encoder->window+1)*encoder->lstms[0]->size);
    r->output_encoder = (float**)malloc(sizeof(float*)*encoder->window);
    r->hiddens = (float**)malloc(sizeof(float*)*decoder->window);
    r->output_error_encoder = (float**)malloc(sizeof(float)*encoder->window);
    r->softmax_array = (float**)calloc(decoder->window,sizeof(float));

    for(i = 0; i < encoder->window; i++){
        r->output_encoder[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
        r->output_error_encoder[i] = (float*)calloc(decoder->lstms[0]->size,sizeof(float));
    }
    
    for(i = 0; i < decoder->window; i++){
        r->hiddens[i] = (float*)calloc(encoder->lstms[0]->size,sizeof(float));
        r->softmax_array[i] = (float*)calloc(encoder->window,sizeof(float));
    }
    
    r->beta1_adam = BETA1_ADAM;
    r->beta2_adam = BETA2_ADAM;
    r->beta3_adamod = BETA3_ADAMOD;
    r->encoder->beta1_adam = r->beta1_adam;
    r->encoder->beta2_adam = r->beta2_adam;
    r->encoder->beta3_adamod = r->beta3_adamod;
    r->decoder->beta1_adam = r->beta1_adam;
    r->decoder->beta2_adam = r->beta2_adam;
    r->decoder->beta3_adamod = r->beta3_adamod;
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
        free(r->hiddens[i]);
    }
    free(r->hiddens);
    free(r->m);
    free_matrix(r->output_encoder,window1);
    free_matrix(r->output_error_encoder,window1);
    free_matrix(r->softmax_array,window2);
    free(r->flatten_fcl_input);
    free(r);
    return;
}


/* This function creates a new recurrent_enc_dec struct that is the same of the input
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
    
    r2->beta1_adam = r->beta1_adam;
    r2->beta2_adam = r->beta2_adam;
    r2->beta3_adamod = r->beta3_adamod;
    
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
void paste_w_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy){
    paste_w_rmodel(r->encoder,copy->encoder);
    paste_w_rmodel(r->decoder,copy->decoder);
    int i;
    for(i = 0; i < r->decoder->window; i++){
        paste_w_model(r->m[i],copy->m[i]);
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
            r->output_encoder[i][j] = 0;
            r->output_error_encoder[i][j] = 0;
        }
    }
    
    for(i = 0; i < r->decoder->window; i++){
        for(j = 0; j < r->encoder->lstms[0]->size; j++){
            r->hiddens[i][j] = 0;
        }
        
        for(j = 0; j < r->encoder->window; j++){
            r->softmax_array[i][j] = 0;
        }
    }
    
}


/* this function saves in 3 files the recurrent enc dec structure
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


/* this function saves in 3 files the recurrent enc dec structure
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
void heavy_save_recurrent_enc_dec(recurrent_enc_dec* r, int n1, int n2, int n3){
    if(r == NULL)
        return;
    heavy_save_rmodel(r->encoder,n1);
    heavy_save_rmodel(r->decoder,n2);
    heavy_save_model(r->m[0],n3);
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
        paste_model(m,r->m[i]);
    }
    free_model(m);
    return r;
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
recurrent_enc_dec* heavy_load_recurrent_enc_dec(char* file1, char* file2, char* file3){
    if(file3 == NULL)
        return NULL;
    rmodel* r1 = heavy_load_rmodel(file1);
    rmodel* r2 = heavy_load_rmodel(file2);
    model* m = heavy_load_model(file3);
    recurrent_enc_dec* r = recurrent_enc_dec_network(r1,r2);
    int i;
    for(i = 0; i < r->decoder->window; i++){
        paste_model(m,r->m[i]);
    }
    free_model(m);
    return r;
}


int count_weights_recurrent_enc_dec(recurrent_enc_dec* m){
    int sum = 0;
    sum+=count_weights_rmodel(m->encoder);
    sum+=count_weights_rmodel(m->decoder);
    return sum + count_weights(m->m[0]);
}

/* Given a encoder_decoder with only long short term memory cells, this functions computes the feed forward
 * for that model given in input the previous hidden state, previous cell state the input model
 * 
 * Input:
 * 
 *             @ float** hidden_state:= the hidden states of the previous cells:= layers x size (they are padded from encoder)
 *             @ float** cell_states:= layers x size
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window) x (decoder->size), pay attention the inputs should be of the size of the decoder-size of encoder they have been shifted
 *             @ int window:= the window of the unrolled cells
 *                @ int size:= the size of the lstms cells
 *                @ int layers:= the number of lstms cells
 *               @ lstm** lstms:= the lstms cells
 *             @ recurrent_enc_dec* rec:= the recurrent encoder decoder structure
 * 
 * 
 * */
void ff_decoder_lstm(float** hidden_states, float** cell_states, float** input_model, int window, int size, int layers, lstm** lstms, recurrent_enc_dec* rec){    
    
    //allocation of the resources

    float* dropout_output = (float*)malloc(sizeof(float)*lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell in vertical
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell in orizontal
    
    int i,j,k;
    
    float temp_drp_value;
    /*feed_forward_passage*/
    
    for(i = 0; i < window; i++){
        for(j = 0; j < layers; j++){
            
            if(j == 0){ //j = 0 means that we are at the first lstm_cell in vertical
                if(i == 0){//i = 0 means we are at the first lstm in orizontal
                    //in this case the h-1 and c-1 come from the last mini_batch
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->dropout_threshold_right);
                   
                    get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->size);
                    
                    copy_array(rec->encoder->lstms[rec->encoder->n_lstm-1]->lstm_hidden[rec->encoder->window-1],rec->hiddens[i],rec->encoder->lstms[0]->size);
                    copy_array(rec->encoder->lstms[rec->encoder->n_lstm-1]->lstm_hidden[rec->encoder->window-1],&rec->flatten_fcl_input[rec->encoder->window*rec->encoder->lstms[0]->size],rec->encoder->lstms[0]->size);
                    model_tensor_input_ff(rec->m[i],rec->encoder->lstms[0]->size*(rec->encoder->window+1),1,1,rec->flatten_fcl_input);
                    softmax(rec->m[i]->fcls[0]->post_activation,rec->softmax_array[i],rec->encoder->window);
                    
                    for(k = 0; k < rec->encoder->window; k++){
                        float* temp_prod = (float*)calloc(rec->encoder->lstms[0]->size,sizeof(float));
                        mul_value(&rec->flatten_fcl_input[k*rec->encoder->lstms[0]->size],rec->softmax_array[i][k],temp_prod,rec->encoder->lstms[0]->size);
                        sum1D(input_model[i],temp_prod,input_model[i],rec->encoder->lstms[0]->size);
                        free(temp_prod);
                    }
                    
                    lstm_ff(input_model[i], dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->size);
                }

                else{
                    
                    get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->size);
                    
                    
                    copy_array(dropout_output,rec->hiddens[i],rec->encoder->lstms[0]->size);
                    copy_array(dropout_output,&rec->flatten_fcl_input[rec->encoder->window*rec->encoder->lstms[0]->size],rec->encoder->lstms[0]->size);
                    model_tensor_input_ff(rec->m[i],rec->encoder->lstms[0]->size*(rec->encoder->window+1),1,1,rec->flatten_fcl_input);
                    softmax(rec->m[i]->fcls[0]->post_activation,rec->softmax_array[i],rec->encoder->window);
                    
                    for(k = 0; k < rec->encoder->window; k++){
                        float* temp_prod = (float*)calloc(rec->encoder->lstms[0]->size,sizeof(float));
                        mul_value(&rec->flatten_fcl_input[k*rec->encoder->lstms[0]->size],rec->softmax_array[i][k],temp_prod,rec->encoder->lstms[0]->size);
                        sum1D(input_model[i],temp_prod,input_model[i],rec->encoder->lstms[0]->size);
                        free(temp_prod);
                    }
                   
                    lstm_ff(input_model[i], dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->size);
                }
            }
            
            else{
                
                if(i == 0){//i = 0 and j != 0 means that we are at the first lstm in orizontal but not in vertical
                    if(lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->dropout_threshold_right);
                   
                    get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->size);
                        
                    lstm_ff(dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->size);
                    
                }    
                else{
                    
                    
                    get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->size);
                        
                    lstm_ff(dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->size);
                }
            }
            
            /* the dropout is applied to each lstm_hidden to feed the deeper lstm cell in vertical, as input*/
            if(i == 0)
                if(lstms[j]->dropout_flag_up == DROPOUT)
                    set_dropout_mask(lstms[j]->size,lstms[j]->dropout_mask_up,lstms[j]->dropout_threshold_up);
            
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_up,lstms[j]->lstm_hidden[i],dropout_output);
            
            if(lstms[j]->dropout_flag_up == DROPOUT_TEST)
                mul_value(dropout_output,lstms[j]->dropout_threshold_up,dropout_output,lstms[j]->size);
            
            if(!j && lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dropout_output,input_model[i],dropout_output,lstms[j]->size);
            else if(j && lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dropout_output,lstms[j-1]->out_up[i],dropout_output,lstms[j]->size);
                
            copy_array(dropout_output,lstms[j]->out_up[i],lstms[j]->size);
        }
    }
    
    free(dropout_output);
    free(dropout_output2);
    
}

/* this function returns the error dfioc and set the error of input_error through a back propagation passage
 * the dfioc returning error has this dimensions: m->layers*4*m->size (for the decoder)
 * 
 * Inputs:
 * 
 * 
 *             @ float** hidden_states:= the hidden sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** cell_states:= the cell sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** input_model:= the input passed to the model, dimensions: m->window*m->size
 *             @ float** error_model:= the error of the model, dimensions: m->window*m->size
 *             @ int window:= the window of the lstms cells
 *                @ int size:= the size of lstms cells
 *                @ int layers:= the number of lastms cells
 *                @ lstm** lstms:= the lstms cells
 *             @ float** input_error:= the error of the inputs of this model, dimensions: m->window*m->size, must be initialized only with m->window, in this case should be always initialized and not set to null
 *              @ recurrent_enc_dec* rec:= the recurrent encoder decoder structure
 * 
 * */
float*** bp_decoder_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window, int size,int layers,lstm** lstms, float** input_error, recurrent_enc_dec* rec){

   /* backpropagation passage*/

    int i,j,k,z,lstm_bp_flag;
    
    float* dropout_output = (float*)malloc(sizeof(float)*lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[0]->size);
    float* dx; //here we store the modified output by dropout coming from the last lstm cell
    float* dz = (float*)calloc(lstms[0]->size,sizeof(float)); //for residual dx
    float*** matrix = (float***)malloc(sizeof(float**)*layers);
    float** temp;
    float* array_now ;
    
    for(i = 0; i < layers; i++){
        matrix[i] = NULL;
    }
    
    /*with i = 0 we should handle the h minus and c minus that are given by the params passed to the function*/
    for(i = window-1; i > 0; i--){
        for(j = layers-1; j >= 0; j--){
            
            dx = (float*)calloc(lstms[0]->size,sizeof(float));
            if(i < window-1 && j == layers-1)
                sum1D(dx,&array_now[rec->encoder->window*rec->encoder->lstms[0]->size],dx,rec->encoder->lstms[0]->size);
            
            if(lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dx,dz,dx,lstms[0]->size);
                
            if(j == layers-1 && i == window-1)
                lstm_bp_flag = 0;
            else if(j != layers-1 && i == window-1)
                lstm_bp_flag = 1;
                
            else if(j == layers-1 && i != window-1)
                lstm_bp_flag = 2;
                
            else
                lstm_bp_flag = 3;
            
            
            if(j == layers-1)
                
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_up,error_model[i],dx);

            if(j == layers-1){
                
                if(j != 0)
                    get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);
                
                
                if(j != 0){
                    if(i == window-1)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size,  lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                }
                
                else{
                    if(i == window-1)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size,  lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    
                }
                
                
                if(matrix[j] != NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;

            }
            
            else if(j != layers-1 && j != 0){
                
                
                get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);

                
                if(i == window-1)
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    
                else
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                
                matrix[j] = temp;
                
                
            }
            
            else{
                
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);

                if(i == window-1)
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                
                else
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j+1]->lstm_z[i],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
            }
            
            copy_array(dx,dz,lstms[0]->size);
            free(dx);
            
            if(!j && input_error != NULL){
                input_error[i] = lstm_dinput(i,lstms[j]->size,matrix[j],lstms[j]);
                
                float* temp_prod2 = (float*)calloc(rec->encoder->window,sizeof(float));
                float* temp_prod3 = (float*)calloc(rec->encoder->window,sizeof(float));
                for(k = 0; k < rec->encoder->window; k++){
                    float* temp_prod = (float*)calloc(rec->encoder->lstms[0]->size,sizeof(float));
                    mul_value(input_error[i],rec->softmax_array[i][k],temp_prod,rec->encoder->lstms[0]->size);
                    sum1D(rec->output_error_encoder[k],temp_prod,rec->output_error_encoder[k],rec->encoder->lstms[0]->size);
                    free(temp_prod);
                    
                    for(z = 0; z < rec->encoder->lstms[0]->size; z++){
                        temp_prod2[k] += rec->flatten_fcl_input[k*rec->encoder->lstms[0]->size+z]*input_error[i][z];
                    }
                }
                derivative_softmax_array(NULL,temp_prod3,rec->softmax_array[i],temp_prod2,rec->encoder->window);
                copy_array(lstms[layers-1]->lstm_hidden[i-1],&rec->flatten_fcl_input[rec->encoder->window*rec->encoder->lstms[0]->size],rec->encoder->lstms[0]->size);
                array_now = model_tensor_input_bp(rec->m[i],(rec->encoder->window+1)*rec->encoder->lstms[0]->size,1,1,rec->flatten_fcl_input,temp_prod3,rec->encoder->window);
                free(temp_prod2);
                free(temp_prod3);
                for(k = 0; k < rec->encoder->window; k++){
                    sum1D(&array_now[k*rec->encoder->lstms[0]->size],rec->output_error_encoder[k],rec->output_error_encoder[k],rec->encoder->lstms[0]->size);
                }
                
            }
            
        }
        
    }
    
    i = 0;
    /* computing back propagation just for the first lstm layers with hidden states defined by the previous batch*/
    for(j = layers-1; j >= 0; j--){
            
        dx = (float*)calloc(lstms[0]->size,sizeof(float));
        
        if(i < window-1 && j == layers-1)
            sum1D(dx,&array_now[rec->encoder->window*rec->encoder->lstms[0]->size],dx,rec->encoder->lstms[0]->size);
        
        if(lstms[j]->residual_flag == LSTM_RESIDUAL)
            sum1D(dx,dz,dx,lstms[0]->size);
        if(j == layers-1 && i == window-1)
            lstm_bp_flag = 0;
        else if(j != layers-1 && i == window-1)
            lstm_bp_flag = 1;
            
        else if(j == layers-1 && i != window-1)
            lstm_bp_flag = 2;
            
        else
            lstm_bp_flag = 3;
        
        
        if(j == layers-1)
            
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_up,error_model[i],dx);

        if(j == layers-1){
            
            if(j != 0)
                get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(j != 0)
                temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,dropout_output,lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            else
                temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        
        else if(j != layers-1 && j != 0){
            
            get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            

            temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, dropout_output,lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
        }
        
        else{
            
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->d_u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        copy_array(dx,dz,lstms[0]->size);
        free(dx);
        
        if(!j && input_error != NULL){
                input_error[i] = lstm_dinput(i,lstms[j]->size,matrix[j],lstms[j]);
                float* temp_prod2 = (float*)calloc(rec->encoder->window,sizeof(float));
                float* temp_prod3 = (float*)calloc(rec->encoder->window,sizeof(float));
                for(k = 0; k < rec->encoder->window; k++){
                    float* temp_prod = (float*)calloc(rec->encoder->lstms[0]->size,sizeof(float));
                    mul_value(input_error[i],rec->softmax_array[i][k],temp_prod,rec->encoder->lstms[0]->size);
                    sum1D(rec->output_error_encoder[k],temp_prod,rec->output_error_encoder[k],rec->encoder->lstms[0]->size);
                    free(temp_prod);
                    
                    for(z = 0; z < rec->encoder->lstms[0]->size; z++){
                        temp_prod2[k] += rec->flatten_fcl_input[k*rec->encoder->lstms[0]->size+z]*input_error[i][z];
                    }
                }
                derivative_softmax_array(NULL,temp_prod3,rec->softmax_array[i],temp_prod2,rec->encoder->window);
                copy_array(lstms[layers-1]->lstm_hidden[i-1],&rec->flatten_fcl_input[rec->encoder->window*rec->encoder->lstms[0]->size],rec->encoder->lstms[0]->size);
                array_now = model_tensor_input_bp(rec->m[i],(rec->encoder->window+1)*rec->encoder->lstms[0]->size,1,1,rec->flatten_fcl_input,temp_prod3,rec->encoder->window);
                free(temp_prod2);
                free(temp_prod3);
                for(k = 0; k < rec->encoder->window; k++){
                    sum1D(&array_now[k*rec->encoder->lstms[0]->size],rec->output_error_encoder[k],rec->output_error_encoder[k],rec->encoder->lstms[0]->size);
                }
        }
    }
    
    free(dropout_output);
    free(dropout_output2);
    free(dz);
    return matrix;
    
}

/* this function returns the error dfioc and set the error of input_error through a back propagation passage
 * the dfioc returning error has this dimensions: m->layers*4*m->size (for the encoder)
 * 
 * Inputs:
 * 
 * 
 *             @ float** hidden_states:= the hidden sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** cell_states:= the cell sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** input_model:= the input passed to the model, dimensions: m->window*m->size
 *             @ float** error_model:= the error of the model, dimensions: m->window*m->size
 *             @ int window:= the window of the lstms cells
 *                @ int size:= the size of lstms cells
 *                @ int layers:= the number of lastms cells
 *                @ lstm** lstms:= the lstms cells
 *             @ float** input_error:= the error of the inputs of this model, dimensions: m->window*m->size, must be initialized only with m->window
 *             @ float** dfioc:= the dfioc given from the decoder
 *             @ float** dropout_mask_dec:= the dropout mask right given by the decoder
 *             @ lstm** first_dec_orizontal:= the lstms of the decoder
 * 
 * */
float*** bp_encoder_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window, int size,int layers,lstm** lstms, float** input_error, float*** dfioc, float** dropout_mask_dec, lstm** first_dec_orizontal){

   /* backpropagation passage*/

    int i,j, lstm_bp_flag;
    
    float* dropout_output = (float*)malloc(sizeof(float)*lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[0]->size);
    float* dx; //here we store the modified output by dropout coming from the last lstm cell
    float* dz = (float*)calloc(lstms[0]->size,sizeof(float)); //for residual dx
    float*** matrix = (float***)malloc(sizeof(float**)*layers);
    float** temp;
    
    for(i = 0; i < layers; i++){
        matrix[i] = NULL;
    }
    /*with i = 0 we should handle the h minus and c minus that are given by the params passed to the function*/
    for(i = window-1; i > 0; i--){
        for(j = layers-1; j >= 0; j--){
            
            dx = (float*)calloc(lstms[0]->size,sizeof(float));
            if(lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dx,dz,dx,lstms[0]->size);
                
            if(j == layers-1 && i == window-1)
                lstm_bp_flag = 2;
            else if(j != layers-1 && i == window-1)
                lstm_bp_flag = 3;
                
            else if(j == layers-1 && i != window-1)
                lstm_bp_flag = 2;
                
            else
                lstm_bp_flag = 3;
            
            
            if(j == layers-1)
                
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_up,error_model[i],dx);

            if(j == layers-1){
                
                if(j != 0)
                    get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
                
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);
                
                if(j != 0){
                    if(i == window-1)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, first_dec_orizontal[j]->lstm_z[0], dfioc[j],NULL,lstms[j]->dropout_mask_up,dropout_mask_dec[j]);
                    else
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size,  lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                }
                
                else{
                    if(i == window-1)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, first_dec_orizontal[j]->lstm_z[0], dfioc[j],NULL,lstms[j]->dropout_mask_up,dropout_mask_dec[j]);
                    else
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->size,  lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    
                }
                
                
                if(matrix[j] != NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;

            }
            
            else if(j != layers-1 && j != 0){
                
                
                get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);
                
                if(i == window-1)
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], first_dec_orizontal[j]->lstm_z[0],dfioc[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,dropout_mask_dec[j]);
                    
                else
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, dropout_output,lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
                
            }
            
            else{
                
                get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);
                

                if(i == window-1)
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], first_dec_orizontal[j]->lstm_z[0],dfioc[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,dropout_mask_dec[j]);
                
                else
                    temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
            }
            
            copy_array(dx,dz,lstms[0]->size);
            free(dx);
            
            if(!j && input_error != NULL)
                input_error[i] = lstm_dinput(i,lstms[j]->size,matrix[j],lstms[j]);
            
        }
        
    }
    
    i = 0;
    /* computing back propagation just for the first lstm layers with hidden states defined by the previous batch*/
    for(j = layers-1; j >= 0; j--){
            
        dx = (float*)calloc(lstms[0]->size,sizeof(float));
        if(lstms[j]->residual_flag == LSTM_RESIDUAL)
            sum1D(dx,dz,dx,lstms[0]->size);
        if(j == layers-1 && i == window-1)
            lstm_bp_flag = 0;
        else if(j != layers-1 && i == window-1)
            lstm_bp_flag = 1;
            
        else if(j == layers-1 && i != window-1)
            lstm_bp_flag = 2;
            
        else
            lstm_bp_flag = 3;
        
        
        if(j == layers-1)
            
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_up,error_model[i],dx);

        if(j == layers-1){
            
            if(j != 0)
                get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(j != 0)
                temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,dropout_output,lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            else
                temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        
        else if(j != layers-1 && j != 0){
            
            get_dropout_array(lstms[j]->size,lstms[j-1]->dropout_mask_up,lstms[j-1]->lstm_hidden[i],dropout_output);
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            

            temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, dropout_output,lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
        }
        
        else{
            
            get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            temp = lstm_bp(lstm_bp_flag,lstms[j]->size, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->d_u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        copy_array(dx,dz,lstms[0]->size);
        free(dx);
        
        if(!j && input_error != NULL)
                input_error[i] = lstm_dinput(i,lstms[j]->size,matrix[j],lstms[j]);
    }
    
    free(dropout_output);
    free(dropout_output2);
    free(dz);
    return matrix;
    
}

/* This function computes the feed forward of a recurrent_enc_dec with group normalization params (for decoder)
 * 
  * Input:
 * 
 *             @ float** hidden_state:= the hidden states of the previous cells:= layers x size
 *             @ float** cell_states:= layers x size
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window x size 
 *             @ recurrent_rec_dec* rec:= the recurrent enc dec that is gonna compute the feed forward
 * 
 * 
 * */ 
void ff_recurrent_dec(float** hidden_states, float** cell_states, float** input_model, recurrent_enc_dec* rec){
    if(rec->decoder == NULL)
        return;
    int i,j,k,z;
    float** temp = (float**)malloc(sizeof(float*)*rec->decoder->window);
    for(i = 0; i < rec->decoder->window; i++){
        temp[i] = input_model[i];
    }
    int n_cells;
    for(i = 0, n_cells = 1;i < rec->decoder->layers; i+=n_cells){
        n_cells = 1;
        for(k = i; k < rec->decoder->layers && rec->decoder->lstms[k]->norm_flag != GROUP_NORMALIZATION; k++,n_cells++);
        if(k == rec->decoder->layers){ n_cells--; k--;}
        ff_decoder_lstm(&hidden_states[i],&cell_states[i],temp,rec->decoder->window,rec->decoder->lstms[0]->size,n_cells,&rec->decoder->lstms[i], rec);
        for(j = 0; j < rec->decoder->window; j++){
            temp[j] = rec->decoder->lstms[k]->out_up[j];
        }
        
        if(rec->decoder->lstms[k]->norm_flag == GROUP_NORMALIZATION){
            for(j = 0; j < rec->decoder->lstms[k]->window/rec->decoder->lstms[k]->n_grouped_cell; j++){
                batch_normalization_feed_forward(rec->decoder->lstms[k]->n_grouped_cell,&temp[j*rec->decoder->lstms[k]->n_grouped_cell],rec->decoder->lstms[k]->bns[j]->temp_vectors,rec->decoder->lstms[k]->bns[j]->vector_dim,rec->decoder->lstms[k]->bns[j]->gamma,rec->decoder->lstms[k]->bns[j]->beta,rec->decoder->lstms[k]->bns[j]->mean,rec->decoder->lstms[k]->bns[j]->var,rec->decoder->lstms[k]->bns[j]->outputs,rec->decoder->lstms[k]->bns[j]->epsilon);
            }
            
            for(j = 0; j < rec->decoder->lstms[k]->window/rec->decoder->lstms[k]->n_grouped_cell; j++){
                for(z = 0; z < rec->decoder->lstms[k]->n_grouped_cell; z++){
                    temp[j*rec->decoder->lstms[k]->n_grouped_cell+z] = rec->decoder->lstms[k]->bns[j]->outputs[z];
                }
            }
            
        }
    }
    
    free(temp);
}



/* This function computes the backpropagation of a recurrent_enc_dec with grouped normalization layers (for decoder)
 * 
 *  * Inputs:
 * 
 * 
 *             @ float** hidden_states:= the hidden sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** cell_states:= the cell sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** input_model:= the input passed to the model, dimensions: m->window*m->size
 *             @ float** error_model:= the error of the model, dimensions: m->window*m->size
 *             @ recurrent_enc_dec* rec:= the recurrent enc dec model
 *             @ float** input_error:= the error of the inputs of this model, dimensions: m->window*m->size, must be initialized only with m->window
 * 
 * */
float*** bp_recurrent_dec(float** hidden_states, float** cell_states, float** input_model, float** error_model, recurrent_enc_dec* rec, float** input_error){
    if(rec->decoder == NULL)
        return NULL;
    float*** ret = (float***)malloc(sizeof(float**)*rec->decoder->layers);//Storing all the returned values of bp of lstms
    float*** ret2;//to handle the returned values of bp of lstms
    float** input_error3 = (float**)malloc(sizeof(float*)*rec->decoder->window);//input error for lstms
    float** error2_model = (float**)malloc(sizeof(float*)*rec->decoder->window);//error propagated
    int i,j,ret_count = rec->decoder->layers-1,z;
    float** temp = (float**)malloc(sizeof(float*)*rec->decoder->window);// inputs of lstms
    for(i = 0; i < rec->decoder->window; i++){
        error2_model[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        copy_array(error_model[i],error2_model[i],rec->decoder->lstms[0]->size);
    }
    int flagg = 0;
    int k = 0;
    while(k > -1){
        int flag = 0;
        for(k = ret_count; k >= 0; k--){
            if(rec->decoder->lstms[k]->norm_flag == GROUP_NORMALIZATION){
                if(flagg) flagg = 0;
                else{
                    flag = 1;
                    break;
                }
            }
        }
        if(flag){
            int n_cells = ret_count-k;
            flagg = 1;
            if(n_cells){
                for(j = 0; j < rec->decoder->lstms[k]->window/rec->decoder->lstms[k]->n_grouped_cell; j++){
                    for(z = 0; z < rec->decoder->lstms[k]->n_grouped_cell; z++){
                        temp[j*rec->decoder->lstms[k]->n_grouped_cell+z] = rec->decoder->lstms[k]->bns[j]->outputs[z];
                    }
                }
                ret2 = bp_decoder_lstm(&hidden_states[k+1],&cell_states[k+1],temp,error2_model,rec->decoder->window,rec->decoder->lstms[0]->size,n_cells,&rec->decoder->lstms[k+1],input_error3,rec);
                for(j = 0; j < rec->decoder->window; j++){
                    copy_array(input_error3[j],error2_model[j],rec->decoder->lstms[0]->size);
                    free(input_error3[j]);
                }
                for(j = ret_count; j > ret_count-n_cells; j--){
                    ret[j] = ret2[n_cells-1-(ret_count-j)];
                }
                free(ret2);
            }
            for(j = 0; j < rec->decoder->lstms[k]->window/rec->decoder->lstms[k]->n_grouped_cell;j++){
                batch_normalization_back_prop(rec->decoder->lstms[k]->n_grouped_cell,&rec->decoder->lstms[k]->out_up[j*rec->decoder->lstms[k]->n_grouped_cell],rec->decoder->lstms[k]->bns[j]->temp_vectors,rec->decoder->lstms[k]->bns[j]->vector_dim,rec->decoder->lstms[k]->bns[j]->gamma,rec->decoder->lstms[k]->bns[j]->beta,rec->decoder->lstms[k]->bns[j]->mean,rec->decoder->lstms[k]->bns[j]->var,&error2_model[j*rec->decoder->lstms[k]->n_grouped_cell],rec->decoder->lstms[k]->bns[j]->d_gamma,rec->decoder->lstms[k]->bns[j]->d_beta,rec->decoder->lstms[k]->bns[j]->error2,rec->decoder->lstms[k]->bns[j]->temp1,rec->decoder->lstms[k]->bns[j]->temp2,rec->decoder->lstms[k]->bns[j]->epsilon);
            }
            
            for(j = 0; j < rec->decoder->window/rec->decoder->lstms[k]->n_grouped_cell; j++){
                for(z = 0; z < rec->decoder->lstms[k]->n_grouped_cell; z++){
                    copy_array(rec->decoder->lstms[k]->bns[j]->error2[z],error2_model[j*rec->decoder->lstms[k]->n_grouped_cell+z],rec->decoder->lstms[0]->size);
                }
            }
            
            if(n_cells)
                ret_count = k;
        }
        
        else{
            int n_cells = ret_count-k;
            int k2 = k;
            if(k < 0) k = 0;
            ret2 = bp_decoder_lstm(&hidden_states[k],&cell_states[k],input_model,error2_model,rec->decoder->window,rec->decoder->lstms[0]->size,n_cells,&rec->decoder->lstms[k],input_error,rec);
            k = k2;

            for(j = ret_count; j > k; j--){
                ret[j] = ret2[j];
            }    
            free(ret2);
        }
    }
    
    free_matrix(error2_model,rec->decoder->window);
    free(input_error3);
    free(temp);
    return ret;
    
}



/* this function computes the toal feedforward of a recurrent enc dec structure
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ float** hidden_states:= the hidden states for encoder size rec->encoder->n_lstm*rec->encoder->lstms[0]->size
 *                 @ float** cell_states:= the cell states for the encoder size rec->encoder->n_lstm*rec->encoder->lstms[0]->size
 *                 @ float** input_model1:= the inputs for the encoder size rec->encoder->window*rec->encder->lstms[0]->size
 *                 @ float** input_model2:= the inputs for the decoder size rec->decoder->window*(rec->decoder->lstms[0]->size-rec->encoder->lstms[0]->size)
 *                 @ recurrent_enc_dec* rec:= the recurrent encoder decoder structure
 * */            
void ff_recurrent_enc_dec(float** hidden_states, float** cell_states, float** input_model1, float** input_model2, recurrent_enc_dec* rec){
    ff_rmodel(hidden_states,cell_states,input_model1,rec->encoder);
    int i;
    for(i = 0; i < rec->encoder->window; i++){
        copy_array(rec->encoder->lstms[rec->encoder->n_lstm-1]->lstm_hidden[i],&rec->flatten_fcl_input[i*rec->encoder->lstms[0]->size],rec->encoder->lstms[0]->size);
    }
    
    float** hiddens = (float**)malloc(sizeof(float*)*rec->encoder->n_lstm);
    float** cells = (float**)malloc(sizeof(float*)*rec->encoder->n_lstm);
    
    for(i = 0; i < rec->encoder->n_lstm; i++){
        hiddens[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        cells[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        copy_array(rec->encoder->lstms[i]->lstm_hidden[rec->encoder->window-1],hiddens[i],rec->encoder->lstms[i]->size);
        copy_array(rec->encoder->lstms[i]->lstm_cell[rec->encoder->window-1],hiddens[i],rec->encoder->lstms[i]->size);
    }
    
    float** input = (float**)malloc(sizeof(float*)*rec->decoder->window);
    
    for(i = 0; i < rec->decoder->window; i++){
        input[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        copy_array(input_model2[i],&input[i][rec->encoder->lstms[0]->size],rec->decoder->lstms[0]->size-rec->encoder->lstms[0]->size);
    }
    
    ff_recurrent_dec(hiddens,cells,input,rec);
    
    free_matrix(hiddens,rec->encoder->n_lstm);
    free_matrix(cells,rec->encoder->n_lstm);
    free_matrix(input,rec->decoder->window);
}


/* this function computes the backpropagation of recurrent encoder decoder
 * 
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ float** hidden_states:= the hidden states for encoder size rec->encoder->n_lstm*rec->encoder->lstms[0]->size
 *                 @ float** cell_states:= the cell states for the encoder size rec->encoder->n_lstm*rec->encoder->lstms[0]->size
 *                 @ float** input_model1:= the inputs for the encoder size rec->encoder->window*rec->encder->lstms[0]->size
 *                 @ float** input_model2:= the inputs for the decoder size rec->decoder->window*(rec->decoder->lstms[0]->size-rec->encoder->lstms[0]->size)
 *                 @ recurrent_enc_dec* rec:= the recurrent encoder decoder structure
 *                 @ float** input_error1:= should be either initialized with the first dimension rec->encoder->window, or should be null
 *                 @ float** input_error2:= should be either initialized with the first dimensions rec->decoder->window*rec->decoder->lstms[0]->size, or should be null
 * */
float*** bp_recurrent_enc_dec(float** hidden_states, float** cell_states, float** input_model1, float** input_model2, float** error_model, recurrent_enc_dec* rec, float** input_error1,float** input_error2){
    int i;
    float** hiddens = (float**)malloc(sizeof(float*)*rec->encoder->n_lstm);
    float** cells = (float**)malloc(sizeof(float*)*rec->encoder->n_lstm);
    
    for(i = 0; i < rec->encoder->n_lstm; i++){
        hiddens[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        cells[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        copy_array(rec->encoder->lstms[i]->lstm_hidden[rec->encoder->window-1],hiddens[i],rec->encoder->lstms[i]->size);
        copy_array(rec->encoder->lstms[i]->lstm_cell[rec->encoder->window-1],hiddens[i],rec->encoder->lstms[i]->size);
    }
    
    float** input = (float**)malloc(sizeof(float*)*rec->decoder->window);
    float** input_error = (float**)malloc(sizeof(float*)*rec->decoder->window);
    
    for(i = 0; i < rec->decoder->window; i++){
        input[i] = (float*)calloc(rec->decoder->lstms[0]->size,sizeof(float));
        copy_array(input_model2[i],&input[i][rec->encoder->lstms[0]->size],rec->decoder->lstms[0]->size-rec->encoder->lstms[0]->size);
    }
    
    float*** dfioc = bp_decoder_lstm(hiddens,cells,input,error_model,rec->decoder->window,rec->decoder->lstms[0]->size,rec->decoder->n_lstm,rec->decoder->lstms,input_error,rec);
    float** dropout_mask = (float**)malloc(sizeof(float*)*rec->decoder->n_lstm);
    for(i = 0; i < rec->decoder->n_lstm; i++){
        dropout_mask[i] = rec->decoder->lstms[i]->dropout_mask_right;
    }
    float*** dfioc2 = bp_encoder_lstm(hidden_states,cell_states,input_model1,rec->output_error_encoder,rec->encoder->window,rec->encoder->lstms[0]->size,rec->encoder->n_lstm,rec->encoder->lstms,input_error1,dfioc,dropout_mask,rec->decoder->lstms);
    
    if(input_error2[i] != NULL)
    for(i = 0; i < rec->decoder->window; i++){
        copy_array(&input_error[i][rec->encoder->lstms[0]->size],input_error2[i],rec->decoder->lstms[0]->size-rec->encoder->lstms[0]->size);
    }
    
    free_matrix(hiddens,rec->encoder->n_lstm);
    free_matrix(cells,rec->encoder->n_lstm);
    free_matrix(input,rec->decoder->window);
    free_matrix(input_error,rec->decoder->window);
    for(i = 0; i < rec->decoder->n_lstm; i++){
        free_matrix(dfioc[i],4);
    }
    free(dfioc);
    
    sum_models_partial_derivatives(rec->m[0],&rec->m[1],rec->decoder->window-1);
    
    return dfioc2;
    
}



/* This function can updates the recurrent enc dec of the network using the adam algorithm or the nesterov momentum or another optimizer
 * 
 * Input:
 * 
 *             @ recurrent_enc_dec* m:= the recurrent model that must be update
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *             @ int mini_batch_size:= the batch used
 *             @ int gradient_descent_flag:= NESTEROV or ADAM (1,2)
 *                @ float* b1:= the hyper parameter b1 of adam algorithm
 *                @ float* b2:= the hyper parameter b2 of adam algorithm
 *                @ int regularization:= NO_REGULARIZATION or L2 (0,1)
 *                @ int total_number_weights:= the number of total weights of the network (for l2 regularization)
 *                @ float lambda:= a float value for l2 regularization
 *                @ unsigned long long int* t:= the number of time radam has been used
 * */
void update_recurrent_enc_dec_model(recurrent_enc_dec* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t){
    if(m == NULL)
        return;
    
    int i;
    
    if(gradient_descent_flag == NESTEROV){
        update_rmodel(m->encoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);    
        update_rmodel(m->decoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);    
        update_model(m->m[0],lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        for(i = 1; i < m->decoder->window; i++){
            paste_w_model(m->m[0],m->m[i]);
        }    
    }
    
    else if(gradient_descent_flag == ADAM || gradient_descent_flag == DIFF_GRAD || gradient_descent_flag == ADAMOD){
        update_rmodel(m->encoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        (*b1)/=m->beta1_adam;
        (*b2)/=m->beta2_adam;  
        update_rmodel(m->decoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        (*b1)/=m->beta1_adam;
        (*b2)/=m->beta2_adam;     
        update_model(m->m[0],lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        for(i = 1; i < m->decoder->window; i++){
            paste_w_model(m->m[0],m->m[i]);
        }    
        
    }
    
    else if(gradient_descent_flag == RADAM){
        update_rmodel(m->encoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        (*b1)/=m->beta1_adam;
        (*b2)/=m->beta2_adam;
        (*t)--;  
        update_rmodel(m->decoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        (*b1)/=m->beta1_adam;
        (*b2)/=m->beta2_adam;
        (*t)--; 
        update_model(m->m[0],lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
        for(i = 1; i < m->decoder->window; i++){
            paste_w_model(m->m[0],m->m[i]);
        }    
    } 
}


void sum_recurrent_enc_dec_partial_derivatives(recurrent_enc_dec* rec1,recurrent_enc_dec* rec2,recurrent_enc_dec* rec3){
	sum_rmodel_partial_derivatives(rec1->encoder,rec2->encoder,rec3->encoder);
	sum_rmodel_partial_derivatives(rec1->decoder,rec2->decoder,rec3->decoder);
	sum_model_partial_derivatives(rec1->m[0],rec2->m[0],rec3->m[0]);
}


void sum_recurrent_enc_decs_partial_derivatives(recurrent_enc_dec* sum, recurrent_enc_dec** rec, int n_models){
	int i;
	for(i = 0; i < n_models; i++){
		sum_recurrent_enc_dec_partial_derivatives(sum,rec[i],sum);
	}
}
