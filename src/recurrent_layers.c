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

/* This function returns a lstm structure, with the big difference with the commond long short term memory
 * standard structure, that each weight array is size*size dimensions (you can pad)
 * 
 * Inputs:
 * 
 *             @ int size:= the size of each hidden layer inside the lstm structure
 *             @ int dropout_flag1:= the dropout flag for the y output of the cell
 *             @ float dropout_threshold1:= the dropout_threshold for the y output
 *             @ int dropout_flag2:= the dropout flag for the hidden output to the next cell
 *             @ float dropout_threshold2:= the dropout threshold for the hidden state of the cell
 *             @ int layer:= the vertical layer of the lstm cell
 *                @ int window:= the number of unrolled cell in orizontal
 * 
 * */
lstm* recurrent_lstm (int input_size, int output_size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window, int residual_flag, int norm_flag, int n_grouped_cell, int training_mode, int feed_forward_flag){
    if(layer < 0 || input_size <= 0 || output_size <= 0){
        fprintf(stderr,"Error: the layer flag must be >= 0 and size param should be > 0\n");
        exit(1);
    }
    
    if(window < 2){
        fprintf(stderr,"Error: your window must be >= 2\n");
        exit(1);
    }
    
    if(norm_flag == GROUP_NORMALIZATION){
        if(n_grouped_cell > window || window%n_grouped_cell){
            fprintf(stderr,"Error: you assumed the group normalization, but the number of grouped cells doesn't divide perfectly the number of unrolled cell during feed forward\n");
            exit(1);
        }
    }
    
    if(residual_flag == LSTM_RESIDUAL && input_size != output_size){
        fprintf(stderr,"Error: if have set the residual for this lstm cell, but your input size does not match your output size!\n");
        exit(1);
    }
    int i,j;
    lstm* lstml = (lstm*)malloc(sizeof(lstm));
    lstml->layer = layer;
    lstml->input_size = input_size;
    lstml->output_size = output_size;
    lstml->dropout_flag_up = dropout_flag1;
    lstml->dropout_flag_right = dropout_flag2;
    lstml->w = (float**)malloc(sizeof(float*)*4);
    lstml->w_active_output_neurons = (int**)malloc(sizeof(int*)*4);
    lstml->u_active_output_neurons = (int**)malloc(sizeof(int*)*4);
    lstml->u = (float**)malloc(sizeof(float*)*4);
    if(training_mode != EDGE_POPUP){
        lstml->d_w = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_w_diff_grad = (float**)malloc(sizeof(float*)*4);
        lstml->d1_w = (float**)malloc(sizeof(float*)*4);
        lstml->d2_w = (float**)malloc(sizeof(float*)*4);
        lstml->d3_w = (float**)malloc(sizeof(float*)*4);
        lstml->d_u = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_u_diff_grad = (float**)malloc(sizeof(float*)*4);
        lstml->d1_u = (float**)malloc(sizeof(float*)*4);
        lstml->d2_u = (float**)malloc(sizeof(float*)*4);
        lstml->d3_u = (float**)malloc(sizeof(float*)*4);
    }
    
    else{
        lstml->d_w = NULL;
        lstml->ex_d_w_diff_grad = NULL;
        lstml->d1_w = NULL;
        lstml->d2_w = NULL;
        lstml->d3_w = NULL;
        lstml->d_u = NULL;
        lstml->ex_d_u_diff_grad = NULL;
        lstml->d1_u = NULL;
        lstml->d2_u = NULL;
        lstml->d3_u = NULL;
    }
    lstml->biases = (float**)malloc(sizeof(float*)*4);
    if(training_mode != EDGE_POPUP){
        lstml->d_biases = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_biases_diff_grad = (float**)malloc(sizeof(float*)*4);
        lstml->d1_biases = (float**)malloc(sizeof(float*)*4);
        lstml->d2_biases = (float**)malloc(sizeof(float*)*4);
        lstml->d3_biases = (float**)malloc(sizeof(float*)*4);
    }
    
    else{
        lstml->d_biases = NULL;
        lstml->ex_d_biases_diff_grad = NULL;
        lstml->d1_biases = NULL;
        lstml->d2_biases = NULL;
        lstml->d3_biases = NULL;
    }
    lstml->lstm_z = (float***)malloc(sizeof(float**)*window);
    lstml->lstm_hidden = (float**)malloc(sizeof(float*)*window);
    lstml->out_up = (float**)malloc(sizeof(float*)*window);
    lstml->lstm_cell = (float**)malloc(sizeof(float*)*window);
    lstml->dropout_mask_up = (float*)malloc(sizeof(float)*output_size);
    lstml->dropout_mask_right = (float*)malloc(sizeof(float)*output_size);
    lstml->dropout_threshold_up = dropout_threshold1;
    
    lstml->dropout_threshold_right = dropout_threshold2;
    lstml->residual_flag = residual_flag;
    lstml->norm_flag = norm_flag;
    lstml->n_grouped_cell = n_grouped_cell;
    if(norm_flag == GROUP_NORMALIZATION){
        lstml->bns = (bn**)malloc(sizeof(bn*)*window/n_grouped_cell);
        for(i = 0; i < window/n_grouped_cell; i++){
            lstml->bns[i] = batch_normalization(n_grouped_cell,output_size,layer,NO_ACTIVATION);
        }
    }
    
    else{
        lstml->bns = NULL;
    }
    
    if(lstml->dropout_flag_up == NO_DROPOUT)
        lstml->dropout_threshold_up = 0;
    if(lstml->dropout_flag_right == NO_DROPOUT)
        lstml->dropout_threshold_right = 0;
    
    if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
        lstml->w_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d_w_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d1_w_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d1_u_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d2_w_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d2_u_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d3_w_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d3_u_scores = (float**)malloc(sizeof(float*)*4);
        lstml->u_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d_u_scores = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_w_scores_diff_grad = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_u_scores_diff_grad = (float**)malloc(sizeof(float*)*4);
        lstml->w_indices = (int**)malloc(sizeof(int*)*4);
        lstml->u_indices = (int**)malloc(sizeof(int*)*4);
    }
    
    else{
        lstml->w_scores = NULL;
        lstml->d_w_scores = NULL;
        lstml->d1_w_scores = NULL;
        lstml->d1_u_scores = NULL;
        lstml->d2_w_scores = NULL;
        lstml->d2_u_scores = NULL;
        lstml->d3_w_scores = NULL;
        lstml->d3_u_scores = NULL;
        lstml->u_scores = NULL;
        lstml->d_u_scores = NULL;
        lstml->ex_d_w_scores_diff_grad = NULL;
        lstml->ex_d_u_scores_diff_grad = NULL;
        lstml->w_indices = NULL;
        lstml->u_indices = NULL;
    }
    
    
    for(i = 0; i < window; i++){
        lstml->lstm_z[i] = (float**)malloc(sizeof(float*)*4);
        lstml->lstm_hidden[i] = (float*)calloc(output_size,sizeof(float));
        lstml->lstm_cell[i] = (float*)calloc(output_size,sizeof(float));
        lstml->out_up[i] = (float*)calloc(output_size,sizeof(float));
        for(j = 0; j < 4; j++){
            lstml->lstm_z[i][j] = (float*)calloc(output_size,sizeof(float));
        }
    }
    
    for(i = 0; i < 4; i++){
        lstml->w[i] = (float*)calloc(output_size*input_size,sizeof(float));
        lstml->u[i] = (float*)calloc(output_size*output_size,sizeof(float));
        lstml->w_active_output_neurons[i] = (int*)calloc(output_size*input_size,sizeof(int));
        lstml->u_active_output_neurons[i] = (int*)calloc(output_size*output_size,sizeof(int));
        for(j = 0; j < output_size*input_size; j++){
            lstml->w[i][j] = random_general_gaussian_xavier_init(input_size);
            lstml->w_active_output_neurons[i][j] = 1;
        }
        for(j = 0; j < output_size*output_size; j++){
            lstml->u[i][j] = random_general_gaussian_xavier_init(output_size);
            lstml->u_active_output_neurons[i][j] = 1;
        }
        
        if(training_mode != EDGE_POPUP){
            lstml->d_w[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->ex_d_w_diff_grad[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d1_w[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d2_w[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d3_w[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d_u[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->ex_d_u_diff_grad[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d1_u[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d2_u[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d3_u[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d_biases[i] = (float*)calloc(output_size,sizeof(float));
            lstml->ex_d_biases_diff_grad[i] = (float*)calloc(output_size,sizeof(float));
            lstml->d1_biases[i] = (float*)calloc(output_size,sizeof(float));
            lstml->d2_biases[i] = (float*)calloc(output_size,sizeof(float));
            lstml->d3_biases[i] = (float*)calloc(output_size,sizeof(float));
        }
        
        else if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
            lstml->w_scores[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d_w_scores[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d1_w_scores[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d1_u_scores[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d2_w_scores[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d2_u_scores[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d3_w_scores[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d3_u_scores[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->u_scores[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d_u_scores[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->ex_d_w_scores_diff_grad[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->ex_d_u_scores_diff_grad[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->w_indices[i] = (int*)calloc(output_size*input_size,sizeof(int));
            lstml->u_indices[i] = (int*)calloc(output_size*output_size,sizeof(int));
            for(j = 0; j < input_size*output_size; j++){
                lstml->w_indices[i][j] = j;
                
            }
            for(j = 0; j < output_size*output_size; j++){
                lstml->u_indices[i][j] = j;
            }
        }
        
        lstml->biases[i] = (float*)calloc(output_size,sizeof(float)); 
        
    }
    
    for(i = 0; i < output_size; i++){
        lstml->dropout_mask_up[i] = 1;
        lstml->dropout_mask_right[i] = 1;
    }
    
    lstml->training_mode = training_mode;
    lstml->feed_forward_flag = feed_forward_flag;
    lstml->k_percentage = 1;
    lstml->window = window;
    
    return lstml;
    
    
}
/* This function returns a lstm structure, with the big difference with the commond long short term memory
 * standard structure, that each weight array is size*size dimensions (you can pad)
 * 
 * Inputs:
 * 
 *             @ int size:= the size of each hidden layer inside the lstm structure
 *             @ int dropout_flag1:= the dropout flag for the y output of the cell
 *             @ float dropout_threshold1:= the dropout_threshold for the y output
 *             @ int dropout_flag2:= the dropout flag for the hidden output to the next cell
 *             @ float dropout_threshold2:= the dropout threshold for the hidden state of the cell
 *             @ int layer:= the vertical layer of the lstm cell
 *                @ int window:= the number of unrolled cell in orizontal
 * 
 * */
lstm* recurrent_lstm_without_learning_parameters (int input_size, int output_size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window, int residual_flag, int norm_flag, int n_grouped_cell, int training_mode, int feed_forward_flag){
    if(layer < 0 || input_size <= 0 || output_size <= 0){
        fprintf(stderr,"Error: the layer flag must be >= 0 and size param should be > 0\n");
        exit(1);
    }
    
    if(window < 2){
        fprintf(stderr,"Error: your window must be >= 2\n");
        exit(1);
    }
    
    if(norm_flag == GROUP_NORMALIZATION){
        if(n_grouped_cell > window || window%n_grouped_cell){
            fprintf(stderr,"Error: you assumed the group normalization, but the number of grouped cells doesn't divide perfectly the number of unrolled cell during feed forward\n");
            exit(1);
        }
    }
    int i,j;
    lstm* lstml = (lstm*)malloc(sizeof(lstm));
    lstml->layer = layer;
    lstml->input_size = input_size;
    lstml->output_size = output_size;
    lstml->dropout_flag_up = dropout_flag1;
    lstml->dropout_flag_right = dropout_flag2;
    lstml->w = NULL;
    lstml->w_active_output_neurons = NULL;
    lstml->u_active_output_neurons = NULL;
    lstml->u = NULL;
    if(training_mode != EDGE_POPUP){
        lstml->d_w = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_w_diff_grad = NULL;
        lstml->d1_w = NULL;
        lstml->d2_w = NULL;
        lstml->d3_w = NULL;
        lstml->d_u = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_u_diff_grad = NULL;
        lstml->d1_u = NULL;
        lstml->d2_u = NULL;
        lstml->d3_u = NULL;
    }
    
    else{
        lstml->d_w = NULL;
        lstml->ex_d_w_diff_grad = NULL;
        lstml->d1_w = NULL;
        lstml->d2_w = NULL;
        lstml->d3_w = NULL;
        lstml->d_u = NULL;
        lstml->ex_d_u_diff_grad = NULL;
        lstml->d1_u = NULL;
        lstml->d2_u = NULL;
        lstml->d3_u = NULL;
    }
    lstml->biases = NULL;
    if(training_mode != EDGE_POPUP){
        lstml->d_biases = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_biases_diff_grad = NULL;
        lstml->d1_biases = NULL;
        lstml->d2_biases = NULL;
        lstml->d3_biases = NULL;
    }
    
    else{
        lstml->d_biases = NULL;
        lstml->ex_d_biases_diff_grad = NULL;
        lstml->d1_biases = NULL;
        lstml->d2_biases = NULL;
        lstml->d3_biases = NULL;
    }
    lstml->lstm_z = (float***)malloc(sizeof(float**)*window);
    lstml->lstm_hidden = (float**)malloc(sizeof(float*)*window);
    lstml->out_up = (float**)malloc(sizeof(float*)*window);
    lstml->lstm_cell = (float**)malloc(sizeof(float*)*window);
    lstml->dropout_mask_up = (float*)malloc(sizeof(float)*output_size);
    lstml->dropout_mask_right = (float*)malloc(sizeof(float)*output_size);
    lstml->dropout_threshold_up = dropout_threshold1;
    
    lstml->dropout_threshold_right = dropout_threshold2;
    lstml->residual_flag = residual_flag;
    lstml->norm_flag = norm_flag;
    lstml->n_grouped_cell = n_grouped_cell;
    if(norm_flag == GROUP_NORMALIZATION){
        lstml->bns = (bn**)malloc(sizeof(bn*)*window/n_grouped_cell);
        for(i = 0; i < window/n_grouped_cell; i++){
            lstml->bns[i] = batch_normalization_without_learning_parameters(n_grouped_cell,output_size,layer,NO_ACTIVATION);
        }
    }
    
    else{
        lstml->bns = NULL;
    }
    
    if(lstml->dropout_flag_up == NO_DROPOUT)
        lstml->dropout_threshold_up = 0;
    if(lstml->dropout_flag_right == NO_DROPOUT)
        lstml->dropout_threshold_right = 0;
    
    if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
        lstml->w_scores = NULL;
        lstml->d_w_scores = (float**)malloc(sizeof(float*)*4);
        lstml->d1_w_scores = NULL;
        lstml->d1_u_scores = NULL;
        lstml->d2_w_scores = NULL;
        lstml->d2_u_scores = NULL;
        lstml->d3_w_scores = NULL;
        lstml->d3_u_scores = NULL;
        lstml->u_scores = NULL;
        lstml->d_u_scores = (float**)malloc(sizeof(float*)*4);
        lstml->ex_d_w_scores_diff_grad = NULL;
        lstml->ex_d_u_scores_diff_grad = NULL;
        lstml->w_indices = NULL;
        lstml->u_indices = NULL;
    }
    
    else{
        lstml->w_scores = NULL;
        lstml->d_w_scores = NULL;
        lstml->d1_w_scores = NULL;
        lstml->d1_u_scores = NULL;
        lstml->d2_w_scores = NULL;
        lstml->d2_u_scores = NULL;
        lstml->d3_w_scores = NULL;
        lstml->d3_u_scores = NULL;
        lstml->u_scores = NULL;
        lstml->d_u_scores = NULL;
        lstml->ex_d_w_scores_diff_grad = NULL;
        lstml->ex_d_u_scores_diff_grad = NULL;
        lstml->w_indices = NULL;
        lstml->u_indices = NULL;
    }
    
    
    for(i = 0; i < window; i++){
        lstml->lstm_z[i] = (float**)malloc(sizeof(float*)*4);
        lstml->lstm_hidden[i] = (float*)calloc(output_size,sizeof(float));
        lstml->lstm_cell[i] = (float*)calloc(output_size,sizeof(float));
        lstml->out_up[i] = (float*)calloc(output_size,sizeof(float));
        for(j = 0; j < 4; j++){
            lstml->lstm_z[i][j] = (float*)calloc(output_size,sizeof(float));
        }
    }
    
    for(i = 0; i < 4; i++){
        
        if(training_mode != EDGE_POPUP){
            lstml->d_w[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d_u[i] = (float*)calloc(output_size*output_size,sizeof(float));
            lstml->d_biases[i] = (float*)calloc(output_size,sizeof(float));
        }
        
        else if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
            
            lstml->d_w_scores[i] = (float*)calloc(output_size*input_size,sizeof(float));
            lstml->d_u_scores[i] = (float*)calloc(output_size*output_size,sizeof(float));
  
        }
        
    }
    
    for(i = 0; i < output_size; i++){
        lstml->dropout_mask_up[i] = 1;
        lstml->dropout_mask_right[i] = 1;
    }
    
    lstml->training_mode = training_mode;
    lstml->feed_forward_flag = feed_forward_flag;
    lstml->k_percentage = 1;
    lstml->window = window;
    
    return lstml;
    
    
}

int exists_d_params_lstm(lstm* l){
    return l->training_mode != EDGE_POPUP;
}

int exists_edge_popup_stuff_lstm(lstm* l){
    return l->training_mode == EDGE_POPUP || l->feed_forward_flag == EDGE_POPUP;
}

int exists_dropout_up(lstm* l){
    return l->dropout_flag_up != NO_DROPOUT;
}

int exists_dropout_right(lstm* l){
    return l->dropout_flag_right != NO_DROPOUT;
}


/* This function frees the space allocated by a rlstm structure
 * 
 * Inputs:
 * 
 *             @ lstm* rlstm:= the lstm structure that must be deallocated
 * 
 * */
void free_recurrent_lstm(lstm* rlstm){
    
    if(rlstm == NULL)
        return;
        
    
    int i,j;
    
    for(i = 0; i < rlstm->window; i++){
        for(j = 0; j < 4; j++){
            free(rlstm->lstm_z[i][j]);
        }
        free(rlstm->lstm_z[i]);
        free(rlstm->lstm_hidden[i]);
        free(rlstm->lstm_cell[i]);
        free(rlstm->out_up[i]);
    }
    
    for(i = 0; i < 4; i++){
        free(rlstm->w[i]);
        free(rlstm->u[i]);
        if(exists_d_params_lstm(rlstm)){
            free(rlstm->d_w[i]);
            free(rlstm->ex_d_w_diff_grad[i]);
            free(rlstm->d1_w[i]);
            free(rlstm->d2_w[i]);
            free(rlstm->d3_w[i]);
            free(rlstm->d_u[i]);
            free(rlstm->ex_d_u_diff_grad[i]);
            free(rlstm->d1_u[i]);
            free(rlstm->d2_u[i]);
            free(rlstm->d3_u[i]);
            free(rlstm->d_biases[i]);
            free(rlstm->d1_biases[i]);
            free(rlstm->d2_biases[i]);
            free(rlstm->d3_biases[i]);
            free(rlstm->ex_d_biases_diff_grad[i]);
        }
        free(rlstm->biases[i]);
        free(rlstm->w_active_output_neurons[i]);
        free(rlstm->u_active_output_neurons[i]);
        
        if(exists_edge_popup_stuff_lstm(rlstm)){
            free(rlstm->w_scores[i]);
            free(rlstm->d_w_scores[i]);
            free(rlstm->d1_w_scores[i]);
            free(rlstm->d1_u_scores[i]);
            free(rlstm->d2_w_scores[i]);
            free(rlstm->d2_u_scores[i]);
            free(rlstm->d3_w_scores[i]);
            free(rlstm->d3_u_scores[i]);
            free(rlstm->u_scores[i]);
            free(rlstm->d_u_scores[i]);
            free(rlstm->ex_d_w_scores_diff_grad[i]);
            free(rlstm->ex_d_u_scores_diff_grad[i]);
            free(rlstm->w_indices[i]);
            free(rlstm->u_indices[i]);
        }
        
    }
    
    if(rlstm->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < rlstm->window/rlstm->n_grouped_cell; i++){
            free_batch_normalization(rlstm->bns[i]);
        }
        free(rlstm->bns);
    }
    
    free(rlstm->w_indices);
    free(rlstm->u_indices);
    free(rlstm->w_active_output_neurons);
    free(rlstm->u_active_output_neurons);
    free(rlstm->w_scores);
    free(rlstm->d_w_scores);
    free(rlstm->d1_w_scores);
    free(rlstm->d1_u_scores);
    free(rlstm->d2_w_scores);
    free(rlstm->d2_u_scores);
    free(rlstm->d3_w_scores);
    free(rlstm->d3_u_scores);
    free(rlstm->u_scores);
    free(rlstm->d_u_scores);
    free(rlstm->ex_d_w_scores_diff_grad);
    free(rlstm->ex_d_u_scores_diff_grad);
    free(rlstm->w);
    free(rlstm->u);
    free(rlstm->d_w);
    free(rlstm->ex_d_w_diff_grad);
    free(rlstm->d1_w);
    free(rlstm->d2_w);
    free(rlstm->d3_w);
    free(rlstm->d_u);
    free(rlstm->ex_d_u_diff_grad);
    free(rlstm->d1_u);
    free(rlstm->d2_u);
    free(rlstm->d3_u);
    free(rlstm->biases);
    free(rlstm->d_biases);
    free(rlstm->ex_d_biases_diff_grad);
    free(rlstm->d1_biases);
    free(rlstm->d2_biases);
    free(rlstm->d3_biases);
    free(rlstm->lstm_z);
    free(rlstm->lstm_hidden);
    free(rlstm->lstm_cell);
    free(rlstm->out_up);
    free(rlstm->dropout_mask_right);
    free(rlstm->dropout_mask_up);
    free(rlstm);

}
/* This function frees the space allocated by a rlstm structure
 * 
 * Inputs:
 * 
 *             @ lstm* rlstm:= the lstm structure that must be deallocated
 * 
 * */
void free_recurrent_lstm_without_learning_parameters(lstm* rlstm){
    
    if(rlstm == NULL)
        return;
        
    
    int i,j;
    
    for(i = 0; i < rlstm->window; i++){
        for(j = 0; j < 4; j++){
            free(rlstm->lstm_z[i][j]);
        }
        free(rlstm->lstm_z[i]);
        free(rlstm->lstm_hidden[i]);
        free(rlstm->lstm_cell[i]);
        free(rlstm->out_up[i]);
    }
    
    for(i = 0; i < 4; i++){
        if(exists_d_params_lstm(rlstm)){
            free(rlstm->d_w[i]);
            free(rlstm->d_u[i]);
            free(rlstm->d_biases[i]);
        }
        
        
        if(exists_edge_popup_stuff_lstm(rlstm)){
            free(rlstm->d_w_scores[i]);
            free(rlstm->d_u_scores[i]);
        }
        
    }
    
    if(rlstm->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < rlstm->window/rlstm->n_grouped_cell; i++){
            free_batch_normalization(rlstm->bns[i]);
        }
        free(rlstm->bns);
    }
    
    free(rlstm->w_indices);
    free(rlstm->u_indices);
    free(rlstm->w_active_output_neurons);
    free(rlstm->u_active_output_neurons);
    free(rlstm->w_scores);
    free(rlstm->d_w_scores);
    free(rlstm->d1_w_scores);
    free(rlstm->d1_u_scores);
    free(rlstm->d2_w_scores);
    free(rlstm->d2_u_scores);
    free(rlstm->d3_w_scores);
    free(rlstm->d3_u_scores);
    free(rlstm->u_scores);
    free(rlstm->d_u_scores);
    free(rlstm->ex_d_w_scores_diff_grad);
    free(rlstm->ex_d_u_scores_diff_grad);
    free(rlstm->w);
    free(rlstm->u);
    free(rlstm->d_w);
    free(rlstm->ex_d_w_diff_grad);
    free(rlstm->d1_w);
    free(rlstm->d2_w);
    free(rlstm->d3_w);
    free(rlstm->d_u);
    free(rlstm->ex_d_u_diff_grad);
    free(rlstm->d1_u);
    free(rlstm->d2_u);
    free(rlstm->d3_u);
    free(rlstm->biases);
    free(rlstm->d_biases);
    free(rlstm->ex_d_biases_diff_grad);
    free(rlstm->d1_biases);
    free(rlstm->d2_biases);
    free(rlstm->d3_biases);
    free(rlstm->lstm_z);
    free(rlstm->lstm_hidden);
    free(rlstm->lstm_cell);
    free(rlstm->out_up);
    free(rlstm->dropout_mask_right);
    free(rlstm->dropout_mask_up);
    free(rlstm);

}

/* This function saves a lstm layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ lstm* rlstm:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_lstm(lstm* rlstm, int n){
    if(rlstm == NULL)
        return;
    int i,j;
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
    
    i = fwrite(&rlstm->feed_forward_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    i = fwrite(&rlstm->k_percentage,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    i = fwrite(&rlstm->training_mode,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    i = fwrite(&rlstm->residual_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->norm_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->n_grouped_cell,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->input_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->output_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->layer,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->dropout_flag_up,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->dropout_flag_right,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->window,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->dropout_threshold_up,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    i = fwrite(&rlstm->dropout_threshold_right,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
        exit(1);
    }
    
    
    for(j = 0; j < 4; j++){
        i = fwrite(rlstm->w[j],sizeof(float)*(rlstm->output_size)*(rlstm->input_size),1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
            exit(1);
        }
        
        i = fwrite(rlstm->u[j],sizeof(float)*(rlstm->output_size)*(rlstm->output_size),1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
            exit(1);
        }
        
        i = fwrite(rlstm->biases[j],sizeof(float)*(rlstm->output_size),1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
            exit(1);
        }
    }
    
    if(exists_edge_popup_stuff_lstm(rlstm)){
        for(j = 0; j < 4; j++){
            i = fwrite(rlstm->w_scores[j],sizeof(float)*(rlstm->output_size)*(rlstm->input_size),1,fw);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
                exit(1);
            }
            
            i = fwrite(rlstm->w_indices[j],sizeof(int)*(rlstm->output_size)*(rlstm->input_size),1,fw);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
                exit(1);
            }
            
            i = fwrite(rlstm->u_indices[j],sizeof(int)*(rlstm->output_size)*(rlstm->output_size),1,fw);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
                exit(1);
            }
            
            i = fwrite(rlstm->u_scores[j],sizeof(float)*(rlstm->output_size)*(rlstm->output_size),1,fw);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred saving a lstm layer\n");
                exit(1);
            }
        }
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    if(rlstm->norm_flag == GROUP_NORMALIZATION){
        for(j = 0; j < rlstm->window/rlstm->n_grouped_cell; j++){
            save_bn(rlstm->bns[j],n);
        }
    }
    free(s);
    
}


/* This function loads a lstm layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
lstm* load_lstm(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i,j;
    
    int input_size = 0, output_size = 0, layer = 0,dropout_flag_up = 0,dropout_flag_right = 0, window = 0, residual_flag = 0, norm_flag = 0, n_grouped_cell = 0, training_mode = 0, feed_forward_flag = 0;
    float dropout_threshold_right = 0,dropout_threshold_up = 0, k_percentage;
    float** w_scores = NULL;
    float** w = (float**)malloc(sizeof(float*)*4);
    float** u_scores = NULL;
    float** u = (float**)malloc(sizeof(float*)*4);
    float** biases = (float**)malloc(sizeof(float*)*4);
    int** w_indices = NULL;
    int** u_indices = NULL;
    
    
    i = fread(&feed_forward_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    i = fread(&k_percentage,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    i = fread(&training_mode,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    
    if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
        w_scores = (float**)malloc(sizeof(float*)*4);
        w_indices = (int**)malloc(sizeof(int*)*4);
        u_scores = (float**)malloc(sizeof(float*)*4);
        u_indices = (int**)malloc(sizeof(int*)*4);
    }
    
    i = fread(&residual_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&norm_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&n_grouped_cell,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&input_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&output_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&layer,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&dropout_flag_up,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&dropout_flag_right,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&window,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&dropout_threshold_up,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    i = fread(&dropout_threshold_right,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
        exit(1);
    }
    
    for(j = 0; j < 4; j++){
        w[j] = (float*)malloc(sizeof(float)*output_size*input_size);
        u[j] = (float*)malloc(sizeof(float)*output_size*output_size);
        biases[j] = (float*)malloc(sizeof(float)*output_size);
        
        i = fread(w[j],sizeof(float)*(output_size)*(input_size),1,fr);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
            exit(1);
        }
        
        i = fread(u[j],sizeof(float)*(output_size)*(output_size),1,fr);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
            exit(1);
        }
        
        i = fread(biases[j],sizeof(float)*(output_size),1,fr);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
            exit(1);
        }
    }
    
    if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
        for(j = 0; j < 4; j++){
            w_scores[j] = (float*)malloc(sizeof(float)*input_size*output_size);
            u_scores[j] = (float*)malloc(sizeof(float)*output_size*output_size);
            w_indices[j] = (int*)malloc(sizeof(int)*input_size*output_size);
            u_indices[j] = (int*)malloc(sizeof(int)*output_size*output_size);
            
            i = fread(w_scores[j],sizeof(float)*(input_size)*(output_size),1,fr);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
                exit(1);
            }
            i = fread(w_indices[j],sizeof(int)*(input_size)*(output_size),1,fr);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
                exit(1);
            }
            i = fread(u_indices[j],sizeof(int)*(output_size)*(output_size),1,fr);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
                exit(1);
            }
            
            i = fread(u_scores[j],sizeof(float)*(output_size)*(output_size),1,fr);
        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a lstm layer\n");
                exit(1);
            }
            
        }
    }
    
    bn** bns = NULL;
    if(norm_flag == GROUP_NORMALIZATION){
        bns = (bn**)malloc(sizeof(bn*)*window/n_grouped_cell);
        for(i = 0; i < window/n_grouped_cell; i++){
            bns[i] = load_bn(fr);
        }
    }
    
    lstm* l = recurrent_lstm(input_size,output_size,dropout_flag_up,dropout_threshold_up,dropout_flag_right,dropout_threshold_right,layer, window, residual_flag,norm_flag,n_grouped_cell,training_mode,feed_forward_flag);
    for(i = 0; i < 4; i++){
        copy_array(w[i],l->w[i],output_size*input_size);
        copy_array(u[i],l->u[i],output_size*output_size);
        copy_array(biases[i],l->biases[i],output_size);
        if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
            copy_array(w_scores[i],l->w_scores[i],output_size*input_size);
            copy_array(u_scores[i],l->u_scores[i],output_size*output_size);
            copy_int_array(u_indices[i],l->u_indices[i],output_size*output_size);
            copy_int_array(w_indices[i],l->w_indices[i],output_size*input_size);
            free(w_scores[i]);
            free(u_scores[i]);
            free(u_indices[i]);
            free(w_indices[i]);
        }
        free(w[i]);
        free(u[i]);
        free(biases[i]);
    }
    
    if(norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < window/n_grouped_cell; i++){
            free_batch_normalization(l->bns[i]);
        }
        l->bns = bns;
    }
    l->k_percentage = k_percentage;
    if(exists_edge_popup_stuff_lstm(l)){
        for(i = 0; i <4; i++){
            get_used_outputs_lstm(l->w_active_output_neurons[i],l->input_size,l->output_size,l->w_indices[i],l->k_percentage);
            get_used_outputs_lstm(l->u_active_output_neurons[i],l->output_size,l->output_size,l->u_indices[i],l->k_percentage);
        }
    }
    else{
        for(i = 0; i <4; i++){
            for(j = 0; j < output_size*input_size; j++){
                l->w_active_output_neurons[i][j] = 1;
            }
            for(j = 0; j < output_size*output_size; j++){
                l->u_active_output_neurons[i][j] = 1;
            }
        }
    }
    free(w);
    free(u);
    free(biases);
    free(w_scores);
    free(u_scores);
    free(u_indices);
    free(w_indices);
    return l;
}



/* This function returns a lstm* layer that is the same copy of the input l
 * except for the temporary arrays used during the feed forward and backprop.
 * You have a lstm* l structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in l are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ lstm* l:= the lstm layer that must be copied
 * 
 * */
lstm* copy_lstm(lstm* l){
    if(l == NULL)
        return NULL;
    int i;
    lstm* copy = recurrent_lstm(l->input_size,l->output_size,l->dropout_flag_up,l->dropout_threshold_up,l->dropout_flag_right,l->dropout_threshold_right,l->layer, l->window,l->residual_flag,l->norm_flag,l->n_grouped_cell,l->training_mode,l->feed_forward_flag);
    for(i = 0; i < 4; i++){
        copy_array(l->w[i],copy->w[i],l->output_size*l->input_size);
        copy_array(l->u[i],copy->u[i],l->output_size*l->output_size);
        copy_int_array(l->w_active_output_neurons[i],copy->w_active_output_neurons[i],l->output_size*l->input_size);
        copy_int_array(l->u_active_output_neurons[i],copy->u_active_output_neurons[i],l->output_size*l->output_size);
        
        if(exists_d_params_lstm(l)){
            copy_array(l->d_w[i],copy->d_w[i],l->output_size*l->input_size);
            copy_array(l->ex_d_w_diff_grad[i],copy->ex_d_w_diff_grad[i],l->output_size*l->input_size);
            copy_array(l->d1_w[i],copy->d1_w[i],l->output_size*l->input_size);
            copy_array(l->d2_w[i],copy->d2_w[i],l->output_size*l->input_size);
            copy_array(l->d3_w[i],copy->d3_w[i],l->output_size*l->input_size);
            copy_array(l->d_u[i],copy->d_u[i],l->output_size*l->output_size);
            copy_array(l->ex_d_u_diff_grad[i],copy->ex_d_u_diff_grad[i],l->output_size*l->output_size);
            copy_array(l->d1_u[i],copy->d1_u[i],l->output_size*l->output_size);
            copy_array(l->d2_u[i],copy->d2_u[i],l->output_size*l->output_size);
            copy_array(l->d3_u[i],copy->d3_u[i],l->output_size*l->output_size);
            copy_array(l->d_biases[i],copy->d_biases[i],l->output_size);
            copy_array(l->ex_d_biases_diff_grad[i],copy->ex_d_biases_diff_grad[i],l->output_size);
            copy_array(l->d1_biases[i],copy->d1_biases[i],l->output_size);
            copy_array(l->d2_biases[i],copy->d2_biases[i],l->output_size);
            copy_array(l->d3_biases[i],copy->d3_biases[i],l->output_size);
        }
        copy_array(l->biases[i],copy->biases[i],l->output_size);
        if(exists_edge_popup_stuff_lstm(l)){
            copy_array(l->w_scores[i],copy->w_scores[i],l->output_size*l->input_size);
            copy_array(l->u_scores[i],copy->u_scores[i],l->output_size*l->output_size);
            copy_array(l->d_w_scores[i],copy->d_w_scores[i],l->output_size*l->input_size);
            copy_array(l->ex_d_w_scores_diff_grad[i],copy->ex_d_w_scores_diff_grad[i],l->output_size*l->input_size);
            copy_array(l->d1_w_scores[i],copy->d1_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d2_w_scores[i],copy->d2_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d3_w_scores[i],copy->d3_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d_u_scores[i],copy->d_u_scores[i],l->output_size*l->output_size);
            copy_array(l->ex_d_u_scores_diff_grad[i],copy->ex_d_u_scores_diff_grad[i],l->output_size*l->output_size);
            copy_array(l->d1_u_scores[i],copy->d1_u_scores[i],l->output_size*l->output_size);
            copy_array(l->d2_u_scores[i],copy->d2_u_scores[i],l->output_size*l->output_size);
            copy_array(l->d3_u_scores[i],copy->d3_u_scores[i],l->output_size*l->output_size);
            copy_int_array(l->u_indices[i],copy->u_indices[i],l->output_size*l->output_size);
            copy_int_array(l->w_indices[i],copy->w_indices[i],l->output_size*l->input_size);
        }
    }
    
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            copy_array(l->bns[i]->gamma,copy->bns[i]->gamma,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d_gamma,copy->bns[i]->d_gamma,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d1_gamma,copy->bns[i]->d1_gamma,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d2_gamma,copy->bns[i]->d2_gamma,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d3_gamma,copy->bns[i]->d3_gamma,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->ex_d_gamma_diff_grad,copy->bns[i]->ex_d_gamma_diff_grad,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->beta,copy->bns[i]->beta,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d_beta,copy->bns[i]->d_beta,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d2_beta,copy->bns[i]->d2_beta,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d3_beta,copy->bns[i]->d3_beta,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->ex_d_beta_diff_grad,copy->bns[i]->ex_d_beta_diff_grad,l->bns[i]->vector_dim);
        }
    }
    
    
    copy->k_percentage = l->k_percentage;
    
    return copy;
}
/* This function returns a lstm* layer that is the same copy of the input l
 * except for the temporary arrays used during the feed forward and backprop.
 * You have a lstm* l structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in l are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ lstm* l:= the lstm layer that must be copied
 * 
 * */
lstm* copy_lstm_without_learning_parameters(lstm* l){
    if(l == NULL)
        return NULL;
    int i;
    lstm* copy = recurrent_lstm_without_learning_parameters(l->input_size,l->output_size,l->dropout_flag_up,l->dropout_threshold_up,l->dropout_flag_right,l->dropout_threshold_right,l->layer, l->window,l->residual_flag,l->norm_flag,l->n_grouped_cell,l->training_mode,l->feed_forward_flag);
    for(i = 0; i < 4; i++){
        
        if(exists_d_params_lstm(l)){
            copy_array(l->d_w[i],copy->d_w[i],l->output_size*l->input_size);
            copy_array(l->d_u[i],copy->d_u[i],l->output_size*l->output_size);
            copy_array(l->d_biases[i],copy->d_biases[i],l->output_size);
        }
        if(exists_edge_popup_stuff_lstm(l)){
            copy_array(l->d_w_scores[i],copy->d_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d_u_scores[i],copy->d_u_scores[i],l->output_size*l->output_size);
        }
    }
    
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            copy_array(l->bns[i]->d_gamma,copy->bns[i]->d_gamma,l->bns[i]->vector_dim);
            copy_array(l->bns[i]->d_beta,copy->bns[i]->d_beta,l->bns[i]->vector_dim);
        }
    }
    
    
    copy->k_percentage = l->k_percentage;
    
    return copy;
}


/* This function returns a lstm* layer that is the same copy of the input l
 * except for the arrays used during the ff and bp
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ lstm* l:= the lstm layer that must be copied
 *             @ lstm* copy:= the lstm layer where l is copied
 * 
 * */
void paste_lstm(lstm* l,lstm* copy){
    if(l == NULL)
        return;
        
    int i;
    for(i = 0; i < 4; i++){
        copy_array(l->w[i],copy->w[i],l->output_size*l->input_size);
        copy_array(l->u[i],copy->u[i],l->output_size*l->output_size);
        copy_int_array(l->w_active_output_neurons[i],copy->w_active_output_neurons[i],l->output_size*l->input_size);
        copy_int_array(l->u_active_output_neurons[i],copy->u_active_output_neurons[i],l->output_size*l->output_size);
        copy_array(l->biases[i],copy->biases[i],l->output_size);
        if(exists_d_params_lstm(l)){
            copy_array(l->d_w[i],copy->d_w[i],l->output_size*l->input_size);
            copy_array(l->ex_d_w_diff_grad[i],copy->ex_d_w_diff_grad[i],l->output_size*l->input_size);
            copy_array(l->d1_w[i],copy->d1_w[i],l->output_size*l->input_size);
            copy_array(l->d2_w[i],copy->d2_w[i],l->output_size*l->input_size);
            copy_array(l->d3_w[i],copy->d3_w[i],l->output_size*l->input_size);
            copy_array(l->d_u[i],copy->d_u[i],l->output_size*l->output_size);
            copy_array(l->ex_d_u_diff_grad[i],copy->ex_d_u_diff_grad[i],l->output_size*l->output_size);
            copy_array(l->d1_u[i],copy->d1_u[i],l->output_size*l->output_size);
            copy_array(l->d2_u[i],copy->d2_u[i],l->output_size*l->output_size);
            copy_array(l->d3_u[i],copy->d3_u[i],l->output_size*l->output_size);
            copy_array(l->d_biases[i],copy->d_biases[i],l->output_size);
            copy_array(l->ex_d_biases_diff_grad[i],copy->ex_d_biases_diff_grad[i],l->output_size);
            copy_array(l->d1_biases[i],copy->d1_biases[i],l->output_size);
            copy_array(l->d2_biases[i],copy->d2_biases[i],l->output_size);
            copy_array(l->d3_biases[i],copy->d3_biases[i],l->output_size);

        }
        
        if(exists_edge_popup_stuff_lstm(l)){
            copy_array(l->w_scores[i],copy->w_scores[i],l->output_size*l->input_size);
            copy_array(l->u_scores[i],copy->u_scores[i],l->output_size*l->output_size);
            copy_array(l->d_w_scores[i],copy->d_w_scores[i],l->output_size*l->input_size);
            copy_array(l->ex_d_w_scores_diff_grad[i],copy->ex_d_w_scores_diff_grad[i],l->output_size*l->input_size);
            copy_array(l->d1_w_scores[i],copy->d1_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d2_w_scores[i],copy->d2_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d3_w_scores[i],copy->d3_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d_u_scores[i],copy->d_u_scores[i],l->output_size*l->output_size);
            copy_array(l->ex_d_u_scores_diff_grad[i],copy->ex_d_u_scores_diff_grad[i],l->output_size*l->output_size);
            copy_array(l->d1_u_scores[i],copy->d1_u_scores[i],l->output_size*l->output_size);
            copy_array(l->d2_u_scores[i],copy->d2_u_scores[i],l->output_size*l->output_size);
            copy_array(l->d3_u_scores[i],copy->d3_u_scores[i],l->output_size*l->output_size);
            copy_int_array(l->u_indices[i],copy->u_indices[i],l->output_size*l->output_size);
            copy_int_array(l->w_indices[i],copy->w_indices[i],l->output_size*l->input_size);
            
        }
    }
    
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            paste_bn(l->bns[i],copy->bns[i]);
        }
    }
    
    copy->k_percentage = l->k_percentage;
    return;
}
/* This function returns a lstm* layer that is the same copy of the input l
 * except for the arrays used during the ff and bp
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ lstm* l:= the lstm layer that must be copied
 *             @ lstm* copy:= the lstm layer where l is copied
 * 
 * */
void paste_lstm_without_learning_parameters(lstm* l,lstm* copy){
    if(l == NULL)
        return;
        
    int i;
    for(i = 0; i < 4; i++){
        if(exists_d_params_lstm(l)){
            copy_array(l->d_w[i],copy->d_w[i],l->output_size*l->input_size);
            copy_array(l->d_u[i],copy->d_u[i],l->output_size*l->output_size);
            copy_array(l->d_biases[i],copy->d_biases[i],l->output_size);
        }
        
        if(exists_edge_popup_stuff_lstm(l)){
            
            copy_array(l->d_w_scores[i],copy->d_w_scores[i],l->output_size*l->input_size);
            copy_array(l->d_u_scores[i],copy->d_u_scores[i],l->output_size*l->output_size);
        }
    }
    
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            paste_bn_without_learning_parameters(l->bns[i],copy->bns[i]);
        }
    }
    
    copy->k_percentage = l->k_percentage;
    return;
}

/* This function returns a lstm* layer that is the same copy of the input l
 * except for the arrays used during the ff and bp
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ lstm* l:= the lstm layer that must be copied
 *             @ lstm* copy:= the lstm layer where l is copied
 * 
 * */
void paste_w_lstm(lstm* l,lstm* copy){
    if(l == NULL)
        return;
        
    int i;
    for(i = 0; i < 4; i++){
        copy_array(l->w[i],copy->w[i],l->output_size*l->input_size);
        copy_array(l->u[i],copy->u[i],l->output_size*l->output_size);
        copy_array(l->biases[i],copy->biases[i],l->output_size);
    }
    
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            paste_w_bn(l->bns[i],copy->bns[i]);
        }
    }
    return;
}
/* This function returns a lstm* layer that is the same copy for the weights and biases
 * of the layer l with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ lstm* l:= the lstm layer that must be copied
 *             @ lstm* copy:= the lstm layer where l is copied
 *             @ float tau:= the tau param
 * */
void slow_paste_lstm(lstm* l,lstm* copy, float tau){
    if(l == NULL)
        return;
    int i,j;
    for(i = 0; i < 4; i++){
        for(j = 0; j < l->output_size*l->input_size; j++){
            copy->w[i][j] = tau*l->w[i][j] + (1-tau)*copy->w[i][j];
            if(exists_d_params_lstm(l)){
                copy->d1_w[i][j] = tau*l->d1_w[i][j] + (1-tau)*copy->d1_w[i][j];
                copy->d2_w[i][j] = tau*l->d2_w[i][j] + (1-tau)*copy->d2_w[i][j];
                copy->d3_w[i][j] = tau*l->d3_w[i][j] + (1-tau)*copy->d3_w[i][j];
                copy->ex_d_w_diff_grad[i][j] = tau*l->ex_d_w_diff_grad[i][j] + (1-tau)*copy->ex_d_w_diff_grad[i][j];
            }
            
            
            if(exists_edge_popup_stuff_lstm(l)){
                copy->w_scores[i][j] = tau*l->w_scores[i][j] + (1-tau)*copy->w_scores[i][j];
                copy->d_w_scores[i][j] = tau*l->d_w_scores[i][j] + (1-tau)*copy->d_w_scores[i][j];
                copy->d1_w_scores[i][j] = tau*l->d1_w_scores[i][j] + (1-tau)*copy->d1_w_scores[i][j];
                copy->d2_w_scores[i][j] = tau*l->d2_w_scores[i][j] + (1-tau)*copy->d2_w_scores[i][j];
                copy->d3_w_scores[i][j] = tau*l->d3_w_scores[i][j] + (1-tau)*copy->d3_w_scores[i][j];
                copy->ex_d_w_scores_diff_grad[i][j] = tau*l->ex_d_w_scores_diff_grad[i][j] + (1-tau)*copy->ex_d_w_scores_diff_grad[i][j];
                copy->w_indices[i][j] = j;
            }
        }
        for(j = 0; j < l->output_size*l->output_size; j++){
            copy->u[i][j] = tau*l->u[i][j] + (1-tau)*copy->u[i][j];
            if(exists_d_params_lstm(l)){
                copy->d1_u[i][j] = tau*l->d1_u[i][j] + (1-tau)*copy->d1_u[i][j];
                copy->d2_u[i][j] = tau*l->d2_u[i][j] + (1-tau)*copy->d2_u[i][j];
                copy->d3_u[i][j] = tau*l->d3_u[i][j] + (1-tau)*copy->d3_u[i][j];
                copy->ex_d_u_diff_grad[i][j] = tau*l->ex_d_u_diff_grad[i][j] + (1-tau)*copy->ex_d_u_diff_grad[i][j];
            }
            
            if(j < l->output_size){
                copy->biases[i][j] = tau*l->biases[i][j] + (1-tau)*copy->biases[i][j];
                if(exists_d_params_lstm(l)){
                    copy->d1_biases[i][j] = tau*l->d1_biases[i][j] + (1-tau)*copy->d1_biases[i][j];
                    copy->d2_biases[i][j] = tau*l->d2_biases[i][j] + (1-tau)*copy->d2_biases[i][j];
                    copy->d3_biases[i][j] = tau*l->d3_biases[i][j] + (1-tau)*copy->d3_biases[i][j];
                    copy->ex_d_biases_diff_grad[i][j] = tau*l->ex_d_biases_diff_grad[i][j] + (1-tau)*copy->ex_d_biases_diff_grad[i][j];
                }
            }
            
            if(exists_edge_popup_stuff_lstm(l)){
                copy->u_scores[i][j] = tau*l->u_scores[i][j] + (1-tau)*copy->u_scores[i][j];
                copy->d_u_scores[i][j] = tau*l->d_u_scores[i][j] + (1-tau)*copy->d_u_scores[i][j];
                copy->d1_u_scores[i][j] = tau*l->d1_u_scores[i][j] + (1-tau)*copy->d1_u_scores[i][j];
                copy->d2_u_scores[i][j] = tau*l->d2_u_scores[i][j] + (1-tau)*copy->d2_u_scores[i][j];
                copy->d3_u_scores[i][j] = tau*l->d3_u_scores[i][j] + (1-tau)*copy->d3_u_scores[i][j];
                copy->ex_d_u_scores_diff_grad[i][j] = tau*l->ex_d_u_scores_diff_grad[i][j] + (1-tau)*copy->ex_d_u_scores_diff_grad[i][j];
                copy->u_indices[i][j] = j;
            }
        }
        
        if(exists_edge_popup_stuff_lstm(l)){
            sort(copy->d_w_scores[i],copy->w_indices[i],0,copy->output_size*copy->input_size-1);
            sort(copy->d_u_scores[i],copy->w_indices[i],0,copy->output_size*copy->output_size-1);
            get_used_outputs_lstm(copy->w_active_output_neurons[i],copy->input_size,copy->output_size,copy->w_indices[i],copy->k_percentage);
            get_used_outputs_lstm(copy->u_active_output_neurons[i],copy->output_size,copy->output_size,copy->u_indices[i],copy->k_percentage);
        }
    }
    
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            slow_paste_bn(l->bns[i],copy->bns[i],tau);
        }
    }
    return;
}


/* this function reset all the arrays of a lstm layer
 * used during the feed forward and backpropagation
 * You have a lstm* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ lstm* f:= a lstm* f layer
 * 
 * */
lstm* reset_lstm(lstm* f){
    if(f == NULL)
        return NULL;
    int i,j,k;
    for(i = 0; i < 4; i++){
        for(j = 0; j < f->output_size*f->input_size; j++){
            if(exists_d_params_lstm(f)){
                f->d_w[i][j] = 0;
            }
        }
        for(j = 0; j < f->output_size*f->output_size; j++){
            if(exists_d_params_lstm(f)){
                f->d_u[i][j] = 0;
            }
            if(j < f->output_size){
                if(exists_d_params_lstm(f))
                f->d_biases[i][j] = 0;
                if(!i){
                    f->dropout_mask_up[j] = 1;
                }
                if(!i){
                    f->dropout_mask_right[j] = 1;
                }
            }
        }
    }
    for(i = 0; i < f->window; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < f->output_size; k++){
                f->lstm_z[i][j][k] = 0;                        
                f->lstm_hidden[i][k] = 0;
                f->lstm_cell[i][k] = 0;
                f->out_up[i][k] = 0;
            }
        }
    }
    
    if(exists_edge_popup_stuff_lstm(f)){
        for(i = 0; i < 4; i++){
            for(j = 0; j < f->output_size*f->input_size; j++){
                f->d_w_scores[i][j] = 0;
            }
            for(j = 0; j < f->output_size*f->output_size; j++){
                f->d_u_scores[i][j] = 0;
            }
            sort(f->w_scores[i],f->w_indices[i],0,f->output_size*f->input_size-1);
            sort(f->u_scores[i],f->u_indices[i],0,f->output_size*f->output_size-1);
            get_used_outputs_lstm(f->w_active_output_neurons[i],f->input_size,f->output_size,f->w_indices[i],f->k_percentage);
            get_used_outputs_lstm(f->u_active_output_neurons[i],f->output_size,f->output_size,f->u_indices[i],f->k_percentage);
        }
        
    }
    
    if(f->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->window/f->n_grouped_cell; i++){
            reset_bn(f->bns[i]);
        }
    }
    return f;
}
/* this function reset all the arrays of a lstm layer
 * used during the feed forward and backpropagation
 * You have a lstm* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ lstm* f:= a lstm* f layer
 * 
 * */
lstm* reset_lstm_except_partial_derivatives(lstm* f){
    if(f == NULL)
        return NULL;
    int i,j,k;
    for(j = 0; j < f->output_size; j++){


        f->dropout_mask_up[j] = 1;
        f->dropout_mask_right[j] = 1;

    }
       
    for(i = 0; i < f->window; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < f->output_size; k++){
                f->lstm_z[i][j][k] = 0;                        
                f->lstm_hidden[i][k] = 0;
                f->lstm_cell[i][k] = 0;
                f->out_up[i][k] = 0;
            }
        }
    }
    
    if(f->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->window/f->n_grouped_cell; i++){
            reset_bn_except_partial_derivatives(f->bns[i]);
        }
    }
    return f;
}
/* this function reset all the arrays of a lstm layer
 * used during the feed forward and backpropagation
 * You have a lstm* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ lstm* f:= a lstm* f layer
 * 
 * */
lstm* reset_lstm_without_dwdb(lstm* f){
    if(f == NULL)
        return NULL;
    int i,j,k;
    for(j = 0; j < f->output_size; j++){


        f->dropout_mask_up[j] = 1;
        f->dropout_mask_right[j] = 1;

    }
       
    for(i = 0; i < f->window; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < f->output_size; k++){
                f->lstm_z[i][j][k] = 0;                        
                f->lstm_hidden[i][k] = 0;
                f->lstm_cell[i][k] = 0;
                f->out_up[i][k] = 0;
            }
        }
    }
    
    if(exists_edge_popup_stuff_lstm(f)){
        for(i = 0; i < 4; i++){
            for(j = 0; j < f->output_size*f->input_size; j++){
                f->d_w_scores[i][j] = 0;
            }
            for(j = 0; j < f->output_size*f->output_size; j++){
                f->d_u_scores[i][j] = 0;
            }
            sort(f->w_scores[i],f->w_indices[i],0,f->output_size*f->input_size-1);
            sort(f->u_scores[i],f->u_indices[i],0,f->output_size*f->output_size-1);
            get_used_outputs_lstm(f->w_active_output_neurons[i],f->input_size,f->output_size,f->w_indices[i],f->k_percentage);
            get_used_outputs_lstm(f->u_active_output_neurons[i],f->output_size,f->output_size,f->u_indices[i],f->k_percentage);
        }
        
    }
    
    if(f->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->window/f->n_grouped_cell; i++){
            reset_bn(f->bns[i]);
        }
    }
    return f;
}
/* this function reset all the arrays of a lstm layer
 * used during the feed forward and backpropagation
 * You have a lstm* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ lstm* f:= a lstm* f layer
 * 
 * */
lstm* reset_lstm_without_dwdb_without_learning_parameters(lstm* f){
    if(f == NULL)
        return NULL;
    int i,j,k;
    for(j = 0; j < f->output_size; j++){


        f->dropout_mask_up[j] = 1;
        f->dropout_mask_right[j] = 1;

    }
       
    for(i = 0; i < f->window; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < f->output_size; k++){
                f->lstm_z[i][j][k] = 0;                        
                f->lstm_hidden[i][k] = 0;
                f->lstm_cell[i][k] = 0;
                f->out_up[i][k] = 0;
            }
        }
    }
    
    if(exists_edge_popup_stuff_lstm(f)){
        for(i = 0; i < 4; i++){
            for(j = 0; j < f->output_size*f->input_size; j++){
                f->d_w_scores[i][j] = 0;
            }
            for(j = 0; j < f->output_size*f->output_size; j++){
                f->d_u_scores[i][j] = 0;
            }
            
        }
        
    }
    
    if(f->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->window/f->n_grouped_cell; i++){
            reset_bn(f->bns[i]);
        }
    }
    return f;
}
/* this function reset all the arrays of a lstm layer
 * used during the feed forward and backpropagation
 * You have a lstm* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ lstm* f:= a lstm* f layer
 * 
 * */
lstm* reset_lstm_without_learning_parameters(lstm* f){
    if(f == NULL)
        return NULL;
    int i,j,k;
    for(i = 0; i < 4; i++){
        for(j = 0; j < f->output_size*f->input_size; j++){
            if(exists_d_params_lstm(f)){
                f->d_w[i][j] = 0;
            }
            
        }
        for(j = 0; j < f->output_size*f->output_size; j++){
            if(exists_d_params_lstm(f)){
                f->d_u[i][j] = 0;
            }
            if(j < f->output_size){
                if(exists_d_params_lstm(f))
                f->d_biases[i][j] = 0;
                if(!i){
                    f->dropout_mask_up[j] = 1;
                }
                if(!i){
                    f->dropout_mask_right[j] = 1;
                }
            }
        }
    }
    for(i = 0; i < f->window; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < f->output_size; k++){
                f->lstm_z[i][j][k] = 0;                        
                f->lstm_hidden[i][k] = 0;
                f->lstm_cell[i][k] = 0;
                f->out_up[i][k] = 0;
            }
        }
    }
    
    if(exists_edge_popup_stuff_lstm(f)){
        for(i = 0; i < 4; i++){
            for(j = 0; j < f->output_size*f->input_size; j++){
                f->d_w_scores[i][j] = 0;
            }
            for(j = 0; j < f->output_size*f->output_size; j++){
                f->d_u_scores[i][j] = 0;
            }
        }
        
    }
    
    if(f->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->window/f->n_grouped_cell; i++){
            reset_bn(f->bns[i]);
        }
    }
    return f;
}





uint64_t size_of_lstm(lstm* l){
    
    uint64_t sum = 0;
    int i;
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            sum+=size_of_bn(l->bns[i]);
        }
    }
    sum += 6*4*l->input_size*l->output_size*sizeof(float);
    sum += 6*4*l->output_size*l->output_size*sizeof(float);
    sum+= 6*4*l->output_size*sizeof(float);
    sum+=7*l->window*l->output_size*sizeof(float);
    sum+=2*l->output_size*sizeof(float);
    if (exists_edge_popup_stuff_lstm(l)){
        sum+= 6*4*l->input_size*l->output_size*sizeof(float);
        sum+= 6*4*l->output_size*l->output_size*sizeof(float);
        sum+=4*l->input_size*l->output_size*sizeof(int);
        sum+=4*l->output_size*l->output_size*sizeof(int);
    }
    return sum; 
    
}
uint64_t size_of_lstm_without_learning_parameters(lstm* l){
    
    uint64_t sum = 0;
    int i;
    if(l->norm_flag == GROUP_NORMALIZATION){
        for(i = 0; i < l->window/l->n_grouped_cell; i++){
            sum+=size_of_bn_without_learning_parameters(l->bns[i]);
        }
    }
    sum += 4*l->input_size*l->output_size*sizeof(float);
    sum += 4*l->output_size*l->output_size*sizeof(float);
    sum+= 4*l->output_size*sizeof(float);
    sum+=7*l->window*l->output_size*sizeof(float);
    sum+=2*l->output_size;
    if (exists_edge_popup_stuff_lstm(l)){
        sum+= 4*l->input_size*l->output_size*sizeof(float);
        sum+= 4*l->output_size*l->output_size*sizeof(float);
    }
    return sum; 
    
}

uint64_t count_weights_lstm(lstm* l){
	return (uint64_t)((l->k_percentage)*4*(l->output_size*l->output_size + l->input_size*l->output_size)); 
}

void get_used_outputs_lstm(int* arr, int input, int output, int* indices, float k_percentage){
    int i,j;
    for(i = input*output-input*output*k_percentage; i < input*output; i++){
        arr[(int)((indices[i]/input))] = 1;
    }
}



/* this function gives the number of float params for biases and weights in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
uint64_t get_array_size_params_lstm(lstm* f){
    uint64_t sum = 0;
    int i;
    if(f->norm_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->window/f->n_grouped_cell; i++){
			sum+=(uint64_t)f->bns[i]->vector_dim*2;
		}
    }
    return (uint64_t)4*(f->input_size*f->output_size + f->output_size*f->output_size + f->output_size)+sum;
}


/* this function gives the number of float params for scores in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
uint64_t get_array_size_scores_lstm(lstm* f){
    return (uint64_t)4*(f->input_size*f->output_size+f->output_size*f->output_size);
}


/* this function gives the number of float params for biases and weights in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
uint64_t get_array_size_weights_lstm(lstm* f){
    uint64_t sum = 0;
    int i;
    if(f->norm_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->window/f->n_grouped_cell; i++){
			sum+=(uint64_t)f->bns[i]->vector_dim*2;
		}
    }
    return (uint64_t)4*(f->input_size*f->output_size + f->output_size*f->output_size)+sum;
}

/* this function pastes the weights and biases from a vector into in a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_params_to_vector_lstm(lstm* f, float* vector){
	int i;
	for(i = 0; i < 4; i++){
		copy_array(f->w[i],&vector[i*f->input_size*f->output_size],f->input_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(f->u[i],&vector[4*f->input_size*f->output_size+i*f->output_size*f->output_size],f->output_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(f->biases[i],&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size)+i*f->output_size],f->output_size);
	}
    if(f->norm_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->window/f->n_grouped_cell; i++){
			copy_array(f->bns[i]->gamma,&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size + f->output_size)+i*2*f->bns[i]->vector_dim],f->bns[i]->vector_dim);
			copy_array(f->bns[i]->beta,&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size + f->output_size)+i*2*f->bns[i]->vector_dim + f->bns[i]->vector_dim],f->bns[i]->vector_dim);
		}
    }
}

/* this function pastes the scores stored in a vector inside a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_scores_to_vector_lstm(lstm* f, float* vector){
	int i;
	for(i = 0; i < 4; i++){
		copy_array(f->w_scores[i],&vector[i*f->input_size*f->output_size],f->input_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(f->u_scores[i],&vector[4*f->input_size*f->output_size + i*f->output_size*f->output_size],f->output_size*f->output_size);
	}
}

/* this function pastes the the weights and biases from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_params_lstm(lstm* f, float* vector){
    int i;
	for(i = 0; i < 4; i++){
		copy_array(&vector[i*f->input_size*f->output_size],f->w[i],f->input_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(&vector[4*f->input_size*f->output_size+i*f->output_size*f->output_size],f->u[i],f->output_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size)+i*f->output_size],f->biases[i],f->output_size);
	}
    if(f->norm_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->window/f->n_grouped_cell; i++){
			copy_array(&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size + f->output_size)+i*2*f->bns[i]->vector_dim],f->bns[i]->gamma,f->bns[i]->vector_dim);
			copy_array(&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size + f->output_size)+i*2*f->bns[i]->vector_dim + f->bns[i]->vector_dim],f->bns[i]->beta,f->bns[i]->vector_dim);
		}
    }
}

/* this function pastes the scores from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_weights_lstm(lstm* f, float* vector){
    int i;
	for(i = 0; i < 4; i++){
		copy_array(&vector[i*f->input_size*f->output_size],f->w[i],f->input_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(&vector[4*f->input_size*f->output_size+i*f->output_size*f->output_size],f->u[i],f->output_size*f->output_size);
	}

    if(f->norm_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->window/f->n_grouped_cell; i++){
			copy_array(&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size)+i*2*f->bns[i]->vector_dim],f->bns[i]->gamma,f->bns[i]->vector_dim);
			copy_array(&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size)+i*2*f->bns[i]->vector_dim + f->bns[i]->vector_dim],f->bns[i]->beta,f->bns[i]->vector_dim);
		}
    }
}

/* this function pastes the the weights from vector to a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_weights_to_vector_lstm(lstm* f, float* vector){
    int i;
	for(i = 0; i < 4; i++){
		copy_array(f->w[i],&vector[i*f->input_size*f->output_size],f->input_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(f->u[i],&vector[4*f->input_size*f->output_size+i*f->output_size*f->output_size],f->output_size*f->output_size);
	}

    if(f->norm_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->window/f->n_grouped_cell; i++){
			copy_array(f->bns[i]->gamma,&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size)+i*2*f->bns[i]->vector_dim],f->bns[i]->vector_dim);
			copy_array(f->bns[i]->beta,&vector[4*(f->input_size*f->output_size+f->output_size*f->output_size)+i*2*f->bns[i]->vector_dim + f->bns[i]->vector_dim],f->bns[i]->vector_dim);
		}
    }
}

/* this function pastes the scores from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_scores_lstm(lstm* f, float* vector){
    int i;
	for(i = 0; i < 4; i++){
		copy_array(&vector[i*f->input_size*f->output_size],f->w_scores[i],f->input_size*f->output_size);
	}
	for(i = 0; i < 4; i++){
		copy_array(&vector[4*f->input_size*f->output_size + i*f->output_size*f->output_size],f->u_scores[i],f->output_size*f->output_size);
	}
}


