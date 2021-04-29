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

/* This function builds a rmodel* structure which can be used to train the network
 * 
 * Input:
 *             
 *             @ int layers:= number of total layers
 *             @ int n_lstm:= same as layer, but only for long short term memory layers
 *             @ lstm** lstms:= your long short term memory layers
 *             @ int window:= is the number of unrolled orizontal recurrent cells are provided
 *             @ int hidden_state_mode:= cna be stateful and stateless (flag STATEFUL, flag STATELESS)
 * 
 * */
rmodel* recurrent_network(int layers, int n_lstm, lstm** lstms, int window, int hidden_state_mode){
    if(!layers || (!n_lstm) || (lstms == NULL)){
        fprintf(stderr,"Error: layers must be > 0 and n_lstm > 0 and lstms != NULL\n");
        exit(1); 
    }
    
    int i,j,k, position;
    
    lstm* temp = NULL;
    
    int** sla = (int**)malloc(sizeof(int*)*layers);
    
    for(i = 0; i < layers; i++){
        sla[i] = (int*)calloc(layers,sizeof(int));
    }
    
    rmodel* m = (rmodel*)malloc(sizeof(rmodel));
    
    for(i = 0; i < n_lstm; i++){
        if(lstms[i]->window != window){
            fprintf(stderr,"Error: your lstm cells must have the same window\n");
            exit(1);
        }
    }
    
    
    /* sorting lstm layers*/
    for(i = 0; i < n_lstm; i++){
                
        for(k = i+1; k < n_lstm; k++){
            if(lstms[i]->layer > lstms[k]->layer){
                temp = lstms[i];
                lstms[i] = lstms[k];
                lstms[k] = temp;
            }
        }
    }
    
    
    /* checking if the layers are sequential or not*/
    position = 0;
    for(i = 0; i < layers; i++){
        /* building sla matrix and gls*/
        k = 0;
        
        for(j = 0; j < n_lstm; j++){
            if(lstms[j]->layer == i){
                sla[i][k] = LSTMS;
                k++;
            }
        }
        
        position += k;
        if(!k && position != layers){
            fprintf(stderr,"Error: your layers are not sequential, missing the layer with index: %d\n",i);
            exit(1);
        }
    }

    for(i = 1; i < n_lstm; i++){
        if(lstms[i]->input_size != lstms[i-1]->output_size){
            fprintf(stderr,"Error: your lstm input-output does not match, layers %d - %d\n",lstms[i]->layer,lstms[i-1]->layer);
            exit(1);
        }
    }
    
    m->layers = layers;
    m->n_lstm = n_lstm;
    m->sla = sla;
    m->lstms = lstms;
    m->window = window;
    m->hidden_state_mode = hidden_state_mode;
    m->beta1_adam = BETA1_ADAM;
    m->beta2_adam = BETA2_ADAM;
    m->beta3_adamod = BETA3_ADAMOD;
        
    return m;
}

/* This function frees the space allocated by a rmodel structure
 * 
 * Input:
 *             @ rmodel* m:= the structure
 * 
 * */
void free_rmodel(rmodel* m){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_lstm; i++){
        free_recurrent_lstm(m->lstms[i]);
    }
    free(m->lstms);
    for(i = 0; i < m->layers; i++){
        free(m->sla[i]);
    }
    free(m->sla);
    free(m);
}
/* This function frees the space allocated by a rmodel structure
 * 
 * Input:
 *             @ rmodel* m:= the structure
 * 
 * */
void free_rmodel_without_learning_parameters(rmodel* m){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_lstm; i++){
        free_recurrent_lstm_without_learning_parameters(m->lstms[i]);
    }
    free(m->lstms);
    for(i = 0; i < m->layers; i++){
        free(m->sla[i]);
    }
    free(m->sla);
    free(m);
}

/* This function copies a rmodel using the copy function for the layers
 * see recurrent_layers.c file
 * 
 * Input:
 *         
 *             @ rmodel* m:= the rmodel that must be copied
 * 
 * */
rmodel* copy_rmodel(rmodel* m){
    if(m == NULL)
        return NULL;
    int i;
    
    lstm** lstms = NULL;
    if(m->lstms!=NULL)
        lstms = (lstm**)malloc(sizeof(lstm*)*m->n_lstm);
    for(i = 0; i < m->n_lstm; i++){
        lstms[i] = copy_lstm(m->lstms[i]);
    }
    rmodel* copy = recurrent_network(m->layers, m->n_lstm,lstms, m->window, m->hidden_state_mode);
    copy->beta1_adam = m->beta1_adam;
    copy->beta2_adam = m->beta2_adam;
    copy->beta3_adamod = m->beta3_adamod;
    return copy;
}
/* This function copies a rmodel using the copy function for the layers
 * see recurrent_layers.c file
 * 
 * Input:
 *         
 *             @ rmodel* m:= the rmodel that must be copied
 * 
 * */
rmodel* copy_rmodel_without_learning_parameters(rmodel* m){
    if(m == NULL)
        return NULL;
    int i;
    
    lstm** lstms = NULL;
    if(m->lstms!=NULL)
        lstms = (lstm**)malloc(sizeof(lstm*)*m->n_lstm);
    for(i = 0; i < m->n_lstm; i++){
        lstms[i] = copy_lstm_without_learning_parameters(m->lstms[i]);
    }
    rmodel* copy = recurrent_network(m->layers, m->n_lstm,lstms, m->window, m->hidden_state_mode);
    copy->beta1_adam = m->beta1_adam;
    copy->beta2_adam = m->beta2_adam;
    copy->beta3_adamod = m->beta3_adamod;
    return copy;
}

/* This function copies a rmodel using the paste function for the layers
 * see recurrent_layers.c file
 * 
 * Input:
 *         
 *             @ rmodel* m:= the rmodel that must be copied
 *             @ rmodel* copy:= the rmodel where m is copied
 * 
 * */
void paste_rmodel(rmodel* m, rmodel* copy){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_lstm; i++){
        paste_lstm(m->lstms[i],copy->lstms[i]);
    }
    return;
}
/* This function copies a rmodel using the paste function for the layers
 * see recurrent_layers.c file
 * 
 * Input:
 *         
 *             @ rmodel* m:= the rmodel that must be copied
 *             @ rmodel* copy:= the rmodel where m is copied
 * 
 * */
void paste_rmodel_without_learning_parameters(rmodel* m, rmodel* copy){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_lstm; i++){
        paste_lstm_without_learning_parameters(m->lstms[i],copy->lstms[i]);
    }
    return;
}

/* This function copies a rmodel using the paste function for the layers
 * see recurrent_layers.c file
 * 
 * Input:
 *         
 *             @ rmodel* m:= the rmodel that must be copied
 *             @ rmodel* copy:= the rmodel where m is copied
 * 
 * */
void paste_w_rmodel(rmodel* m, rmodel* copy){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_lstm; i++){
        paste_w_lstm(m->lstms[i],copy->lstms[i]);
    }
    return;
}
/* This function copies a rmodel with the rule: teta_i:= teta_j*tau +(1-tau)*teta_i
 * 
 * Input:
 *         
 *             @ rmodel* m:= the rmodel that must be copied
 *             @ rmodel* copy:= the model where m is copied
 *             @ float tau:= the tau param
 * 
 * */
void slow_paste_rmodel(rmodel* m, rmodel* copy, float tau){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_lstm; i++){
        slow_paste_lstm(m->lstms[i],copy->lstms[i],tau);
    }
    return;
}

/* This function resets a rmodel
 * returns a rmodel equal to the one as input but with all resetted except for weights and biases
 * */
rmodel* reset_rmodel(rmodel* m){
    if(m == NULL)
        return NULL;
    int i;
    for(i = 0; i < m->n_lstm; i++){
        reset_lstm(m->lstms[i]);
    }
    return m;
}
/* This function resets a rmodel
 * returns a rmodel equal to the one as input but with all resetted except for weights and biases
 * */
rmodel* reset_rmodel_without_learning_parameters(rmodel* m){
    if(m == NULL)
        return NULL;
    int i;
    for(i = 0; i < m->n_lstm; i++){
        reset_lstm_without_learning_parameters(m->lstms[i]);
    }
    return m;
}

/* This function saves a rmodel(recurrent network) on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ rmodel* m:= the actual recurrent network that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_rmodel(rmodel* m, int n){
    if(m == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"w");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&m->layers,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the rmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->n_lstm,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the rmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->window,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the rmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->hidden_state_mode,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the rmodel\n");
        exit(1);
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    for(i = 0; i < m->n_lstm; i++){
        save_lstm(m->lstms[i],n);
    }
    
    
    
    free(s);
}



/* This function loads a recurrent network model from a .bin file with name file
 * 
 * Input:
 * 
 *             @ char* file:= the binary file from which the rmodel will be loaded
 * 
 * */
rmodel* load_rmodel(char* file){
    if(file == NULL)
        return NULL;
    int i;
    FILE* fr = fopen(file,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",file);
        exit(1);
    }
    
    int layers = 0,n_lstm = 0, window = 0, hidden_state_mode = 0;
    
    i = fread(&layers,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the rmodel\n");
        exit(1);
    }
    
    i = fread(&n_lstm,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the rmodel\n");
        exit(1);
    }
    
    i = fread(&window,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the rmodel\n");
        exit(1);
    }
    
    i = fread(&hidden_state_mode,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the rmodel\n");
        exit(1);
    }
    

    lstm** lstms;
    
    if(!n_lstm)
        lstms = NULL;
    else
        lstms = (lstm**)malloc(sizeof(lstm*)*n_lstm);
        
    for(i = 0; i < n_lstm; i++){
        lstms[i] = load_lstm(fr);
    }
    
    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file);
        exit(1);
    }
    
    rmodel* m = recurrent_network(layers,n_lstm,lstms, window, hidden_state_mode);
    
    return m;
    
}



/* Given a rmodel with only long short term memory cells, this functions computes the feed forward
 * for that model given in input the previous hidden state, previous cell state the input model
 * 
 * Input:
 * 
 *             @ float** hidden_state:= the hidden states of the previous cells:= layers x size
 *             @ float** cell_states:= layers x size
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window) x size 
 *             @ int window:= the window of the unrolled cells
 *                @ int size:= the size of the lstms cells
 *                @ int layers:= the number of lstms cells
 *               @ lstm** lstms:= the lstms cells
 * 
 * 
 * */
void ff_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, int window, int layers, lstm** lstms){    
    
    //allocation of the resources

    float* dropout_output = (float*)malloc(sizeof(float)*lstms[0]->output_size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[0]->output_size); //here we store the modified output by dropout coming from an lstm cell
    
    int i,j;
    
    float temp_drp_value;
    /*feed_forward_passage*/
    
    for(i = 0; i < window; i++){
        for(j = 0; j < layers; j++){
            
            if(j == 0){ //j = 0 means that we are at the first lstm_cell in vertical
                if(i == 0){//i = 0 means we are at the first lstm in orizontal
                    //in this case the h-1 and c-1 come from the last mini_batch
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->dropout_threshold_right);
                   
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(input_model[i], dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size);
                    else if(lstms[j]->feed_forward_flag == EDGE_POPUP)
                    lstm_ff_edge_popup(lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices, input_model[i], dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);

                }

                else{
                    
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(input_model[i], dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size);
                    else
                    lstm_ff_edge_popup(lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,input_model[i], dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);
                    
                }
            }
            
            else{
                
                if(i == 0){//i = 0 and j != 0 means that we are at the first lstm in orizontal but not in vertical
                    if(lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->dropout_threshold_right);
                   
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size);
                    else{
                    lstm_ff_edge_popup(lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);
                    
                    }
                }    
                else{
                    
                    
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(dropout_output, dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size);
                    else
                    lstm_ff_edge_popup(lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);
                }
            }
            
            /* the dropout is applied to each lstm_hidden to feed the deeper lstm cell in vertical, as input*/
            if(!i)
                if(lstms[j]->dropout_flag_up == DROPOUT)
                    set_dropout_mask(lstms[j]->output_size,lstms[j]->dropout_mask_up,lstms[j]->dropout_threshold_up);
            
            free(dropout_output);
            dropout_output = (float*)malloc(sizeof(float)*lstms[j]->output_size);
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_up,lstms[j]->lstm_hidden[i],dropout_output);
            
            if(lstms[j]->dropout_flag_up == DROPOUT_TEST)
                mul_value(dropout_output,lstms[j]->dropout_threshold_up,dropout_output,lstms[j]->output_size);
            
            if(!j && lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dropout_output,input_model[i],dropout_output,lstms[j]->output_size);
            else if(j && lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dropout_output,lstms[j-1]->out_up[i],dropout_output,lstms[j]->output_size);
                
            copy_array(dropout_output,lstms[j]->out_up[i],lstms[j]->output_size);
            free(dropout_output2);
            dropout_output2 = (float*)malloc(sizeof(float)*lstms[j]->output_size);
            
        }
    }
    
    free(dropout_output);
    free(dropout_output2);
    
}
/* Given a rmodel with only long short term memory cells, this functions computes the feed forward
 * for that model given in input the previous hidden state, previous cell state the input model
 * 
 * Input:
 * 
 *             @ float** hidden_state:= the hidden states of the previous cells:= layers x size
 *             @ float** cell_states:= layers x size
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window) x size 
 *             @ int window:= the window of the unrolled cells
 *                @ int size:= the size of the lstms cells
 *                @ int layers:= the number of lstms cells
 *               @ lstm** lstms:= the lstms cells
 * 
 * 
 * */
void ff_rmodel_lstm_opt(float** hidden_states, float** cell_states, float** input_model, int window, int layers, lstm** lstms, lstm** lstms2){    
    
    //allocation of the resources

    float* dropout_output = (float*)malloc(sizeof(float)*lstms[0]->output_size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[0]->output_size); //here we store the modified output by dropout coming from an lstm cell
    
    int i,j;
    
    float temp_drp_value;
    /*feed_forward_passage*/
    
    for(i = 0; i < window; i++){
        for(j = 0; j < layers; j++){
            
            if(j == 0){ //j = 0 means that we are at the first lstm_cell in vertical
                if(i == 0){//i = 0 means we are at the first lstm in orizontal
                    //in this case the h-1 and c-1 come from the last mini_batch
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->dropout_threshold_right);
                   
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(input_model[i], dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size, lstms[j]->output_size);
                    else if(lstms[j]->feed_forward_flag == EDGE_POPUP)
                    lstm_ff_edge_popup(lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices, input_model[i], dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);

                }

                else{
                    
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(input_model[i], dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size, lstms[j]->output_size);
                    else
                    lstm_ff_edge_popup(lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,input_model[i], dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);
                    
                }
            }
            
            else{
                
                if(i == 0){//i = 0 and j != 0 means that we are at the first lstm in orizontal but not in vertical
                    if(lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->dropout_threshold_right);
                   
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size, lstms[j]->output_size);
                    else
                    lstm_ff_edge_popup(lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size,lstms[j]->output_size,lstms[j]->k_percentage);
                }    
                else{
                    
                    
                    get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->output_size);
                    if(lstms[j]->feed_forward_flag == FULLY_FEED_FORWARD)
                    lstm_ff(dropout_output, dropout_output2, lstms[j]->lstm_cell[i-1], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size, lstms[j]->output_size);
                    else
                    lstm_ff_edge_popup(lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,dropout_output, dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms2[j]->w, lstms2[j]->u, lstms2[j]->biases, lstms[j]->lstm_z[i], lstms[j]->input_size, lstms[j]->output_size,lstms[j]->k_percentage);
                }
            }
            
            /* the dropout is applied to each lstm_hidden to feed the deeper lstm cell in vertical, as input*/
            if(!i)
                if(lstms[j]->dropout_flag_up == DROPOUT)
                    set_dropout_mask(lstms[j]->output_size,lstms[j]->dropout_mask_up,lstms[j]->dropout_threshold_up);
            
            free(dropout_output);
            dropout_output = (float*)malloc(sizeof(float)*lstms[j]->output_size);
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_up,lstms[j]->lstm_hidden[i],dropout_output);
            
            if(lstms[j]->dropout_flag_up == DROPOUT_TEST)
                mul_value(dropout_output,lstms[j]->dropout_threshold_up,dropout_output,lstms[j]->output_size);
            
            if(!j && lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dropout_output,input_model[i],dropout_output,lstms[j]->output_size);
            else if(j && lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(dropout_output,lstms[j-1]->out_up[i],dropout_output,lstms[j]->output_size);
                
            copy_array(dropout_output,lstms[j]->out_up[i],lstms[j]->output_size);
            free(dropout_output2);
            dropout_output2 = (float*)malloc(sizeof(float)*lstms[j]->output_size);
        }
    }
    
    free(dropout_output);
    free(dropout_output2);
    
}


/* this function returns the error dfioc and set the error of input_error through a back propagation passage
 * the dfioc returning error has this dimensions: m->layers*4*m->size
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
 * 
 * */
float*** bp_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window,int layers,lstm** lstms, float** input_error){

   /* backpropagation passage*/

    int i,j, lstm_bp_flag;
    
    float* dropout_output2 = NULL;
    float* dx; //here we store the modified output by dropout coming from the last lstm cell
    float* dz = (float*)calloc(lstms[layers-1]->output_size,sizeof(float)); //for residual dx
    float*** matrix = (float***)malloc(sizeof(float**)*layers);
    float** temp;
    int output_up = 0;
    for(i = 0; i < layers; i++){
        matrix[i] = NULL;
    }
    /*with i = 0 we should handle the h minus and c minus that are given by the params passed to the function*/
    for(i = window-1; i > 0; i--){
        for(j = layers-1; j >= 0; j--){
            dx = (float*)calloc(lstms[j]->output_size,sizeof(float));
            if(j < layers-1 && lstms[j+1]->residual_flag == LSTM_RESIDUAL)
                sum1D(dx,dz,dx,lstms[j]->output_size);
                
            if(j == layers-1 && i == window-1)
                lstm_bp_flag = 0;
            else if(j != layers-1 && i == window-1)
                lstm_bp_flag = 1;
                
            else if(j == layers-1 && i != window-1)
                lstm_bp_flag = 2;
                
            else
                lstm_bp_flag = 3;
            
            if(j != layers-1)
                output_up = lstms[j+1]->output_size;
            
            free(dropout_output2);
            dropout_output2 = (float*)malloc(sizeof(float)*lstms[j]->output_size);
            
            if(j == layers-1)
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_up,error_model[i],dx);

            if(j == layers-1){
                
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);
                
                if(j != 0){
                    if(i == window-1){
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j]->k_percentage,NULL,NULL);
                    }
                    else{
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size, output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,  lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j]->k_percentage,NULL,NULL);
                    
                    }
                }
                
                else{
                    if(i == window-1){
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j]->k_percentage,NULL,NULL);
                    
                    }
                    else{
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,  lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j]->k_percentage,NULL,NULL);
                    }
                }
                
                
                if(matrix[j] != NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;

            }
            
            else if(j != layers-1 && j){
                
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);

                
                if(i == window-1){
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j+1]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j+1]->k_percentage,lstms[j+1]->w_active_output_neurons,lstms[j+1]->u_active_output_neurons);
                }
                else{
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j+1]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j+1]->k_percentage,lstms[j+1]->w_active_output_neurons,lstms[j+1]->u_active_output_neurons);
                }
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
                
            }
            
            else{
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);

                if(i == window-1){
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)    
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j+1]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j+1]->k_percentage,lstms[j+1]->w_active_output_neurons,lstms[j+1]->u_active_output_neurons);
                }
                else{
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j+1]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j+1]->k_percentage,lstms[j+1]->w_active_output_neurons,lstms[j+1]->u_active_output_neurons);
                }
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
            }
            
            free(dz);
            dz = (float*)calloc(lstms[j]->output_size,sizeof(float));
            copy_array(dx,dz,lstms[j]->output_size);
            free(dx);
            
            if(!j && input_error != NULL){
                input_error[i] = lstm_dinput(i,lstms[j]->output_size,matrix[j],lstms[j]);
                if(lstms[j]->residual_flag == LSTM_RESIDUAL)
                    sum1D(input_error[i],dz,input_error[i],lstms[j]->output_size);
            }
            
            
        }
        
    }
    
    i = 0;
    /* computing back propagation just for the first lstm layers with hidden states defined by the previous batch*/
    for(j = layers-1; j >= 0; j--){
        dx = (float*)calloc(lstms[j]->output_size,sizeof(float));
        if(j < layers-1 && lstms[j+1]->residual_flag == LSTM_RESIDUAL)
            sum1D(dx,dz,dx,lstms[j]->output_size);
        if(j == layers-1 && i == window-1)
            lstm_bp_flag = 0;
        else if(j != layers-1 && i == window-1)
            lstm_bp_flag = 1;
            
        else if(j == layers-1 && i != window-1)
            lstm_bp_flag = 2;
            
        else
            lstm_bp_flag = 3;
        
        
        if(j != layers-1)
            output_up = lstms[j+1]->output_size;
        
        free(dropout_output2);
        dropout_output2 = (float*)malloc(sizeof(float)*lstms[j]->output_size);
        
        
        if(j == layers-1)
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_up,error_model[i],dx);

        if(j == layers-1){
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(j != 0){
                if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                else if(lstms[j]->training_mode == EDGE_POPUP)
                temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j]->k_percentage,NULL,NULL);
            }
            else{
                if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                else if(lstms[j]->training_mode == EDGE_POPUP)
                temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u, lstms[j]->d_biases, lstms[j]->w, lstms[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j]->k_percentage,NULL,NULL);
            }
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        
        else if(j != layers-1 && j != 0){
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            else if(lstms[j]->training_mode == EDGE_POPUP)
            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j+1]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j+1]->k_percentage,lstms[j+1]->w_active_output_neurons,lstms[j+1]->u_active_output_neurons);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
        }
        
        else{
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            else if(lstms[j]->training_mode == EDGE_POPUP)
            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms[j]->d_w,lstms[j]->d_u,lstms[j]->d_biases,lstms[j]->w,lstms[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms[j]->w_active_output_neurons, lstms[j]->u_active_output_neurons, lstms[j+1]->w_indices,lstms[j]->u_indices,lstms[j]->d_w_scores,lstms[j]->d_u_scores,lstms[j+1]->k_percentage,lstms[j+1]->w_active_output_neurons,lstms[j+1]->u_active_output_neurons);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        
        free(dz);
        dz = (float*)calloc(lstms[j]->output_size,sizeof(float));
        copy_array(dx,dz,lstms[j]->output_size);
        free(dx);
        
        if(!j && input_error != NULL){
            input_error[i] = lstm_dinput(i,lstms[j]->output_size,matrix[j],lstms[j]);
            if(lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(input_error[i],dz,input_error[i],lstms[j]->output_size);
        }
    }
    
    free(dropout_output2);
    free(dz);
    return matrix;
    
}
/* this function returns the error dfioc and set the error of input_error through a back propagation passage
 * the dfioc returning error has this dimensions: m->layers*4*m->size
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
 * 
 * */
float*** bp_rmodel_lstm_opt(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window,int layers,lstm** lstms, float** input_error, lstm** lstms2){

   /* backpropagation passage*/

    int i,j, lstm_bp_flag;
    
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[layers-1]->output_size);
    float* dx; //here we store the modified output by dropout coming from the last lstm cell
    float* dz = (float*)calloc(lstms[layers-1]->output_size,sizeof(float)); //for residual dx
    float*** matrix = (float***)malloc(sizeof(float**)*layers);
    float** temp;
    int output_up = 0;
    for(i = 0; i < layers; i++){
        matrix[i] = NULL;
    }
    /*with i = 0 we should handle the h minus and c minus that are given by the params passed to the function*/
    for(i = window-1; i > 0; i--){
        for(j = layers-1; j >= 0; j--){
            dx = (float*)calloc(lstms[j]->output_size,sizeof(float));
            if(j < layers-1 && lstms[j+1]->residual_flag == LSTM_RESIDUAL)
                sum1D(dx,dz,dx,lstms[j]->output_size);
                
            if(j == layers-1 && i == window-1)
                lstm_bp_flag = 0;
            else if(j != layers-1 && i == window-1)
                lstm_bp_flag = 1;
                
            else if(j == layers-1 && i != window-1)
                lstm_bp_flag = 2;
                
            else
                lstm_bp_flag = 3;
            
            if(j != layers-1)
            output_up = lstms[j+1]->output_size;
            
            free(dropout_output2);
            dropout_output2 = (float*)malloc(sizeof(float)*lstms[j]->output_size);
            
            if(j == layers-1)
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_up,error_model[i],dx);

            if(j == layers-1){
                
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);
                
                if(j != 0){
                    if(i == window-1){
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j]->k_percentage,NULL,NULL);
                    }
                    else{
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,  lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,  lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j]->k_percentage,NULL,NULL);
                    
                    }
                }
                
                else{
                    if(i == window-1){
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j]->k_percentage,NULL,NULL);
                    
                    }
                    else{
                        if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,  lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                        else if(lstms[j]->training_mode == EDGE_POPUP)
                            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up,  lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j]->k_percentage,NULL,NULL);
                    }
                }
                
                
                if(matrix[j] != NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;

            }
            
            else if(j != layers-1 && j){
                
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);

                
                if(i == window-1){
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j+1]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j+1]->k_percentage,lstms2[j+1]->w_active_output_neurons,lstms2[j+1]->u_active_output_neurons);
                }
                else{
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j+1]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j+1]->k_percentage,lstms2[j+1]->w_active_output_neurons,lstms2[j+1]->u_active_output_neurons);
                }
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
                
            }
            
            else{
                
                get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);

                if(i == window-1){
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)    
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j+1]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j+1]->k_percentage,lstms2[j+1]->w_active_output_neurons,lstms2[j+1]->u_active_output_neurons);
                }
                else{
                    if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                        temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size, output_up,lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                    else if(lstms[j]->training_mode == EDGE_POPUP)
                        temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,lstms[j]->lstm_cell[i-1], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j+1]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j+1]->k_percentage,lstms2[j+1]->w_active_output_neurons,lstms2[j+1]->u_active_output_neurons);
                }
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
            }
            
            free(dz);
            dz = (float*)calloc(lstms[j]->output_size,sizeof(float));
            copy_array(dx,dz,lstms[j]->output_size);
            free(dx);
            
            if(!j && input_error != NULL){
                input_error[i] = lstm_dinput_opt(i,lstms[j]->output_size,matrix[j],lstms[j],lstms2[j]);
                if(lstms[j]->residual_flag == LSTM_RESIDUAL)
                    sum1D(input_error[i],dz,input_error[i],lstms[j]->output_size);
            }
            
        }
        
    }
    
    i = 0;
    /* computing back propagation just for the first lstm layers with hidden states defined by the previous batch*/
    for(j = layers-1; j >= 0; j--){
        dx = (float*)calloc(lstms[j]->output_size,sizeof(float));
        if(j < layers-1 && lstms[j+1]->residual_flag == LSTM_RESIDUAL)
            sum1D(dx,dz,dx,lstms[j]->output_size);
        if(j == layers-1 && i == window-1)
            lstm_bp_flag = 0;
        else if(j != layers-1 && i == window-1)
            lstm_bp_flag = 1;
            
        else if(j == layers-1 && i != window-1)
            lstm_bp_flag = 2;
            
        else
            lstm_bp_flag = 3;
        
        if(j != layers-1)
            output_up = lstms[j+1]->output_size;
            
        free(dropout_output2);
        dropout_output2 = (float*)malloc(sizeof(float)*lstms[j]->output_size);
        
        if(j == layers-1)
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_up,error_model[i],dx);

        if(j == layers-1){
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(j != 0){
                if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                else if(lstms[j]->training_mode == EDGE_POPUP)
                temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j]->k_percentage,NULL,NULL);
            }
            else{
                if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
                temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
                else if(lstms[j]->training_mode == EDGE_POPUP)
                temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u, lstms2[j]->d_biases, lstms2[j]->w, lstms2[j]->u, lstms[j]->lstm_z[i], dx,input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, lstms[j]->lstm_z[i+1],matrix[j],NULL,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j]->k_percentage,NULL,NULL);
            }
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        
        else if(j != layers-1 && j != 0){
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            else if(lstms[j]->training_mode == EDGE_POPUP)
            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, lstms[j-1]->out_up[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j+1]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j+1]->k_percentage,lstms2[j+1]->w_active_output_neurons,lstms2[j+1]->u_active_output_neurons);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
        }
        
        else{
            
            get_dropout_array(lstms[j]->output_size,lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            if(lstms[j]->training_mode == GRADIENT_DESCENT || lstms[j]->training_mode == FREEZE_TRAINING)
            temp = lstm_bp(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right);
            else if(lstms[j]->training_mode == EDGE_POPUP)
            temp = lstm_bp_edge_popup(lstm_bp_flag,lstms[j]->input_size,lstms[j]->output_size,output_up, lstms2[j]->d_w,lstms2[j]->d_u,lstms2[j]->d_biases,lstms2[j]->w,lstms2[j]->u,lstms[j]->lstm_z[i], dx, input_model[i],lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], lstms[j+1]->lstm_z[i], matrix[j+1], lstms[j]->lstm_z[i+1],matrix[j],lstms2[j+1]->w,lstms[j]->dropout_mask_up,lstms[j]->dropout_mask_right,lstms2[j]->w_active_output_neurons, lstms2[j]->u_active_output_neurons, lstms2[j+1]->w_indices,lstms2[j]->u_indices,lstms2[j]->d_w_scores,lstms2[j]->d_u_scores,lstms2[j+1]->k_percentage,lstms2[j+1]->w_active_output_neurons,lstms2[j+1]->u_active_output_neurons);
            if(matrix[j] != NULL)
                free_matrix(matrix[j],4);
            matrix[j] = temp;
            
        }
        
        free(dz);
        dz = (float*)calloc(lstms[j]->output_size,sizeof(float));
        copy_array(dx,dz,lstms[j]->output_size);
        free(dx);
        
        if(!j && input_error != NULL){
            input_error[i] = lstm_dinput_opt(i,lstms[j]->output_size,matrix[j],lstms[j],lstms2[j]);
            if(lstms[j]->residual_flag == LSTM_RESIDUAL)
                sum1D(input_error[i],dz,input_error[i],lstms[j]->output_size);
        }
    }
    
    free(dropout_output2);
    free(dz);
    return matrix;
    
}


/* This function returs the total number of weights in the rmodel m
 * 
 * Input
 * 
 *             @ rmodel* m:= the recurrent model
 * 
 * */
uint64_t count_weights_rmodel(rmodel* m){
    /*
    int i;
    uint64_t sum = 0;
    for(i = 0; i < m->n_lstm; i++){
        if(m->lstms[i]->feed_forward_flag == FULLY_FEED_FORWARD)
            sum+=m->lstms[i]->size*m->lstms[i]->size*8;
        else
            sum+=m->lstms[i]->size*m->lstms[i]->size*8*m->lstms[i]->k_percentage;
    }
        
    for(i = 0; i < m->n_lstm; i++){
        if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
            sum+=(m->lstms[i]->window/m->lstms[i]->n_grouped_cell)*m->lstms[i]->bns[0]->vector_dim;
        }
    }
    return sum;
    * */
    return 0;
}








/* this function return a vector of float of a lstm cell for the dL/Dinput of the same lstms cell
 * 
 * Inputs:
 * 
 * 
 *                 @ int index:= the index of the unrolled lstm cell
 *                 @ int output:= the size of the lstm cell
 *                 @ float** returning error the dfioc of the lstm cell
 *                 @ lstm* lstms:= the lstm cell
 * 
 * */
float* lstm_dinput(int index, int output, float** returning_error, lstm* lstms){
    
    int k,k2;
    
    float* ret_err;
    
    float* temp = (float*)malloc(sizeof(float)*output);
    float* temp2 = (float*)malloc(sizeof(float)*output);
    float* temp3 = (float*)calloc(lstms->input_size,sizeof(float));
        
    ret_err = (float*)calloc(lstms->input_size,sizeof(float));
    
    
    sigmoid_array(lstms->lstm_z[index][1],temp,output);//obtaining i
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[1][k] && !lstms->u_active_output_neurons[1][k]){
            temp[k] = 0;
        }
    }
    dot1D(temp,returning_error[3],temp,output);//computing dc*i
    sigmoid_array(lstms->lstm_z[index][3],temp2,output);//dzc
    derivative_sigmoid_array_given_the_sigmoid(temp2,temp2,output);
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[3][k] && !lstms->u_active_output_neurons[3][k]){
            temp2[k] = 0;
        }
    }
    dot1D(temp,temp2,temp,output);
    
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms->w[3][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms->k_percentage; k < output*lstms->input_size; k++){
            if(lstms->w_active_output_neurons[3][(int)(lstms->w_indices[3][k]/lstms->input_size)]){
                temp3[lstms->w_indices[3][k]%lstms->input_size] = lstms->w[3][lstms->w_indices[3][k]]*temp[(int)(lstms->w_indices[3][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
    
    free(temp3);
    temp3 = (float*)calloc(lstms->input_size,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][2],temp2,output);//dzo
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[2][k] && !lstms->u_active_output_neurons[2][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[2],temp2,temp,output);
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms->w[2][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms->k_percentage; k < output*lstms->input_size; k++){
            if(lstms->w_active_output_neurons[2][(int)(lstms->w_indices[2][k]/lstms->input_size)]){
                temp3[lstms->w_indices[2][k]%lstms->input_size] = lstms->w[2][lstms->w_indices[2][k]]*temp[(int)(lstms->w_indices[2][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
    
    free(temp3);
    temp3 = (float*)calloc(lstms->input_size,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][1],temp2,output);//dzi
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[1][k] && !lstms->u_active_output_neurons[1][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[1],temp2,temp,output);
    
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms->w[1][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms->k_percentage; k < output*lstms->input_size; k++){
            if(lstms->w_active_output_neurons[1][(int)(lstms->w_indices[1][k]/lstms->input_size)]){
                temp3[lstms->w_indices[1][k]%lstms->input_size] = lstms->w[1][lstms->w_indices[1][k]]*temp[(int)(lstms->w_indices[1][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
    
    free(temp3);
    temp3 = (float*)calloc(lstms->input_size,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][0],temp2,output);//dzf
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[0][k] && !lstms->u_active_output_neurons[0][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[0],temp2,temp,output);
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms->w[0][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms->k_percentage; k < output*lstms->input_size; k++){
            if(lstms->w_active_output_neurons[0][(int)(lstms->w_indices[0][k]/lstms->input_size)]){
                temp3[lstms->w_indices[0][k]%lstms->input_size] = lstms->w[0][lstms->w_indices[0][k]]*temp[(int)(lstms->w_indices[0][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
        
    free(temp);
    free(temp2);
    free(temp3);
    
    return ret_err;
}

/* this function return a vector of float of a lstm cell for the dL/Dinput of the same lstms cell
 * 
 * Inputs:
 * 
 * 
 *                 @ int index:= the index of the unrolled lstm cell
 *                 @ int output:= the size of the lstm cell
 *                 @ float** returning error the dfioc of the lstm cell
 *                 @ lstm* lstms:= the lstm cell
 * 
 * */
float* lstm_dinput_opt(int index, int output, float** returning_error, lstm* lstms, lstm* lstms2){
    
    int k,k2;
    
    float* ret_err;
    
    float* temp = (float*)malloc(sizeof(float)*output);
    float* temp2 = (float*)malloc(sizeof(float)*output);
    float* temp3 = (float*)calloc(lstms->input_size,sizeof(float));
        
    ret_err = (float*)calloc(lstms->input_size,sizeof(float));
    
    
    sigmoid_array(lstms->lstm_z[index][1],temp,output);//obtaining i
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[1][k] && !lstms2->u_active_output_neurons[1][k]){
            temp[k] = 0;
        }
    }
    dot1D(temp,returning_error[3],temp,output);//computing dc*i
    sigmoid_array(lstms->lstm_z[index][3],temp2,output);//dzc
    derivative_sigmoid_array_given_the_sigmoid(temp2,temp2,output);
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[3][k] && !lstms2->u_active_output_neurons[3][k]){
            temp2[k] = 0;
        }
    }
    dot1D(temp,temp2,temp,output);
    
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms2->w[3][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms2->k_percentage; k < output*lstms->input_size; k++){
            if(lstms2->w_active_output_neurons[3][(int)(lstms2->w_indices[3][k]/lstms->input_size)]){
                temp3[lstms2->w_indices[3][k]%lstms->input_size] = lstms2->w[3][lstms2->w_indices[3][k]]*temp[(int)(lstms2->w_indices[3][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
    
    free(temp3);
    temp3 = (float*)calloc(lstms->input_size,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][2],temp2,output);//dzo
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[2][k] && !lstms2->u_active_output_neurons[2][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[2],temp2,temp,output);
    if(lstms2->feed_forward_flag != EDGE_POPUP && lstms2->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms2->w[2][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms2->k_percentage; k < output*lstms->input_size; k++){
            if(lstms2->w_active_output_neurons[2][(int)(lstms2->w_indices[2][k]/lstms->input_size)]){
                temp3[lstms2->w_indices[2][k]%lstms->input_size] = lstms2->w[2][lstms2->w_indices[2][k]]*temp[(int)(lstms2->w_indices[2][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
    
    free(temp3);
    temp3 = (float*)calloc(lstms->input_size,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][1],temp2,output);//dzi
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[1][k] && !lstms2->u_active_output_neurons[1][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[1],temp2,temp,output);
    
    if(lstms2->feed_forward_flag != EDGE_POPUP && lstms2->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms2->w[1][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms2->k_percentage; k < output*lstms->input_size; k++){
            if(lstms2->w_active_output_neurons[1][(int)(lstms2->w_indices[1][k]/lstms->input_size)]){
                temp3[lstms2->w_indices[1][k]%lstms->input_size] = lstms2->w[1][lstms2->w_indices[1][k]]*temp[(int)(lstms2->w_indices[1][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
    
    free(temp3);
    temp3 = (float*)calloc(lstms->input_size,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][0],temp2,output);//dzf
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[0][k] && !lstms2->u_active_output_neurons[0][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[0],temp2,temp,output);
    if(lstms2->feed_forward_flag != EDGE_POPUP && lstms2->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < lstms->input_size; k2++){
                temp3[k2] = lstms2->w[0][k*lstms->input_size+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*lstms->input_size*lstms->k_percentage; k < output*lstms->input_size; k++){
            if(lstms2->w_active_output_neurons[0][(int)(lstms2->w_indices[0][k]/lstms->input_size)]){
                temp3[lstms2->w_indices[0][k]%lstms->input_size] = lstms2->w[0][lstms2->w_indices[0][k]]*temp[(int)(lstms2->w_indices[0][k]/lstms->input_size)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,lstms->input_size);
        
    free(temp);
    free(temp2);
    free(temp3);
    
    return ret_err;
}



/* this function return a vector of float of a lstm cell for the dL/Dh of the same lstms cell
 * 
 * Inputs:
 * 
 * 
 *                 @ int index:= the index of the unrolled lstm cell
 *                 @ int output:= the size of the lstm cell
 *                 @ float** returning error the dfioc of the lstm cell
 *                 @ lstm* lstms:= the lstm cell
 * 
 * */
float* lstm_dh(int index, int output, float** returning_error, lstm* lstms){
    
    int k,k2;
    
    float* ret_err;
    
    float* temp = (float*)malloc(sizeof(float)*output);
    float* temp2 = (float*)malloc(sizeof(float)*output);
    float* temp3 = (float*)calloc(output,sizeof(float));
        
    ret_err = (float*)calloc(output,sizeof(float));
    
    
    sigmoid_array(lstms->lstm_z[index][1],temp,output);//obtaining i
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[1][k] && !lstms->u_active_output_neurons[1][k]){
            temp[k] = 0;
        }
    }
    dot1D(temp,returning_error[3],temp,output);//computing dc*i
    sigmoid_array(lstms->lstm_z[index][3],temp2,output);//dzc
    derivative_sigmoid_array_given_the_sigmoid(temp2,temp2,output);
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[3][k] && !lstms->u_active_output_neurons[3][k]){
            temp2[k] = 0;
        }
    }
    dot1D(temp,temp2,temp,output);
    
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms->u[3][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms->k_percentage; k < output*output; k++){
            if(lstms->u_active_output_neurons[3][(int)(lstms->u_indices[3][k]/output)]){
                temp3[lstms->u_indices[3][k]%output] = lstms->u[3][lstms->u_indices[3][k]]*temp[(int)(lstms->u_indices[3][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][2],temp2,output);//dzo
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[2][k] && !lstms->u_active_output_neurons[2][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[2],temp2,temp,output);
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms->u[2][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms->k_percentage; k < output*output; k++){
            if(lstms->u_active_output_neurons[2][(int)(lstms->u_indices[2][k]/output)]){
                temp3[lstms->u_indices[2][k]%output] = lstms->u[2][lstms->u_indices[2][k]]*temp[(int)(lstms->u_indices[2][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][1],temp2,output);//dzi
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[1][k] && !lstms->u_active_output_neurons[1][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[1],temp2,temp,output);
    
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms->u[1][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms->k_percentage; k < output*output; k++){
            if(lstms->u_active_output_neurons[1][(int)(lstms->u_indices[1][k]/output)]){
                temp3[lstms->u_indices[1][k]%output] = lstms->u[1][lstms->u_indices[1][k]]*temp[(int)(lstms->u_indices[1][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][0],temp2,output);//dzf
    for(k = 0; k < output; k++){
        if(!lstms->w_active_output_neurons[0][k] && !lstms->u_active_output_neurons[0][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[0],temp2,temp,output);
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms->u[0][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms->k_percentage; k < output*output; k++){
            if(lstms->u_active_output_neurons[0][(int)(lstms->u_indices[0][k]/output)]){
                temp3[lstms->u_indices[0][k]%output] = lstms->u[0][lstms->u_indices[0][k]]*temp[(int)(lstms->u_indices[0][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
        
    free(temp);
    free(temp2);
    free(temp3);
    
    return ret_err;
}
/* this function return a vector of float of a lstm cell for the dL/Dh of the same lstms cell
 * 
 * Inputs:
 * 
 * 
 *                 @ int index:= the index of the unrolled lstm cell
 *                 @ int output:= the size of the lstm cell
 *                 @ float** returning error the dfioc of the lstm cell
 *                 @ lstm* lstms:= the lstm cell
 * 
 * */
float* lstm_dh_opt(int index, int output, float** returning_error, lstm* lstms, lstm* lstms2){
    
    int k,k2;
    
    float* ret_err;
    
    float* temp = (float*)malloc(sizeof(float)*output);
    float* temp2 = (float*)malloc(sizeof(float)*output);
    float* temp3 = (float*)calloc(output,sizeof(float));
        
    ret_err = (float*)calloc(output,sizeof(float));
    
    
    sigmoid_array(lstms->lstm_z[index][1],temp,output);//obtaining i
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[1][k] && !lstms2->u_active_output_neurons[1][k]){
            temp[k] = 0;
        }
    }
    dot1D(temp,returning_error[3],temp,output);//computing dc*i
    sigmoid_array(lstms->lstm_z[index][3],temp2,output);//dzc
    derivative_sigmoid_array_given_the_sigmoid(temp2,temp2,output);
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[3][k] && !lstms2->u_active_output_neurons[3][k]){
            temp2[k] = 0;
        }
    }
    dot1D(temp,temp2,temp,output);
    
    if(lstms->feed_forward_flag != EDGE_POPUP && lstms->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms2->u[3][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms2->k_percentage; k < output*output; k++){
            if(lstms2->u_active_output_neurons[3][(int)(lstms2->u_indices[3][k]/output)]){
                temp3[lstms2->u_indices[3][k]%output] = lstms2->u[3][lstms2->u_indices[3][k]]*temp[(int)(lstms2->u_indices[3][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][2],temp2,output);//dzo
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[2][k] && !lstms2->u_active_output_neurons[2][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[2],temp2,temp,output);
    if(lstms2->feed_forward_flag != EDGE_POPUP && lstms2->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms2->u[2][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms2->k_percentage; k < output*output; k++){
            if(lstms2->u_active_output_neurons[2][(int)(lstms2->u_indices[2][k]/output)]){
                temp3[lstms2->u_indices[2][k]%output] = lstms2->u[2][lstms2->u_indices[2][k]]*temp[(int)(lstms2->u_indices[2][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][1],temp2,output);//dzi
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[1][k] && !lstms2->u_active_output_neurons[1][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[1],temp2,temp,output);
    
    if(lstms2->feed_forward_flag != EDGE_POPUP && lstms2->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms2->u[1][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms2->k_percentage; k < output*output; k++){
            if(lstms2->u_active_output_neurons[1][(int)(lstms2->u_indices[1][k]/output)]){
                temp3[lstms2->u_indices[1][k]%output] = lstms2->u[1][lstms2->u_indices[1][k]]*temp[(int)(lstms2->u_indices[1][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][0],temp2,output);//dzf
    for(k = 0; k < output; k++){
        if(!lstms2->w_active_output_neurons[0][k] && !lstms2->u_active_output_neurons[0][k]){
            temp2[k] = 0;
        }
    }
    dot1D(returning_error[0],temp2,temp,output);
    if(lstms2->feed_forward_flag != EDGE_POPUP && lstms2->training_mode != EDGE_POPUP){
        for(k = 0; k < output; k++){
            for(k2 = 0; k2 < output; k2++){
                temp3[k2] = lstms2->u[0][k*output+k2]*temp[k];
            }
        }
    }
    
    else{
        for(k = output*output*lstms->k_percentage; k < output*output; k++){
            if(lstms2->u_active_output_neurons[0][(int)(lstms2->u_indices[0][k]/output)]){
                temp3[lstms2->u_indices[0][k]%output] = lstms2->u[0][lstms2->u_indices[0][k]]*temp[(int)(lstms2->u_indices[0][k]/output)];
            }
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
        
    free(temp);
    free(temp2);
    free(temp3);
    
    return ret_err;
}


/* This function computes the feed forward of a rmodel with group normalization params
 * 
  * Input:
 * 
 *             @ float** hidden_state:= the hidden states of the previous cells:= layers x size
 *             @ float** cell_states:= layers x size
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window x size 
 *             @ rmodel* m:= the rmodel that is gonna compute the feed forward
 * 
 * 
 * */ 
void ff_rmodel(float** hidden_states, float** cell_states, float** input_model, rmodel* m){
    if(m == NULL)
        return;
    int i,j,k,z;
    float** temp = (float**)malloc(sizeof(float*)*m->window);
    for(i = 0; i < m->window; i++){
        temp[i] = input_model[i];
    }
    int n_cells;
    for(i = 0, n_cells = 1;i < m->layers; i+=n_cells){
        n_cells = 1;
        for(k = i; k < m->layers && m->lstms[k]->norm_flag != GROUP_NORMALIZATION; k++,n_cells++);
        if(k == m->layers){ n_cells--; k--;}
        ff_rmodel_lstm(&hidden_states[i],&cell_states[i],temp,m->window,n_cells,&m->lstms[i]);
        for(j = 0; j < m->window; j++){
            temp[j] = m->lstms[k]->out_up[j];
        }
        
        if(m->lstms[k]->norm_flag == GROUP_NORMALIZATION){
            for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell; j++){
                batch_normalization_feed_forward(m->lstms[k]->n_grouped_cell,&temp[j*m->lstms[k]->n_grouped_cell],m->lstms[k]->bns[j]->temp_vectors,m->lstms[k]->bns[j]->vector_dim,m->lstms[k]->bns[j]->gamma,m->lstms[k]->bns[j]->beta,m->lstms[k]->bns[j]->mean,m->lstms[k]->bns[j]->var,m->lstms[k]->bns[j]->outputs,m->lstms[k]->bns[j]->epsilon);
            }
            
            for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell; j++){
                for(z = 0; z < m->lstms[k]->n_grouped_cell; z++){
                    temp[j*m->lstms[k]->n_grouped_cell+z] = m->lstms[k]->bns[j]->outputs[z];
                }
            }
            
        }
    }
    
    free(temp);
}

/* This function computes the feed forward of a rmodel with group normalization params
 * 
  * Input:
 * 
 *             @ float** hidden_state:= the hidden states of the previous cells:= layers x size
 *             @ float** cell_states:= layers x size
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window x size 
 *             @ rmodel* m:= the rmodel that is gonna compute the feed forward
 * 
 * 
 * */ 
void ff_rmodel_opt(float** hidden_states, float** cell_states, float** input_model, rmodel* m, rmodel* m2){
    if(m == NULL)
        return;
    int i,j,k,z;
    float** temp = (float**)malloc(sizeof(float*)*m->window);
    for(i = 0; i < m->window; i++){
        temp[i] = input_model[i];
    }
    int n_cells;
    for(i = 0, n_cells = 1;i < m->layers; i+=n_cells){
        n_cells = 1;
        for(k = i; k < m->layers && m->lstms[k]->norm_flag != GROUP_NORMALIZATION; k++,n_cells++);
        if(k == m->layers){ n_cells--; k--;}
        ff_rmodel_lstm_opt(&hidden_states[i],&cell_states[i],temp,m->window,n_cells,&m->lstms[i],&m2->lstms[i]);
        for(j = 0; j < m->window; j++){
            temp[j] = m->lstms[k]->out_up[j];
        }
        
        if(m->lstms[k]->norm_flag == GROUP_NORMALIZATION){
            for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell; j++){
                batch_normalization_feed_forward(m->lstms[k]->n_grouped_cell,&temp[j*m->lstms[k]->n_grouped_cell],m->lstms[k]->bns[j]->temp_vectors,m->lstms[k]->bns[j]->vector_dim,m2->lstms[k]->bns[j]->gamma,m2->lstms[k]->bns[j]->beta,m->lstms[k]->bns[j]->mean,m->lstms[k]->bns[j]->var,m->lstms[k]->bns[j]->outputs,m->lstms[k]->bns[j]->epsilon);
            }
            
            for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell; j++){
                for(z = 0; z < m->lstms[k]->n_grouped_cell; z++){
                    temp[j*m->lstms[k]->n_grouped_cell+z] = m->lstms[k]->bns[j]->outputs[z];
                }
            }
            
        }
    }
    
    free(temp);
}


/* This function computes the backpropagation of a rmodel with grouped normalization layers
 * 
 *  * Inputs:
 * 
 * 
 *             @ float** hidden_states:= the hidden sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** cell_states:= the cell sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** input_model:= the input passed to the model, dimensions: m->window*m->size
 *             @ float** error_model:= the error of the model, dimensions: m->window*m->size
 *             @ rmodel* m:= the recurrent model
 *             @ float** input_error:= the error of the inputs of this model, dimensions: m->window*m->size, must be initialized only with m->window
 * 
 * */
float*** bp_rmodel(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m, float** input_error){
    if(m == NULL)
        return NULL;
    float*** ret = (float***)malloc(sizeof(float**)*m->layers);//Storing all the returned values of bp of lstms
    float*** ret2;//to handle the returned values of bp of lstms
    float** input_error3 = (float**)malloc(sizeof(float*)*m->window);//input error for lstms
    float** error2_model = (float**)malloc(sizeof(float*)*m->window);//error propagated
    int i,j,ret_count = m->layers-1,z;
    float** temp = (float**)malloc(sizeof(float*)*m->window);// inputs of lstms
    for(i = 0; i < m->window; i++){
        error2_model[i] = (float*)calloc(m->lstms[m->layers-1]->output_size,sizeof(float));
        copy_array(error_model[i],error2_model[i],m->lstms[m->layers-1]->output_size);
    }
    int flagg = 0;
    int k = 0;
    while(k > -1){
        int flag = 0;
        for(k = ret_count; k >= 0; k--){
            if(m->lstms[k]->norm_flag == GROUP_NORMALIZATION){
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
                for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell; j++){
                    for(z = 0; z < m->lstms[k]->n_grouped_cell; z++){
                        temp[j*m->lstms[k]->n_grouped_cell+z] = m->lstms[k]->bns[j]->outputs[z];
                    }
                }
                ret2 = bp_rmodel_lstm(&hidden_states[k+1],&cell_states[k+1],temp,error2_model,m->window,n_cells,&m->lstms[k+1],input_error3);
                for(j = 0; j < m->window; j++){
					free(error2_model[j]);
					error2_model[j] = (float*)calloc(m->lstms[k+1]->input_size,sizeof(float));
                    copy_array(input_error3[j],error2_model[j],m->lstms[k+1]->input_size);
                    free(input_error3[j]);
                }
                for(j = ret_count; j > ret_count-n_cells; j--){
                    ret[j] = ret2[n_cells-1-(ret_count-j)];
                }
                free(ret2);
            }
            int zz;
            for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell;j++){
                batch_normalization_back_prop(m->lstms[k]->n_grouped_cell,&m->lstms[k]->out_up[j*m->lstms[k]->n_grouped_cell],m->lstms[k]->bns[j]->temp_vectors,m->lstms[k]->bns[j]->vector_dim,m->lstms[k]->bns[j]->gamma,m->lstms[k]->bns[j]->beta,m->lstms[k]->bns[j]->mean,m->lstms[k]->bns[j]->var,&error2_model[j*m->lstms[k]->n_grouped_cell],m->lstms[k]->bns[j]->d_gamma,m->lstms[k]->bns[j]->d_beta,m->lstms[k]->bns[j]->error2,m->lstms[k]->bns[j]->temp1,m->lstms[k]->bns[j]->temp2,m->lstms[k]->bns[j]->epsilon);
            }
            for(j = 0; j < m->window/m->lstms[k]->n_grouped_cell; j++){
                for(z = 0; z < m->lstms[k]->n_grouped_cell; z++){
					free(error2_model[j*m->lstms[k]->n_grouped_cell+z]);
					error2_model[j*m->lstms[k]->n_grouped_cell+z] = (float*)calloc(m->lstms[k]->output_size,sizeof(float));
                    copy_array(m->lstms[k]->bns[j]->error2[z],error2_model[j*m->lstms[k]->n_grouped_cell+z],m->lstms[k]->output_size);
                }
            }
            
            if(n_cells)
                ret_count = k;
        }
        
        else{
            
            int n_cells = ret_count-k;
            int k2 = k;
            if(k < 0) k = 0;
            ret2 = bp_rmodel_lstm(&hidden_states[k],&cell_states[k],input_model,error2_model,m->window,n_cells,&m->lstms[k],input_error);
            k = k2;

            for(j = ret_count; j > k; j--){
                ret[j] = ret2[j];
            }    
            free(ret2);
        }
    }
    
    free_matrix(error2_model,m->window);
    free(input_error3);
    free(temp);
    return ret;
    
}
/* This function computes the backpropagation of a rmodel with grouped normalization layers
 * 
 *  * Inputs:
 * 
 * 
 *             @ float** hidden_states:= the hidden sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** cell_states:= the cell sates passed to each first orizontal cell, dimensions:m->layer*m->size
 *             @ float** input_model:= the input passed to the model, dimensions: m->window*m->size
 *             @ float** error_model:= the error of the model, dimensions: m->window*m->size
 *             @ rmodel* m:= the recurrent model
 *             @ float** input_error:= the error of the inputs of this model, dimensions: m->window*m->size, must be initialized only with m->window
 * 
 * */
float*** bp_rmodel_opt(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m, float** input_error, rmodel* m2){
    if(m == NULL)
        return NULL;
    float*** ret = (float***)malloc(sizeof(float**)*m->layers);//Storing all the returned values of bp of lstms
    float*** ret2;//to handle the returned values of bp of lstms
    float** input_error3 = (float**)malloc(sizeof(float*)*m->window);//input error for lstms
    float** error2_model = (float**)malloc(sizeof(float*)*m->window);//error propagated
    int i,j,ret_count = m->layers-1,z;
    float** temp = (float**)malloc(sizeof(float*)*m->window);// inputs of lstms
    for(i = 0; i < m->window; i++){
        error2_model[i] = (float*)calloc(m->lstms[m->layers-1]->output_size,sizeof(float));
        copy_array(error_model[i],error2_model[i],m->lstms[m->layers-1]->output_size);
    }
    int flagg = 0;
    int k = 0;
    while(k > -1){
        int flag = 0;
        for(k = ret_count; k >= 0; k--){
            if(m->lstms[k]->norm_flag == GROUP_NORMALIZATION){
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
                for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell; j++){
                    for(z = 0; z < m->lstms[k]->n_grouped_cell; z++){
                        temp[j*m->lstms[k]->n_grouped_cell+z] = m->lstms[k]->bns[j]->outputs[z];
                    }
                }
                ret2 = bp_rmodel_lstm_opt(&hidden_states[k+1],&cell_states[k+1],temp,error2_model,m->window,n_cells,&m->lstms[k+1],input_error3,&m2->lstms[k+1]);
                for(j = 0; j < m->window; j++){
					free(error2_model[j]);
					error2_model[j] = (float*)calloc(m->lstms[k+1]->input_size,sizeof(float));
                    copy_array(input_error3[j],error2_model[j],m->lstms[k+1]->input_size);
                    free(input_error3[j]);
                }
                for(j = ret_count; j > ret_count-n_cells; j--){
                    ret[j] = ret2[n_cells-1-(ret_count-j)];
                }
                free(ret2);
            }
            int zz;
            for(j = 0; j < m->lstms[k]->window/m->lstms[k]->n_grouped_cell;j++){
                batch_normalization_back_prop(m->lstms[k]->n_grouped_cell,&m->lstms[k]->out_up[j*m->lstms[k]->n_grouped_cell],m->lstms[k]->bns[j]->temp_vectors,m->lstms[k]->bns[j]->vector_dim,m2->lstms[k]->bns[j]->gamma,m2->lstms[k]->bns[j]->beta,m->lstms[k]->bns[j]->mean,m->lstms[k]->bns[j]->var,&error2_model[j*m->lstms[k]->n_grouped_cell],m->lstms[k]->bns[j]->d_gamma,m->lstms[k]->bns[j]->d_beta,m->lstms[k]->bns[j]->error2,m->lstms[k]->bns[j]->temp1,m->lstms[k]->bns[j]->temp2,m->lstms[k]->bns[j]->epsilon);
            }
            for(j = 0; j < m->window/m->lstms[k]->n_grouped_cell; j++){
                for(z = 0; z < m->lstms[k]->n_grouped_cell; z++){
					free(error2_model[j*m->lstms[k]->n_grouped_cell+z]);
					error2_model[j*m->lstms[k]->n_grouped_cell+z] = (float*)calloc(m->lstms[k]->output_size,sizeof(float));
                    copy_array(m->lstms[k]->bns[j]->error2[z],error2_model[j*m->lstms[k]->n_grouped_cell+z],m->lstms[k]->output_size);
                }
            }
            
            if(n_cells)
                ret_count = k;
        }
        
        else{
            
            int n_cells = ret_count-k;
            int k2 = k;
            if(k < 0) k = 0;
            ret2 = bp_rmodel_lstm_opt(&hidden_states[k],&cell_states[k],input_model,error2_model,m->window,n_cells,&m->lstms[k],input_error,&m2->lstms[k]);
            k = k2;

            for(j = ret_count; j > k; j--){
                ret[j] = ret2[j];
            }    
            free(ret2);
        }
    }
    
    free_matrix(error2_model,m->window);
    free(input_error3);
    free(temp);
    return ret;
    
}


