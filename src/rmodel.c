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
    /*There is no check if the sizes match or not, this happen during the feed forward*/
    
    m->layers = layers;
    m->n_lstm = n_lstm;
    m->sla = sla;
    m->lstms = lstms;
    m->window = window;
    m->hidden_state_mode = hidden_state_mode;
        
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
    
    fw = fopen(s,"a");
    
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
void ff_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, int window, int size, int layers, lstm** lstms){    
    
    //allocation of the resources

    float* dropout_output = (float*)malloc(sizeof(float)*lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    
    int i,j;
    
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
                    
                    lstm_ff(input_model[i], dropout_output2, cell_states[j], lstms[j]->lstm_cell[i], lstms[j]->lstm_hidden[i], lstms[j]->w, lstms[j]->u, lstms[j]->biases, lstms[j]->lstm_z[i], lstms[j]->size);
                }

                else{
                    
                    get_dropout_array(lstms[j]->size,lstms[j]->dropout_mask_right,lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,lstms[j]->dropout_threshold_right,dropout_output2,lstms[j]->size);
                        
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
float*** bp_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window, int size,int layers,lstm** lstms, float** input_error){

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
            
            if(!j && input_error != NULL)
                input_error[i] = lstm_dinput(i,lstms[j]->size,matrix[j],lstms[j]);
            
        }
        
    }
    
    i = 0;
    /* computing back propagation just for the first lstm layers with hidden states defined by the previous batch*/
    for(j = layers-1; j >= 0; j--){
            
        dx = (float*)calloc(lstms[0]->size,sizeof(float));
        
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
        free(dx);
        
        if(!j && input_error != NULL)
                input_error[i] = lstm_dinput(i,lstms[j]->size,matrix[j],lstms[j]);
    }
    
    free(dropout_output);
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
int count_weights_rmodel(rmodel* m){
    int i,sum = 0;
    sum+=m->n_lstm*m->lstms[0]->size*m->lstms[0]->size*8;
    for(i = 0; i < m->layers; i++){
        if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
            sum+=(m->lstms[i]->window/m->lstms[i]->n_grouped_cell)*m->lstms[i]->bns[0]->vector_dim;
        }
    }
    return m->n_lstm*m->lstms[0]->size*m->lstms[0]->size*8;
}


/* This function can update the rmodel of the network using the adam algorithm or the nesterov momentum
 * 
 * Input:
 * 
 *             @ rmodel* m:= the recurrent model that must be update
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
void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t){
    if(m == NULL)
        return;
    
    int i,count = 0,count2 = 0,j,k = 0;
    
    for(i = 0; i < m->layers; i++){
        if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
            count++;
            count2+=m->lstms[i]->window/m->lstms[i]->n_grouped_cell;
        }
    }
    
    bmodel* bm = NULL;
    if(count){
        bm = (bmodel*)malloc(sizeof(bmodel*));
        bm->n_bn = count2;
        bm->bns = (bn**)malloc(sizeof(bn*)*count2);
    
        for(i = 0; i < m->layers; i++){
            if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
                for(j = 0; j < m->lstms[i]->window/m->lstms[i]->n_grouped_cell; j++){
                    bm->bns[k] = m->lstms[i]->bns[j];
                    k++;
                }
            }
        }
        
        update_bmodel(bm,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda);
        
        if(gradient_descent_flag == ADAM){
            (*b1)/=BETA1_ADAM;
            (*b2)/=BETA2_ADAM;
        }
        
        free(bm->bns);
        free(bm);
    }
    
    lambda*=(float)mini_batch_size;
    
    if(regularization == L2_REGULARIZATION)
        add_l2_lstm_layer(m,total_number_weights,lambda);
    
    
    
    if(gradient_descent_flag == NESTEROV)
        update_lstm_layer_nesterov(m,lr,momentum,mini_batch_size);
    
    
    else if(gradient_descent_flag == ADAM){
        update_lstm_layer_adam(m,lr,mini_batch_size, (*b1), (*b2));
        (*b1)*=BETA1_ADAM;
        (*b2)*=BETA2_ADAM;
    }
    
    else if(gradient_descent_flag == RADAM){
        update_lstm_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t);
        (*b1)*=BETA1_ADAM;
        (*b2)*=BETA2_ADAM;
        (*t)++;
    }     
    

}


/* This function sum the partial derivatives in rmodel m1 and m2 in m3
 * 
 * Input:
 *     
 *             @ rmodel* m:= first input rmodel
 *             @ rmodel* m2:= second input rmodel
 *             @ rmodel* m3:= output rmodel
 * 
 * */
void sum_rmodel_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    sum_lstm_layers_partial_derivatives(m,m2,m3);
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
    float* temp3 = (float*)calloc(output,sizeof(float));
        
    ret_err = (float*)calloc(output,sizeof(float));
    
    
    sigmoid_array(lstms->lstm_z[index][1],temp,output);//obtaining i
    dot1D(temp,returning_error[3],temp,output);//computing dc*i
    derivative_tanhh_array(lstms->lstm_z[index][3],temp2,output);//dzc
    dot1D(temp,temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->w[3][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][2],temp2,output);//dzo
    dot1D(returning_error[2],temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->w[2][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][1],temp2,output);//dzi
    dot1D(returning_error[1],temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->w[1][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][0],temp2,output);//dzf
    dot1D(returning_error[0],temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->w[0][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
        
    free(temp);
    free(temp2);
    free(temp3);
    
    return ret_err;
}


float* lstm_dh(int index, int output, float** returning_error, lstm* lstms){
    
    int k,k2;
    
    float* ret_err;
    
    float* temp = (float*)malloc(sizeof(float)*output);
    float* temp2 = (float*)malloc(sizeof(float)*output);
    float* temp3 = (float*)calloc(output,sizeof(float));
        
    ret_err = (float*)calloc(output,sizeof(float));
    
    
    sigmoid_array(lstms->lstm_z[index][1],temp,output);//obtaining i
    dot1D(temp,returning_error[3],temp,output);//computing dc*i
    derivative_tanhh_array(lstms->lstm_z[index][3],temp2,output);//dzc
    dot1D(temp,temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->u[3][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][2],temp2,output);//dzo
    dot1D(returning_error[2],temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->u[2][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][1],temp2,output);//dzi
    dot1D(returning_error[1],temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->u[1][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    free(temp3);
    temp3 = (float*)calloc(output,sizeof(float));
    
    derivative_sigmoid_array(lstms->lstm_z[index][0],temp2,output);//dzf
    dot1D(returning_error[0],temp2,temp,output);
    for(k = 0; k < output; k++){
        for(k2 = 0; k2 < output; k2++){
            temp3[k2] = lstms->u[0][k*output+k2]*temp[k];
        }
    }
    
    sum1D(ret_err,temp3,ret_err,output);
    
    dot1D(ret_err,lstms->dropout_mask_right,ret_err,output);
    
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
    int i = 0,j,k,z;
    
    float** temp = (float**)malloc(sizeof(float*)*m->window);
    for(i = 0; i < m->window; i++){
        temp[i] = input_model[i];
    }
    i = 0;
    while(i < m->layers){
        for(k = 0; i < m->layers; i++){
            k++;
            if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
                i++;
                break;
            }
        }
        
        
        ff_rmodel_lstm(&hidden_states[i-k],&cell_states[i-k],temp,m->window,m->lstms[0]->size,k,&m->lstms[i-k]);
        for(j = 0; j < m->window; j++){
            temp[j] = m->lstms[i-1]->out_up[j];
        }
        
        if(m->lstms[i-1]->norm_flag == GROUP_NORMALIZATION){
            for(j = 0; j < m->lstms[i-1]->window/m->lstms[i-1]->n_grouped_cell; j++){
                batch_normalization_feed_forward(m->lstms[i-1]->n_grouped_cell,&temp[j*m->lstms[i-1]->n_grouped_cell],m->lstms[i-1]->bns[j]->temp_vectors,m->lstms[i-1]->bns[j]->vector_dim,m->lstms[i-1]->bns[j]->gamma,m->lstms[i-1]->bns[j]->beta,m->lstms[i-1]->bns[j]->mean,m->lstms[i-1]->bns[j]->var,m->lstms[i-1]->bns[j]->outputs,m->lstms[i-1]->bns[j]->epsilon);
            }
            
            for(j = 0; j < m->lstms[i-1]->window/m->lstms[i-1]->n_grouped_cell; j++){
                for(z = 0; z < m->lstms[i-1]->n_grouped_cell; z++){
                    temp[j*m->lstms[i-1]->n_grouped_cell+z] = m->lstms[i-1]->bns[j]->outputs[z];
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
    int i,j,ret_count = m->layers,z;
    float** temp = (float**)malloc(sizeof(float*)*m->window);// inputs of lstms
    for(i = 0; i < m->window; i++){
        error2_model[i] = (float*)calloc(m->lstms[0]->size,sizeof(float));
        copy_array(error_model[i],error2_model[i],m->lstms[0]->size);
    }
    i = m->layers-1;
    while(i >= 0){
        int k = 1;
        for(k = 1; i >= 0; i--, k++){
            if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION)
                break;
            
        }
        if(i >= 0){
            if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
                for(j = 0; j < m->lstms[i]->window/m->lstms[i]->n_grouped_cell;j++){
                    for(z = 0; z < m->lstms[i]->n_grouped_cell; z++){
                        temp[j*m->lstms[i]->n_grouped_cell+z] = m->lstms[i]->bns[j]->outputs[z];
                    }
                }
                if(m->layers-i-k){
                    ret_count-=k;
                    ret2 = bp_rmodel_lstm(&hidden_states[ret_count],&cell_states[ret_count],temp,error2_model,m->window,m->lstms[0]->size,k,&m->lstms[i+1],input_error3);
                    for(j = 0; j < m->window; j++){
                        copy_array(input_error3[j],error2_model[j],m->lstms[0]->size);
                        free(input_error3[j]);
                    }
                    for(j = ret_count; j < ret_count+k; j++){
                        ret[j] = ret2[j-ret_count];
                        }
                    free(ret2);

                }
                for(j = 0; j < m->lstms[i]->window/m->lstms[i]->n_grouped_cell;j++){
                    batch_normalization_back_prop(m->lstms[i]->n_grouped_cell,&m->lstms[i]->out_up[j*m->lstms[i]->n_grouped_cell],m->lstms[i]->bns[j]->temp_vectors,m->lstms[i]->bns[j]->vector_dim,m->lstms[i]->bns[j]->gamma,m->lstms[i]->bns[j]->beta,m->lstms[i]->bns[j]->mean,m->lstms[i]->bns[j]->var,&error2_model[j*m->lstms[i]->n_grouped_cell],m->lstms[i]->bns[j]->d_gamma,m->lstms[i]->bns[j]->d_beta,m->lstms[i]->bns[j]->error2,m->lstms[i]->bns[j]->temp1,m->lstms[i]->bns[j]->temp2,m->lstms[i]->bns[j]->epsilon);
                }
                
                for(j = 0; j < m->window/m->lstms[i]->n_grouped_cell; j++){
                    for(z = 0; z < m->lstms[i]->n_grouped_cell; z++){
                        copy_array(m->lstms[i]->bns[j]->error2[z],error2_model[j*m->lstms[i]->n_grouped_cell+z],m->lstms[0]->size);
                    }
                }
            }
        }
        
        else{
            for(k = 0; k < m->layers; k++){
                if(m->lstms[k]->norm_flag == GROUP_NORMALIZATION){
                    k++;
                    break;
                }
            }
            for(j = 0; j < m->window; j++){
                temp[j] = input_model[j];
            }
            
            ret2 = bp_rmodel_lstm(hidden_states,cell_states,temp,error2_model,m->window,m->lstms[0]->size,k,&m->lstms[i],input_error);
            for(j = 0; j < k; j++){
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


