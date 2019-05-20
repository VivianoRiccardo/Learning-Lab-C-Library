#include "llab.h"

/* This function builds a rmodel* structure which can be used to train the network
 * 
 * Input:
 *             
 *             @ int layers:= number of total layers
 *             @ int n_lstm:= same as layer, but only for long short term memory layers
 *             @ lstm** lstms:= your long short term memory layers
 *                @ int window:= is the number of unrolled orizontal recurrent cells are provided
 *                @ int hidden_state_mode:= cna be stateful and stateless (flag STATEFUL, flag STATELESS)
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
 *                @ float tau:= the tau param
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
 *             @ float** input_model:= the seq-to-seq inputs dimensions: (window+1) x size 
 *             @ rmodel* m:= the rmodel that is gonna compute the feed forward
 * 
 * 
 * */
void ff_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, rmodel* m){    
    
    //allocation of the resources

    float* dropout_output = (float*)malloc(sizeof(float)*m->lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*m->lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    
    int i,j;
    
    float temp_drp_value;
    /*feed_forward_passage*/
    
    for(i = 0; i < m->window; i++){
        for(j = 0; j < m->layers; j++){
            
            if(j == 0){ //j = 0 means that we are at the first lstm_cell in vertical
                if(i == 0){//i = 0 means we are at the first lstm in orizontal
                
                    //in this case the h-1 and c-1 come from the last mini_batch
                    if(m->lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->dropout_threshold_right);
                   
                    if(m->lstms[j]->dropout_flag_right != DROPOUT){
                        temp_drp_value = m->lstms[j]->dropout_threshold_right;
                        m->lstms[j]->dropout_threshold_right = 0;
                    }
                    get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    if(m->lstms[j]->dropout_flag_right != DROPOUT)
                        m->lstms[j]->dropout_threshold_right = temp_drp_value;
                    
                    
                    if(m->lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,m->lstms[j]->dropout_threshold_right,dropout_output2,m->lstms[j]->size);
                    
                    lstm_ff(input_model[i], dropout_output2, cell_states[j], m->lstms[j]->lstm_cell[i], m->lstms[j]->lstm_hidden[i], m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->biases, m->lstms[j]->lstm_z[i], m->lstms[j]->size);
                }

                else{
                    if(m->lstms[j]->dropout_flag_right != DROPOUT){
                        temp_drp_value = m->lstms[j]->dropout_threshold_right;
                        m->lstms[j]->dropout_threshold_right = 0;
                    }
                    
                    get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(m->lstms[j]->dropout_flag_right != DROPOUT)
                        m->lstms[j]->dropout_threshold_right = temp_drp_value;
                    
                    
                    if(m->lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,m->lstms[j]->dropout_threshold_right,dropout_output2,m->lstms[j]->size);
                        
                    lstm_ff(input_model[i], dropout_output2, m->lstms[j]->lstm_cell[i-1], m->lstms[j]->lstm_cell[i], m->lstms[j]->lstm_hidden[i], m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->biases, m->lstms[j]->lstm_z[i], m->lstms[j]->size);
                }
            }
            
            else{
                
                if(i == 0){//i = 0 and j != 0 means that we are at the first lstm in orizontal but not in vertical
                    if(m->lstms[j]->dropout_flag_right == DROPOUT)
                        set_dropout_mask(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->dropout_threshold_right);
                    if(m->lstms[j]->dropout_flag_right != DROPOUT){
                        temp_drp_value = m->lstms[j]->dropout_threshold_right;
                        m->lstms[j]->dropout_threshold_right = 0;
                    }
                    
                    get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);//dropout for h between recurrent connections
                    
                    if(m->lstms[j]->dropout_flag_right != DROPOUT)
                        m->lstms[j]->dropout_threshold_right = temp_drp_value;
                    
                    
                    if(m->lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,m->lstms[j]->dropout_threshold_right,dropout_output2,m->lstms[j]->size);
                        
                    lstm_ff(dropout_output, dropout_output2, cell_states[j], m->lstms[j]->lstm_cell[i], m->lstms[j]->lstm_hidden[i], m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->biases, m->lstms[j]->lstm_z[i], m->lstms[j]->size);
                    
                }    
                else{
                    if(m->lstms[j]->dropout_flag_right != DROPOUT){
                        temp_drp_value = m->lstms[j]->dropout_threshold_right;
                        m->lstms[j]->dropout_threshold_right = 0;
                    }
                    
                    get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->lstm_hidden[i-1],dropout_output2);//dropout for h between recurrent connections
                    
                    if(m->lstms[j]->dropout_flag_right != DROPOUT)
                        m->lstms[j]->dropout_threshold_right = temp_drp_value;
                    
                    
                    if(m->lstms[j]->dropout_flag_right == DROPOUT_TEST)
                        mul_value(dropout_output2,m->lstms[j]->dropout_threshold_right,dropout_output2,m->lstms[j]->size);
                        
                    lstm_ff(dropout_output, dropout_output2, cell_states[j], m->lstms[j]->lstm_cell[i], m->lstms[j]->lstm_hidden[i], m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->biases, m->lstms[j]->lstm_z[i], m->lstms[j]->size);
                }
            }
            
            /* the dropout is applied to each lstm_hidden to feed the deeper lstm cell in vertical, as input*/
            if(i == 0)
                set_dropout_mask(m->lstms[j]->size,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_threshold_up);
            
            if(m->lstms[j]->dropout_flag_right != DROPOUT){
                temp_drp_value = m->lstms[j]->dropout_threshold_up;
                m->lstms[j]->dropout_threshold_up = 0;
            }
            
            get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_up,m->lstms[j]->lstm_hidden[i],dropout_output);
            
            if(m->lstms[j]->dropout_flag_right != DROPOUT)
                m->lstms[j]->dropout_threshold_up = temp_drp_value;
            
            
            if(m->lstms[j]->dropout_flag_right == DROPOUT_TEST)
                mul_value(dropout_output,m->lstms[j]->dropout_threshold_up,dropout_output,m->lstms[j]->size);
    
                
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
 *             @ rmodel* m:= the recurrent model
 *             @ float** input_error:= the error of the inputs of this model, dimensions: m->window*m->size, must be initialized only with m->window
 * 
 * */
float*** bp_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m, float** input_error){

   /* backpropagation passage*/

    int i,j, lstm_bp_flag;
    
    float* dropout_output = (float*)malloc(sizeof(float)*m->lstms[0]->size); //here we store the modified output by dropout coming from an lstm cell
    float* dropout_output2 = (float*)malloc(sizeof(float)*m->lstms[0]->size);
    float* dx; //here we store the modified output by dropout coming from the last lstm cell
    float*** matrix = (float***)malloc(sizeof(float**)*m->layers);
    float** temp;
    
    for(i = 0; i < m->layers; i++){
        matrix[i] = NULL;
    }
    /*with i = 0 we should handle the h minus and c minus that are given by the params passed to the function*/
    for(i = m->window-1; i > 0; i--){
        for(j = m->layers-1; j >= 0; j--){
            
            dx = (float*)calloc(m->lstms[0]->size,sizeof(float));
            
            if(j == m->layers-1 && i == m->window-1)
                lstm_bp_flag = 0;
            else if(j != m->layers-1 && i == m->window-1)
                lstm_bp_flag = 1;
                
            else if(j == m->layers-1 && i != m->window-1)
                lstm_bp_flag = 2;
                
            else
                lstm_bp_flag = 3;
            
            
            if(j == m->layers-1)
                
                get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_up,error_model[i],dx);

            if(j == m->layers-1){
                
                if(j != 0)
                    get_dropout_array(m->lstms[j]->size,m->lstms[j-1]->dropout_mask_up,m->lstms[j-1]->lstm_hidden[i],dropout_output);
                get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->lstm_hidden[i-1],dropout_output2);
                
                
                if(j != 0){
                    if(i == m->window-1)
                        temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u, m->lstms[j]->d_biases, m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->lstm_z[i], dx,dropout_output,m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                    else
                        temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size,  m->lstms[j]->d_w,m->lstms[j]->d_u, m->lstms[j]->d_biases, m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->lstm_z[i], dx,dropout_output,m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], NULL, NULL, m->lstms[j]->lstm_z[i+1],matrix[j],NULL,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                }
                
                else{
                    if(i == m->window-1)
                        temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u, m->lstms[j]->d_biases, m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->lstm_z[i], dx,input_model[i],m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], NULL, NULL, NULL, NULL,NULL,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                    else
                        temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size,  m->lstms[j]->d_w,m->lstms[j]->d_u, m->lstms[j]->d_biases, m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->lstm_z[i], dx,input_model[i],m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], NULL, NULL, m->lstms[j]->lstm_z[i+1],matrix[j],NULL,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                    
                }
                
                
                if(matrix[j] != NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;

            }
            
            else if(j != m->layers-1 && j != 0){
                
                
                get_dropout_array(m->lstms[j]->size,m->lstms[j-1]->dropout_mask_up,m->lstms[j-1]->lstm_hidden[i],dropout_output);
                get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->lstm_hidden[i-1],dropout_output2);

                
                if(i == m->window-1)
                    temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u,m->lstms[j]->d_biases,m->lstms[j]->w,m->lstms[j]->u,m->lstms[j]->lstm_z[i], dx, dropout_output,m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], m->lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,m->lstms[j+1]->w,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                    
                else
                    temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u,m->lstms[j]->d_biases,m->lstms[j]->w,m->lstms[j]->u,m->lstms[j]->lstm_z[i], dx, dropout_output,m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], m->lstms[j+1]->lstm_z[i], matrix[j+1], m->lstms[j]->lstm_z[i+1],matrix[j],m->lstms[j+1]->w,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
                
            }
            
            else{
                
                get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,m->lstms[j]->lstm_hidden[i-1],dropout_output2);

                if(i == m->window-1)
                    temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u,m->lstms[j]->d_biases,m->lstms[j]->w,m->lstms[j]->u,m->lstms[j]->lstm_z[i], dx, input_model[i],m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], m->lstms[j+1]->lstm_z[i], matrix[j+1], NULL,NULL,m->lstms[j+1]->w,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                
                else
                    temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u,m->lstms[j]->d_biases,m->lstms[j]->w,m->lstms[j]->u,m->lstms[j]->lstm_z[i], dx, input_model[i],m->lstms[j]->lstm_cell[i],dropout_output2,m->lstms[j]->lstm_cell[i-1], m->lstms[j+1]->lstm_z[i], matrix[j+1], m->lstms[j+1]->lstm_z[i],matrix[j],m->lstms[j+1]->w,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
                
                if(matrix[j]!= NULL)
                    free_matrix(matrix[j],4);
                
                matrix[j] = temp;
                
            }
            

            free(dx);
        }
        
    }
    
    i = 0;
    /* computing back propagation just for the first lstm layers with hidden states defined by the previous batch*/
    for(j = m->layers-1; j >= 0; j--){
            
        dx = (float*)calloc(m->lstms[0]->size,sizeof(float));
        
        if(j == m->layers-1 && i == m->window-1)
            lstm_bp_flag = 0;
        else if(j != m->layers-1 && i == m->window-1)
            lstm_bp_flag = 1;
            
        else if(j == m->layers-1 && i != m->window-1)
            lstm_bp_flag = 2;
            
        else
            lstm_bp_flag = 3;
        
        
        if(j == m->layers-1)
            
            get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_up,error_model[i],dx);

        if(j == m->layers-1){
            
            if(j != 0)
                get_dropout_array(m->lstms[j]->size,m->lstms[j-1]->dropout_mask_up,m->lstms[j-1]->lstm_hidden[i],dropout_output);
            get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            if(j != 0)
                temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u, m->lstms[j]->d_biases, m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->lstm_z[i], dx,dropout_output,m->lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, m->lstms[j]->lstm_z[i+1],matrix[j],NULL,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
            else
                temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u, m->lstms[j]->d_biases, m->lstms[j]->w, m->lstms[j]->u, m->lstms[j]->lstm_z[i], dx,input_model[i],m->lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], NULL, NULL, m->lstms[j]->lstm_z[i+1],matrix[j],NULL,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
            
            
            matrix[j] = temp;
            
        }
        
        else if(j != m->layers-1 && j != 0){
            
            get_dropout_array(m->lstms[j]->size,m->lstms[j-1]->dropout_mask_up,m->lstms[j-1]->lstm_hidden[i],dropout_output);
            get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            

            temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u,m->lstms[j]->d_biases,m->lstms[j]->w,m->lstms[j]->u,m->lstms[j]->lstm_z[i], dx, dropout_output,m->lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], m->lstms[j+1]->lstm_z[i], matrix[j+1], m->lstms[j]->lstm_z[i+1],matrix[j],m->lstms[j+1]->w,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
            
            matrix[j] = temp;
        }
        
        else{
            
            get_dropout_array(m->lstms[j]->size,m->lstms[j]->dropout_mask_right,hidden_states[j],dropout_output2);
            
            temp = lstm_bp(lstm_bp_flag,m->lstms[j]->size, m->lstms[j]->d_w,m->lstms[j]->d_u,m->lstms[j]->d_biases,m->lstms[j]->w,m->lstms[j]->d_u,m->lstms[j]->lstm_z[i], dx, input_model[i],m->lstms[j]->lstm_cell[i],dropout_output2,cell_states[j], m->lstms[j+1]->lstm_z[i], matrix[j+1], m->lstms[j]->lstm_z[i+1],matrix[j],m->lstms[j+1]->w,m->lstms[j]->dropout_mask_up,m->lstms[j]->dropout_mask_right);
            
            matrix[j] = temp;
            
        }
        free(dx);
    }
    
    free(dropout_output);
    free(dropout_output2);
    
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
 * 
 * */
void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda){
    if(m == NULL)
        return;
    
    lambda*=mini_batch_size;
    
    if(regularization == L2_REGULARIZATION)
        add_l2_lstm_layer(m,total_number_weights,lambda);
    
    
    
    if(gradient_descent_flag == NESTEROV)
        update_lstm_layer_nesterov(m,lr,momentum,mini_batch_size);
    
    
    else if(gradient_descent_flag == ADAM){
        update_lstm_layer_adam(m,lr,mini_batch_size, (*b1), (*b2));
        (*b1)*=BETA1_ADAM;
        (*b2)*=BETA2_ADAM;
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

