#include "llab.h"

/* This function builds a model* structure which can be used to train the network
 * 
 * Input:
 *             
 *             @ int layers:= number of total layers, this means that if you have 2 layers with the same layer id 
 *                            then layers = 2. For example if you have 2 fully-connected layers with same layer id = 0
 *                            then layers param must be set to 2. if you have 3 layers, 2 with same layer id and 1 with another
 *                            layer id, then layers = 3
 *             @ int n_rl:= same as layers but only for residual layers
 *             @ int n_cl:= same as layer but only for convolutional layers. (the convolutional layers inside residual layer must not be count)
 *             @ int n_fcl:= same as layer, but only for fully-connected layers
 *             @ rl** rls:= your residual layers
 *             @ cl** cls:= your convolutional layers
 *             @ fcl** fcls:= your fully-connected layers
 * 
 * */
model* network(int layers, int n_rl, int n_cl, int n_fcl, rl** rls, cl** cls, fcl** fcls){
    if(!layers || (!n_rl && !n_cl && !n_fcl) || (!n_rl && rls != NULL) || (!n_cl && cls!= NULL) || (!n_fcl && fcls != NULL)){
        printf("Error: layers must be > 0 and at least one between n_rl, n_cl, n_fcl must be > 0\n");
        exit(1); 
    }
    
    int i,j,k, position, count, k1,k2,k3;
    
    
    /*checking if the residual layer has the right size from the input to the output*/
    for(i = 0; i < n_rl; i++){
        if(rls[i]->cls[rls[i]->n_cl-1]->post_pooling){
            if(rls[i]->cls[rls[i]->n_cl-1]->n_kernels*rls[i]->cls[rls[i]->n_cl-1]->rows2*rls[i]->cls[rls[i]->n_cl-1]->cols2 != rls[i]->channels*rls[i]->input_rows*rls[i]->input_cols){
                printf("Error: you have a residual layer where the input size doesn't correspond to the last convolutional layer size of the residual layer\n");
                exit(1);
            }
        }
        
        else{
            if(rls[i]->cls[rls[i]->n_cl-1]->n_kernels*rls[i]->cls[rls[i]->n_cl-1]->rows1*rls[i]->cls[rls[i]->n_cl-1]->cols1 != rls[i]->channels*rls[i]->input_rows*rls[i]->input_cols){
                printf("Error: you have a residual layer where the input size doesn't correspond to the last convolutional layer size of the residual layer\n");
                exit(1);
            }
        }
    }
    
    cl* temp = NULL;
    fcl* temp2 = NULL;
    rl* temp3 = NULL;
    int** sla = (int**)malloc(sizeof(int*)*layers);
    for(i = 0; i < layers; i++){
        sla[i] = (int*)calloc(layers,sizeof(int));
    }
    
    model* m = (model*)malloc(sizeof(model));
    
    /* sorting conv layers inside residual layers*/
       
    for(i = 0; i <  n_rl; i++){
        for(count = 0; count < rls[i]->n_cl; count++){
            j = 0;
            temp = rls[i]->cls[j];
            position = j;
            
            for(k = 1; k < rls[i]->n_cl; k++){
                if(rls[i]->cls[position]->layer > rls[i]->cls[k]->layer){
                    rls[i]->cls[position] = rls[i]->cls[k];
                    rls[i]->cls[k] = temp;
                    position = k;
                }
            }
        }
        /* checking if the convolutional layers of residual layers are sequential*/
        for(count = 1; count < rls[i]->n_cl; count++){
            if(rls[i]->cls[count]->layer - rls[i]->cls[count-1]->layer >= 2){
                printf("Error: you have a residual layer with no sequential sub-convolutional-layers\n");
                exit(1);
            }
        }
    }
    
    /* sorting residual layers*/
    for(i = 0; i < n_rl; i++){
        j = 0;
        temp3 = rls[j];
        position = j;
        
        for(k = 1; k < n_rl; k++){
            if(rls[position]->cls[0]->layer > rls[k]->cls[0]->layer){
                rls[position] = rls[k];
                rls[k] = temp3;
                position = k;
            }
        }
    }
    
    /* sorting conv layers*/
    for(i = 0; i < n_cl; i++){
        j = 0;
        temp = cls[j];
        position = j;
        
        for(k = 1; k < n_cl; k++){        
            if(cls[position]->layer > cls[k]->layer){
                cls[position] = cls[k];
                cls[k] = temp;
                position = k;
            }
        }
    }
    
    /* sorting fully-connected layers*/
    for(i = 0; i < n_fcl; i++){
        j = 0;
        temp2 = fcls[j];
        position = j;
        
        for(k = 1; k < n_fcl; k++){
            if(fcls[position]->layer > fcls[k]->layer){
                fcls[position] = fcls[k];
                fcls[k] = temp2;
                position = k;
            }
        }
    }
    
    /* checking if the layers are sequential or not*/
    position = 0;
    for(i = 0; i < layers; i++){
        /* building sla matrix and gls*/
        k = 0;
        for(j = 0; j < n_rl; j++){
            for(count = 0; count < rls[j]->n_cl; count++){
                if(rls[j]->cls[count]->layer == i){
                    sla[i][k] = RLS; 
                    k++;
                }
            }
        }
        
        for(j = 0; j < n_cl; j++){
            if(cls[j]->layer == i){
                sla[i][k] = CLS;
                k++;
            }
        }
        
        for(j = 0; j < n_fcl; j++){
            if(fcls[j]->layer == i){
                sla[i][k] = FCLS;
                k++;
            }
        }
        
        position += k;
        if(!k && position != layers){
            printf("Error: your layers are not sequential, missing the layer with index: %d\n",i);
            exit(1);
        }
    }
    /*There is no check if the sizes match or not, this happen during the feed forward*/
    
    m->layers = layers;
    m->n_rl = n_rl;
    m->n_cl = n_cl;
    m->n_fcl = n_fcl;
    m->sla = sla;
    m->rls = rls;
    m->cls = cls;
    m->fcls = fcls;
        
    return m;
}

/* This function frees the space allocated by a model structure
 * 
 * Input:
 *             @ model* m:= the structure
 * 
 * */
void free_model(model* m){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_rl; i++){
        free_residual(m->rls[i]);
    }
    free(m->rls);
    for(i = 0; i < m->n_cl; i++){
        free_convolutional(m->cls[i]);
    }
    free(m->cls);
    for(i = 0; i < m->n_fcl; i++){
        free_fully_connected(m->fcls[i]);
    }
    free(m->fcls);
    for(i = 0; i < m->layers; i++){
        free(m->sla[i]);
    }
    free(m->sla);
    free(m);
}


/* This function copies a model using the copy function for the layers
 * see layers.c file
 * 
 * Input:
 *         
 *             @ model* m:= the model that must be copied
 * 
 * */
model* copy_model(model* m){
    if(m == NULL)
        return NULL;
    int i;
    
    fcl** fcls = NULL;
    if(m->fcls!=NULL)
        fcls = (fcl**)malloc(sizeof(fcl*)*m->n_fcl);
    cl** cls = NULL;
    if(m->cls!=NULL)
        cls = (cl**)malloc(sizeof(cl*)*m->n_cl);
        
    rl** rls = NULL;
    if(m->rls!=NULL)
        rls = (rl**)malloc(sizeof(rl*)*m->n_rl);
    for(i = 0; i < m->n_fcl; i++){
        fcls[i] = copy_fcl(m->fcls[i]);
    }
    for(i = 0; i < m->n_cl; i++){
        cls[i] = copy_cl(m->cls[i]);
    }
    for(i = 0; i < m->n_rl; i++){
        rls[i] = copy_rl(m->rls[i]);
    }
    model* copy = network(m->layers, m->n_rl, m->n_cl, m->n_fcl, rls, cls, fcls);
    return copy;
}

/* This function resets a model using the copy model function
 * */
void reset_model(model** m){
    if(m == NULL){
        printf("Error: you passed a NULL pointer in reset_model\n");
        exit(1);
    }
    model* copy = copy_model((*m));
    free_model((*m));
    (*m) = copy;
}




/* This function saves a model(network) on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ model* m:= the actual network that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_model(model* m, int n){
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
        printf("Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&m->layers,sizeof(int),1,fw);
    
    if(i != 1){
        printf("Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fwrite(&m->n_rl,sizeof(int),1,fw);
    
    if(i != 1){
        printf("Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fwrite(&m->n_cl,sizeof(int),1,fw);
    
    if(i != 1){
        printf("Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fwrite(&m->n_fcl,sizeof(int),1,fw);
    
    if(i != 1){
        printf("Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fclose(fw);
    if(i!=0){
        printf("Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    for(i = 0; i < m->n_rl; i++){
        save_rl(m->rls[i],n);
    }
    
    for(i = 0; i < m->n_cl; i++){
        save_cl(m->cls[i],n);
    }
    
    for(i = 0; i < m->n_fcl; i++){
        save_fcl(m->fcls[i],n);
    }
    
    free(s);
}

/* This function loads a network model from a .bin file with name file
 * 
 * Input:
 * 
 *             @ char* file:= the binary file from which the model will be loaded
 * 
 * */
model* load_model(char* file){
    if(file == NULL)
        return NULL;
    int i;
    FILE* fr = fopen(file,"r");
    
    if(fr == NULL){
        printf("Error: error during the opening of the file %s\n",file);
        exit(1);
    }
    
    int layers = 0,n_cl = 0,n_rl = 0,n_fcl = 0;
    
    i = fread(&layers,sizeof(int),1,fr);
    if(i != 1){
        printf("Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_rl,sizeof(int),1,fr);
    
    if(i != 1){
        printf("Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_cl,sizeof(int),1,fr);
    
    if(i != 1){
        printf("Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_fcl,sizeof(int),1,fr);
    
    if(i != 1){
        printf("Error: an error occurred loading the model\n");
        exit(1);
    }
    
    rl** rls = (rl**)malloc(sizeof(rl*)*n_rl);
    cl** cls = (cl**)malloc(sizeof(cl*)*n_cl);
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*n_fcl);
    
    for(i = 0; i < n_rl; i++){
        rls[i] = load_rl(fr);
    }
    
    for(i = 0; i < n_cl; i++){
        cls[i] = load_cl(fr);
    }
    
    for(i = 0; i < n_fcl; i++){
        fcls[i] = load_fcl(fr);
    }
    
    i = fclose(fr);
    if(i!=0){
        printf("Error: an error occurred closing the file %s\n",file);
        exit(1);
    }
    
    model* m = network(layers,n_rl,n_cl,n_fcl,rls,cls,fcls);
    
    return m;
    
}

/* This function compute the feed forward between 2 fully-connected layer
 * 
 * Input:
 *             @ fcl* f1:= the input fully-connected layer
 *             @ fcl* f2:= the output fully-connected layer
 * 
 * Warning:
 *             The dropout between 2 layers is not applied immediately but only
 *             during the next layer.
 *             For example:
 *                 f1 at the end of activation must apply dropout, this means that
 *                 the f1 mask is already applied, but f1->post_activation doesn't have
 *                 the dropout, so in this case during this feed forward we must apply
 *                 the dropout to the f1->post_activation array.
 *                 In the same way, if we must apply the dropout to f2 then in this function
 *                 we set the dropout_mask of f2, but f2->post_activation doesn't have the dropout
 * 
 * */
void ff_fcl_fcl(fcl* f1, fcl* f2){
    
    if(f1->output != f2->input){
        printf("Error: the sizes between 2 fully-connected layers don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    float* temp = NULL;
    int i;
    
    /* computing the pre-activation array for f2 from f1*/
    
    /* no activation for f1*/
    if(f1->activation_flag == NO_ACTIVATION){
        if(f1->dropout_flag == NO_DROPOUT){
            fully_connected_feed_forward(f1->pre_activation, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                temp = (float*)malloc(sizeof(float)*f2->input);
                get_dropout_array(f2->input,f1->dropout_mask,f1->pre_activation,temp);
                fully_connected_feed_forward(temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                temp = (float*)malloc(sizeof(float)*f2->input);
                mul_value(f2->pre_activation,f1->dropout_threshold,temp,f2->input);
                fully_connected_feed_forward(temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
            }
        }
    }
    
    /* activation for f1*/
    else{
        if(f1->dropout_flag == NO_DROPOUT){
            fully_connected_feed_forward(f1->post_activation, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                temp = (float*)malloc(sizeof(float)*f2->input);
                get_dropout_array(f2->input,f1->dropout_mask,f1->post_activation,temp);
                fully_connected_feed_forward(temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                temp = (float*)malloc(sizeof(float)*f2->input);
                mul_value(f2->post_activation,f1->dropout_threshold,temp,f2->input);
                fully_connected_feed_forward(temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
            }
        }
    }
    
    /* computing the activation for f2 (if the activation_flag is > 0)*/
    if(f2->activation_flag == SIGMOID)
        sigmoid_array(f2->pre_activation,f2->post_activation,f2->output);
    else if(f2->activation_flag == RELU)
        relu_array(f2->pre_activation,f2->post_activation,f2->output);
    else if(f2->activation_flag == SOFTMAX)
        softmax(f2->pre_activation,f2->post_activation,f2->output);
    else if(f2->activation_flag == TANH)
        tanhh_array(f2->pre_activation,f2->post_activation,f2->output);
    
    /* setting the dropout mask, if dropout flag is != 0*/
    if(f2->dropout_flag)
        set_dropout_mask(f2->output, f2->dropout_mask, f2->dropout_threshold);
    
    free(temp);
}


/* This function compute the feed forward between a fully-connected input layer
 * and a convolutional output layer
 * 
 * Input:
 *             @ fcl* f1:= the input fully-connected layer
 *             @ cl* f2:= the output convolutional layer
 * Warning:
 *             The dropout between 2 layers is not applied immediately but only
 *             during the next layer.
 *             For example:
 *                 f1 at the end of activation must apply dropout, this means that
 *                 the f1 mask is already applied, but f1->post_activation doesn't have
 *                 the dropout, so in this case during this feed forward we must apply
 *                 the dropout to the f1->post_activation array.
 *  
 * */
void ff_fcl_cl(fcl* f1, cl* f2){
    if(f1->output != f2->channels*f2->input_rows*f2->input_cols){
        printf("Error: the sizes between an input fully-connected layer and an output convolutional layer don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }

    float* temp = NULL;
    int i,j,k,z;
    
    /* f2 pre activation with no activation for f1*/
     if(f1->activation_flag == NO_ACTIVATION){
        if(f1->dropout_flag == NO_DROPOUT){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_feed_forward(f1->pre_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
            }
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                temp = (float*)malloc(sizeof(float)*f2->channels*f2->input_rows*f2->input_cols);
                get_dropout_array(f1->output,f1->dropout_mask,f1->pre_activation,temp);
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_feed_forward(temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                }
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                temp = (float*)malloc(sizeof(float)*f2->channels*f2->input_rows*f2->input_cols);
                mul_value(f1->pre_activation,f1->dropout_threshold,temp,f1->output);
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_feed_forward(temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                }
            }
        }
    }
    
    /* f2 pre activation with activation for f1*/
    else{
        if(f1->dropout_flag == NO_DROPOUT){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_feed_forward(f1->post_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
            }
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                temp = (float*)malloc(sizeof(float)*f2->channels*f2->input_rows*f2->input_cols);
                get_dropout_array(f1->output,f1->dropout_mask,f1->post_activation,temp);
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_feed_forward(temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                }
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                temp = (float*)malloc(sizeof(float)*f2->channels*f2->input_rows*f2->input_cols);
                mul_value(f1->post_activation,f1->dropout_threshold,temp,f1->output);
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_feed_forward(temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                }
            }
        }
    }
    
    /* activation for f2, if there is any activation*/
    if(f2->activation_flag == SIGMOID){
        if(f2->padding1_rows){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    sigmoid_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                }
            }
        }
        
        else
            sigmoid_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

    }
        
    
    else if(f2->activation_flag == RELU){
        if(f2->padding1_rows){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    relu_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                }
            }
        }
        
        else
            relu_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

    }
    else if(f2->activation_flag == TANH){
        if(f2->padding1_rows){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    tanhh_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                }
            }
        }
        
        else
            tanhh_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

    }
    /* normalization for f2, if there is any normalization*/
    if(f2->normalization_flag){
        for(i = 0; i < f2->n_kernels; i++){
            for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                for(k = f2->padding1_rows; k < f2->cols1-f2->padding1_rows;k++){
                    if(f2->activation_flag != NO_ACTIVATION)
                        local_response_normalization_feed_forward(f2->post_activation,f2->post_normalization, i,j,k, f2->n_kernels, f2->rows1, f2->cols1,N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                    else
                        local_response_normalization_feed_forward(f2->pre_activation,f2->post_normalization, i,j,k, f2->n_kernels, f2->rows1, f2->cols1,N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                }
            }
        }    
    }
    
    /* pooling for f2, if there is any pooling*/
    if(f2->pooling_flag != NO_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            if(f2->normalization_flag != NO_NORMALIZATION){
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->post_normalization[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->post_normalization[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
            
            else if(f2->activation_flag != NO_ACTIVATION){
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->post_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->post_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
            
            else{
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->pre_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->pre_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
        }
    }
    
    free(temp);
    
}


/* This function compute the feed forward between a convolutional input layer
 * and a fully-connected output layer
 * 
 * Input:
 *             @ cl* f1:= the input convolutional layer
 *             @ fcl* f2:= the output fully-connected layer
 * Warning:
 *             The dropout between 2 layers is not applied immediately but only
 *             during the next layer.
 *             For example:
 *                 if we must apply the dropout to f2 then in this function
 *                 we set the dropout_mask of f2, but f2->post_activation doesn't have the dropout
 * 
 * */
void ff_cl_fcl(cl* f1, fcl* f2){
    if(f1->pooling_flag && f1->n_kernels*f1->rows2*f1->cols2 != f2->input){
        printf("Error: the sizes between an input convolutional layer and an output fully-connected layer don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    else if(!f1->pooling_flag && f1->n_kernels*f1->rows1*f1->cols1 != f2->input){
        printf("Error: the sizes between an input convolutional layer and an output fully-connected layer don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    int i;
    
    /* computing the pre-activation array for f2 from f1*/
    
    /* pooling for f1*/
    if(f1->pooling_flag)
        fully_connected_feed_forward(f1->post_pooling, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
    
    /* no pooling for f1, but normalization*/
    else if(f1->normalization_flag)
        fully_connected_feed_forward(f1->post_normalization, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
    
    /* no pooling, no normalization for f1, but activation*/
    else if(f1->activation_flag)
        fully_connected_feed_forward(f1->post_activation, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
    
    /* no pooling, no normalization, no activation for f1*/
    else
        fully_connected_feed_forward(f1->pre_activation, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
    
    /* computing the activation for f2 (if the activation_flag is > 0)*/
    if(f2->activation_flag == SIGMOID)
        sigmoid_array(f2->pre_activation,f2->post_activation,f2->output);
    else if(f2->activation_flag == RELU)
        relu_array(f2->pre_activation,f2->post_activation,f2->output);
    else if(f2->activation_flag == SOFTMAX)
        softmax(f2->pre_activation,f2->post_activation,f2->output);
    else if(f2->activation_flag == TANH)
        tanhh_array(f2->pre_activation,f2->post_activation,f2->output);
    

    
    /* setting the dropout mask, if dropout flag is != 0*/
    if(f2->dropout_flag)
        set_dropout_mask(f2->output, f2->dropout_mask, f2->dropout_threshold);
    
    
}


/* This function compute the feed forward between 2 convolutional layers
 * 
 * Input:
 *             @ cl* f1:= the input convolutional layer
 *             @ cl* f2:= the output convolutional layer
 * 
 * */
void ff_cl_cl(cl* f1, cl* f2){
    if(f1->pooling_flag && f1->n_kernels*f1->rows2*f1->cols2 != f2->channels*f2->input_rows*f2->input_cols){
        printf("Error: the sizes between 2 convolutional layers don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    else if(!f1->pooling_flag && f1->n_kernels*f1->rows1*f1->cols1 != f2->channels*f2->input_rows*f2->input_cols){
        printf("Error: the sizes between 2 convolutional layers don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }

    int i,j,k,z;
    
    /* pooling for f1*/
    if(f1->pooling_flag){
        for(i = 0; i < f2->n_kernels; i++){
            convolutional_feed_forward(f1->post_pooling, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
        }    
    }
            
    /* no pooling for f1, but normalization*/
    else if(f1->normalization_flag){
        for(i = 0; i < f2->n_kernels; i++){
            convolutional_feed_forward(f1->post_normalization, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
        }    
    }
    /* no pooling, no normalization for f1, but activation*/
    else if(f1->activation_flag){
        for(i = 0; i < f2->n_kernels; i++){
            convolutional_feed_forward(f1->post_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
        }    
    }
    /* no pooling, no normalization, no activation for f1*/
    else{
        for(i = 0; i < f2->n_kernels; i++){
            convolutional_feed_forward(f1->pre_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
        }    
    }
    
    /* activation for f2, if there is any activation*/
    if(f2->activation_flag == SIGMOID){
        if(f2->padding1_rows){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    sigmoid_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                }
            }
        }
        
        else
            sigmoid_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

    }
        
    
    else if(f2->activation_flag == RELU){
        if(f2->padding1_rows){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    relu_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                }
            }
        }
        
        else
            relu_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

    }
    else if(f2->activation_flag == TANH){
        if(f2->padding1_rows){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    tanhh_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                }
            }
        }
        
        else
            tanhh_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

    }
    /* normalization for f2, if there is any normalization*/
    if(f2->normalization_flag){
        for(i = 0; i < f2->n_kernels; i++){
            for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                for(k = f2->padding1_rows; k < f2->cols1-f2->padding1_rows;k++){
                    if(f2->activation_flag)
                        local_response_normalization_feed_forward(f2->post_activation,f2->post_normalization, i,j,k, f2->n_kernels, f2->rows1, f2->cols1,N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                    else
                        local_response_normalization_feed_forward(f2->pre_activation,f2->post_normalization, i,j,k, f2->n_kernels, f2->rows1, f2->cols1,N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                }
            }
        }    
    }
    
    /* pooling for f2, if there is any pooling*/
    if(f2->pooling_flag){
        for(i = 0; i < f2->n_kernels; i++){
            if(f2->normalization_flag){
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->post_normalization[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->post_normalization[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
            
            else if(f2->activation_flag){
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->post_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->post_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
            
            else{
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->pre_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->pre_activation[i*f2->rows1*f2->cols1], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
        }
    }
    
    
}

/* This function computes the derivative of weights and biases of a fully-connected layer f2
 * applied to a previous fully connected layer f1, and returns the error of the last function of the
 * previous layer. For example:
 * if the previous layer applied only the pre_activation then the float* vector returned is the DL/df1->pre_activation
 * if the previous layer applied only pre_activation and post activation then the float* vector returned is DL/df1->post_activation
 * if the previous layer applied pre activation, post activation and dropout then the float* vector returned is
 * DL/df1->post_activation without the dropout applied, in this case the dropout must be applied during the backpropagation of f1.
 * If f2 applied the dropout, then the float* error passed as param could be DL/df2->pre_activation or DL/df2->post_activation
 * in both the cases we must apply the dropout_mask to this error.
 * 
 * Input:
 * 
 *             @ fcl* f1:= the fully-connected input layer
 *             @ fcl* f2:= the fully-connected current layer
 *             @ float* error:= the error passed
 * 
 * Warning:
 *             if we have softmax as activation function of f2, (softmax can be applied only for the last fully-connected layers)
 *             then the error passed as param is not DL/Df2->post_activation but is L where L is the error
 * */
float* bp_fcl_fcl(fcl* f1, fcl* f2, float* error){
    int i;
    float* temp = (float*)calloc(f2->output,sizeof(float));
    float* temp3 = (float*)calloc(f2->output,sizeof(float));
    float* temp2 = (float*)calloc(f2->input,sizeof(float));
    float* error2 = (float*)calloc(f2->input,sizeof(float));
    
    /*computing the backpropagation for f2*/
    if(f2->dropout_flag){
        dot1D(error,f2->dropout_mask,temp,f2->output);
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_cross_entropy_reduced_form_with_softmax_array(f2->post_activation,  error,temp3, f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
    }
    
    else{
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_cross_entropy_reduced_form_with_softmax_array(f2->post_activation,  error,temp3, f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        
        else{
            copy_array(error,temp,f2->output);
        }
    }
    /* computing the weight and bias derivatives for f2 applied to f1 output*/
    if(f1->dropout_flag){
        if(f1->activation_flag){
            dot1D(f1->post_activation,f1->dropout_mask,temp2,f2->input);
            fully_connected_back_prop(temp2, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
        }
        
        else{
            dot1D(f1->pre_activation,f1->dropout_mask,temp2,f2->input);
            fully_connected_back_prop(temp2, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
        }
    }
    
    else{
        if(f1->activation_flag){
            fully_connected_back_prop(f1->post_activation, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
        }
        
        else{
            fully_connected_back_prop(f1->pre_activation, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
        }
    }
    
    free(temp);
    free(temp2);
    free(temp3);
    
    return error2;
    
}


/* This function computes the derivative of weights and biases of a convolutional layer f2
 * applied to a previous fully connected layer f1, and returns the error of the last function of the
 * previous layer. For example:
 * if the previous layer applied only the pre_activation then the float* vector returned is the DL/df1->pre_activation
 * if the previous layer applied only pre_activation and post activation then the float* vector returned is DL/df1->post_activation
 * if the previous layer applied pre activation, post activation and dropout then the float* vector returned is
 * DL/df1->post_activation without the dropout applied, in this case the dropout must be applied during the backpropagation of f1.
 * 
 * Input:
 * 
 *             @ fcl* f1:= the fully-connected input layer
 *             @ cl* f2:= the convolutional current layer
 *             @ float* error:= the error passed
 * */ 
float* bp_fcl_cl(fcl* f1, cl* f2, float* error){
    int i,j,k;
    float* temp = (float*)calloc(f2->n_kernels*f2->rows1*f2->cols1,sizeof(float));
    float* temp2 = (float*)calloc(f2->n_kernels*f2->rows1*f2->cols1,sizeof(float));
    float* temp3 = (float*)calloc(f2->n_kernels*f2->rows1*f2->cols1,sizeof(float));
    float* error2 = (float*)calloc(f1->output,sizeof(float));
    
    /* computing backpropagation for f2*/
    if(f2->pooling_flag == MAX_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            if(f2->normalization_flag){
                max_pooling_back_prop(&f2->post_normalization[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &temp[i*f2->rows1*f2->cols1]);
            }
            else if(f2->activation_flag){
                max_pooling_back_prop(&f2->post_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &temp[i*f2->rows1*f2->cols1]);
            }
            else{
                max_pooling_back_prop(&f2->pre_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &temp[i*f2->rows1*f2->cols1]);
            }
        }
    }
    
    else if(f2->pooling_flag == AVARAGE_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            avarage_pooling_back_prop(&temp[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
        }
    }
    
    else{
        copy_array(error,temp,f2->n_kernels*f2->rows1*f2->cols1);
    }
    
    
    if(f2->normalization_flag){
        for(i = 0; i < f2->n_kernels; i++){
            for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                for(k = f2->padding1_rows; k < f2->cols1-f2->padding1_rows; k++){
                    if(f2->activation_flag)
                        local_response_normalization_back_prop(f2->post_activation,temp2,temp, i,j,k,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                    else
                        local_response_normalization_back_prop(f2->pre_activation,temp2,temp, i,j,k,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);

                }
            }
        }
        
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == NO_ACTIVATION){
            copy_array(temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        
    }
    
    else{
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
    }
    
    /* computing the weight and bias derivatives for f2 applied to f1 output*/
    if(f1->dropout_flag){
        if(f1->activation_flag){
            
            dot1D(f1->post_activation,f1->dropout_mask,temp2,f1->output);
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_back_prop(temp2, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            }

        }
        
        else{
            dot1D(f1->pre_activation,f1->dropout_mask,temp2,f1->output);
            
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_back_prop(temp2, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            }
        }
    }
    
    else{
        if(f1->activation_flag){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_back_prop(f1->post_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            }
        }
        
        else{
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_back_prop(f1->pre_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            }
        }
    }
    
    free(temp);
    free(temp2);
    free(temp3);
    
    return error2;
    
}


/* This function computes the derivative of weights and biases of a convolutional layer f2
 * applied to a previous convolutional layer f1, and returns the error of the last function of the
 * previous layer. For example:
 * if the previous layer applied only the pre_activation then the float* vector returned is the DL/df1->pre_activation
 * if the previous layer applied only pre_activation and post activation then the float* vector returned is DL/df1->post_activation
 * if the previous layer applied normallization then the float* vector returned is DL/df1->post_normalization
 * if the previous layer applied pooling then the float* vector returned is DL/df1->post_pooling
 * 
 * Input:
 * 
 *             @ cl* f1:= the convolutional input layer
 *             @ cl* f2:= the convolutional current layer
 *             @ float* error:= the error passed
 * */
float* bp_cl_cl(cl* f1, cl* f2, float* error){
    int i,j,k;
    float* temp = (float*)calloc(f2->n_kernels*f2->rows1*f2->cols1,sizeof(float));
    float* temp2 = (float*)calloc(f2->n_kernels*f2->rows1*f2->cols1,sizeof(float));
    float* temp3 = (float*)calloc(f2->n_kernels*f2->rows1*f2->cols1,sizeof(float));
    float* error2 = (float*)calloc(f2->channels*f2->input_rows*f2->input_cols,sizeof(float));
    
    /* computing backpropagation for f2*/
    if(f2->pooling_flag == MAX_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            if(f2->normalization_flag)
                max_pooling_back_prop(&f2->post_normalization[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &temp[i*f2->rows1*f2->cols1]);
            else if(f2->activation_flag)
                max_pooling_back_prop(&f2->post_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &temp[i*f2->rows1*f2->cols1]);
            else
                max_pooling_back_prop(&f2->pre_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &temp[i*f2->rows1*f2->cols1]);
        }
    }
    
    else if(f2->pooling_flag == AVARAGE_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            avarage_pooling_back_prop(&temp[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
        }
    }
    
    else{
        copy_array(error,temp,f2->n_kernels*f2->rows1*f2->cols1);
    }
    
    if(f2->normalization_flag){
        for(i = 0; i < f2->n_kernels; i++){
            for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                for(k = f2->padding1_rows; k < f2->cols1-f2->padding1_rows; k++){
                    if(f2->activation_flag)
                        local_response_normalization_back_prop(f2->post_activation,temp2,temp, i,j-f2->padding1_rows,k-f2->padding1_rows,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                    else
                        local_response_normalization_back_prop(f2->pre_activation,temp2,temp, i,j,k,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);

                }
            }
        }
        
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == NO_ACTIVATION){
            copy_array(temp2,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        
    }
    
    else{
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
        
        if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->n_kernels*f2->rows1*f2->cols1);
            dot1D(temp3,temp,temp,f2->n_kernels*f2->rows1*f2->cols1);
        }
    }
    
    /* computing the weight and bias derivatives for f2 applied to f1 output*/
    
    for(i = 0; i < f2->n_kernels; i++){
        if(f1->pooling_flag)
            convolutional_back_prop(f1->post_pooling, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
        else if(f1->normalization_flag)
            convolutional_back_prop(f1->post_normalization, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
        else if(f1->activation_flag)
            convolutional_back_prop(f1->post_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
        else
            convolutional_back_prop(f1->pre_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&temp[i*f2->rows1*f2->cols1],error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
    }
    
    
    free(temp);
    free(temp2);
    free(temp3);
    
    return error2;
    
}


/* This function computes the derivative of weights and biases of a fully-connected layer f2
 * applied to a previous convolutional layer f1, and returns the error of the last function of the
 * previous layer. For example:
 * if the previous layer applied only the pre_activation then the float* vector returned is the DL/df1->pre_activation
 * if the previous layer applied only pre_activation and post activation then the float* vector returned is DL/df1->post_activation
 * if the previous layer applied normallization then the float* vector returned is DL/df1->post_normalization
 * if the previous layer applied pooling then the float* vector returned is DL/df1->post_pooling
 * 
 * Input:
 * 
 *             @ cl* f1:= the convolutional input layer
 *             @ fcl* f2:= the fully-connected current layer
 *             @ float* error:= the error passed
 * 
 * Warning:
 *             if we have softmax as activation function of f2, (softmax can be applied only for the last fully-connected layers)
 *             then the error passed as param is not DL/Df2->post_activation but is L where L is the error
 * */
float* bp_cl_fcl(cl* f1, fcl* f2, float* error){
    int i;
    float* temp = (float*)calloc(f2->output,sizeof(float));
    float* temp2 = (float*)calloc(f2->input,sizeof(float));
    float* temp3 = (float*)calloc(f2->output,sizeof(float));
    float* error2 = (float*)calloc(f2->input,sizeof(float));
    
    /*computing the backpropagation for f2*/
    if(f2->dropout_flag){
        dot1D(error,f2->dropout_mask,temp,f2->output);
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_cross_entropy_reduced_form_with_softmax_array(f2->post_activation,  error,temp3, f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,temp,temp,f2->output);
        }
    }
    
    else{
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_cross_entropy_reduced_form_with_softmax_array(f2->post_activation,  error,temp3, f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,temp3,f2->output);
            dot1D(temp3,error,temp,f2->output);
        }
        
        else{
            copy_array(error,temp,f2->output);
        }
    }
    /* computing the weight and bias derivatives for f2 applied to f1 output*/
        if(f1->pooling_flag)
            fully_connected_back_prop(f1->post_pooling, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
        else if(f1->normalization_flag)
            fully_connected_back_prop(f1->post_normalization, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
        else if(f1->activation_flag)
            fully_connected_back_prop(f1->post_activation, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
        else
            fully_connected_back_prop(f1->pre_activation, temp, f2->weights,error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
    
    
    free(temp);
    free(temp2);
    free(temp3);
    
    return error2;
    
}


/* This function computes the feed-forward for a model m. each layer at the index l makes the feed-forward
 * for the first layer at the index l-1. if the input is a 1d array then you should split its dimension
 * in 3 dimension to turn the input in a tensor, for example:
 * I have an input array of legth 59, then i can split this in 3 dimensions: depth = 1, row = 1, cols = 59
 * 
 * Input:
 *             
 *             @ model* m:= the model with the layers
 *             @ int tensor_depth:= the depth of the input tensor
 *             @ int tensor_i:= the number of rows of the tensor
 *             @ int tensor_j:= the number of columns of the tensor
 *             @ float* input:= your input array
 * 
 * */
void model_tensor_input_ff(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input){
    if(m == NULL)
        return;
    int i,j,z,w,count,count2,z2,k1 = 0, k2 = 0, k3 = 0;
    
    /* Setting the input inside a convolutional structure*/
    cl* temp = (cl*)malloc(sizeof(cl));
    temp->post_activation = (float*)malloc(sizeof(float)*tensor_depth*tensor_i*tensor_j);
    temp->normalization_flag = NO_NORMALIZATION;
    temp->pooling_flag = NO_POOLING;
    temp->activation_flag = SIGMOID;
    temp->n_kernels = tensor_depth;
    temp->rows1 = tensor_i;
    temp->cols1 = tensor_j;
    copy_array(input,temp->post_activation,tensor_depth*tensor_i*tensor_j);
        
    /* apply the feed forward to the model*/
    for(i = 0; i < m->layers; i++){
        for(j = 0; j < m->layers && m->sla[i][j] != 0; j++){
            
                
            if(!i){
                if(m->sla[i][j] == FCLS){
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    ff_cl_fcl(temp,m->fcls[k1]);
                    k1++;
                }
                
                else if(m->sla[i][j] == CLS){
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    ff_cl_cl(temp,m->cls[k2]);
                    k2++;
                }
                
                else if(m->sla[i][j] == RLS){
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k3; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                
                    
                    if(m->rls[z]->cls[k3-count]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(k3-count == 0)
                        copy_array(temp->post_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    ff_cl_cl(temp,m->rls[z]->cls[k3-count]);
                    
                    if(k3-count == m->rls[z]->n_cl-1){
                        if(m->rls[z]->cls[k3-count]->pooling_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_pooling,m->rls[z]->cls[k3-count]->post_pooling,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows2*m->rls[z]->cls[k3-count]->cols2);
                        else if(m->rls[z]->cls[k3-count]->normalization_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_normalization,m->rls[z]->cls[k3-count]->post_normalization,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        else if(m->rls[z]->cls[k3-count]->activation_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_activation,m->rls[z]->cls[k3-count]->post_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        else
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->pre_activation,m->rls[z]->cls[k3-count]->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                    }
                    
                    k3++;
                    
                    
                }
            }
            
            else{
                
                if(m->sla[i][j] == FCLS){
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        ff_fcl_fcl(m->fcls[k1-1],m->fcls[k1]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        ff_cl_fcl(m->cls[k2-1],m->fcls[k1]);
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        ff_cl_fcl(m->rls[z2]->cls[k3-1-count2],m->fcls[k1]);
                    }
                    
                    k1++;
                }
                
                else if(m->sla[i][j] == CLS){
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        ff_fcl_cl(m->fcls[k1-1],m->cls[k2]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        ff_cl_cl(m->cls[k2-1],m->cls[k2]);
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        ff_cl_cl(m->rls[z2]->cls[k3-1-count2],m->cls[k2]);
                    }
                    k2++;
                }
                
                else if(m->sla[i][j] == RLS){
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k3; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                
                    
                    if(m->rls[z]->cls[k3-count]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    if(m->sla[i-1][0] == FCLS){
                        if(k3-count == 0){
                            if(m->fcls[k1-1]->dropout_flag){
                                if(m->fcls[k1-1]->activation_flag){
                                    dot1D(m->fcls[k1-1]->post_activation,m->fcls[k1-1]->dropout_mask,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                }
                                else{
                                    dot1D(m->fcls[k1-1]->pre_activation,m->fcls[k1-1]->dropout_mask,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                }
                            }
                            else{
                                if(m->fcls[k1-1]->activation_flag){
                                    copy_array(m->fcls[k1-1]->post_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                }
                                else{
                                    copy_array(m->fcls[k1-1]->pre_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                }
                            }
                        }
                    
                        ff_fcl_cl(m->fcls[k1-1],m->rls[z]->cls[k3-count]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        if(k3-count == 0){
                            if(m->cls[k2-1]->pooling_flag){
                                copy_array(m->cls[k2-1]->post_pooling,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                            }
                            else if(m->cls[k2-1]->normalization_flag){
                                copy_array(m->cls[k2-1]->post_normalization,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                            }
                            
                            else if(m->cls[k2-1]->activation_flag){
                                copy_array(m->cls[k2-1]->post_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                            }
                            else{
                                copy_array(m->cls[k2-1]->pre_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);                                
                            }
                        }
                        ff_cl_cl(m->cls[k2-1],m->rls[z]->cls[k3-count]);
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                        
                        if(k3-count == 0){
                            if(m->rls[z2]->cls[k3-1-count2]->pooling_flag){
                                copy_array(m->rls[z2]->cls[k3-1-count2]->post_pooling,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                            }
                            else if(m->rls[z2]->cls[k3-1-count2]->normalization_flag){
                                copy_array(m->rls[z2]->cls[k3-1-count2]->post_normalization,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                            }
                            
                            else if(m->rls[z2]->cls[k3-1-count2]->activation_flag){
                                copy_array(m->rls[z2]->cls[k3-1-count2]->post_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                            }
                            else{
                                copy_array(m->rls[z2]->cls[k3-1-count2]->pre_activation,m->rls[z]->input,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);                                
                            }
                        }
                        
                        ff_cl_cl(m->rls[z2]->cls[k3-1-count2],m->rls[z]->cls[k3-count]);
                    }
                    
                    if(k3-count == m->rls[z]->n_cl-1){
                        if(m->rls[z]->cls[k3-count]->pooling_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_pooling,m->rls[z]->cls[k3-count]->post_pooling,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows2*m->rls[z]->cls[k3-count]->cols2);
                        else if(m->rls[z]->cls[k3-count]->normalization_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_normalization,m->rls[z]->cls[k3-count]->post_normalization,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        else if(m->rls[z]->cls[k3-count]->activation_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_activation,m->rls[z]->cls[k3-count]->post_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        else
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->pre_activation,m->rls[z]->cls[k3-count]->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                    }
                    
                    k3++;
                    
                    
                }
                
            }
            
        }
    }
    
    free(temp->post_activation);
    free(temp);
}





/* This function computes the back-propagation for a model m. each first layer at the index l makes the backprop
 * from the first layer at the index l+1. if the input is a 1d array then you should split its dimension
 * in 3 dimension to turn the input in a tensor, for example:
 * I have an input array of legth 59, then i can split this in 3 dimensions: depth = 1, row = 1, cols = 59
 * 
 * Input:
 *             
 *             @ model* m:= the model with the layers
 *             @ int tensor_depth:= the depth of the input tensor
 *             @ int tensor_i:= the number of rows of the tensor
 *             @ int tensor_j:= the number of columns of the tensor
 *             @ float* input:= your input array
 *                @ float* error:= the error of the last layer of the last function computed
 *                @ int error_dimension:= the dimension of the float* error vector
 * 
 * */
float* model_tensor_input_bp(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension){
    if(m == NULL)
        return NULL;
        
    int i,j,z,w,count,count2,z2,k1 = m->n_fcl, k2 = m->n_cl, k3 = 0;
    for(i = 0; i < m->n_rl; i++){
        k3+=m->rls[i]->n_cl;
    }
 
    
    /* Setting the input inside a convolutional structure*/
    cl* temp = (cl*)malloc(sizeof(cl));
    temp->post_activation = (float*)malloc(sizeof(float)*tensor_depth*tensor_i*tensor_j);
    temp->normalization_flag = NO_NORMALIZATION;
    temp->pooling_flag = NO_POOLING;
    temp->activation_flag = SIGMOID;
    temp->n_kernels = tensor_depth;
    temp->rows1 = tensor_i;
    temp->cols1 = tensor_j;
    copy_array(input,temp->post_activation,tensor_depth*tensor_i*tensor_j);
    
    float* error1 = (float*)malloc(sizeof(float)*error_dimension);
    copy_array(error,error1,error_dimension);
        
    float* error2 = NULL;    
    float* error_residual = NULL;    
    /* apply the backpropagation to the model*/
    for(i = m->layers-1; i >= 0; i--){
        for(j = 0; j < 1 && m->sla[i][j] != 0; j++){
            
            
            if(!i){
                if(m->sla[i][j] == FCLS){
                    k1--;
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    error2 = bp_cl_fcl(temp,m->fcls[k1],error1);
                    
                }
                
                else if(m->sla[i][j] == CLS){
                    k2--;
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    error2 = bp_cl_cl(temp,m->cls[k2],error1);
                    
                    
                }
                
                else if(m->sla[i][j] == RLS){
                    k3--;
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k3; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                    
                    if(k3-count == m->rls[z]->n_cl-1){
                        free(error_residual);
                        error_residual = (float*)calloc(m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols,sizeof(float));
                        copy_array(error1,error_residual,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);   
                    }
                    
                    if(m->rls[z]->cls[k3-count]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    
                    error2 = bp_cl_cl(temp,m->rls[z]->cls[k3-count],error1);
                    
                    
                    
                    if(k3-count == 0)
                        sum1D(error2,error_residual,error2,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    
                    
                    
                }
            }
            
            else{
                
                if(m->sla[i][j] == FCLS){
                    k1--;
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        error2 = bp_fcl_fcl(m->fcls[k1-1],m->fcls[k1],error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error2 = bp_cl_fcl(m->cls[k2-1],m->fcls[k1], error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        error2 = bp_cl_fcl(m->rls[z2]->cls[k3-1-count2],m->fcls[k1],error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                    
                }
                
                else if(m->sla[i][j] == CLS){
                    k2--;
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        error2 = bp_fcl_cl(m->fcls[k1-1],m->cls[k2],error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error2 = bp_cl_cl(m->cls[k2-1],m->cls[k2],error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        error2 = bp_cl_cl(m->rls[z2]->cls[k3-1-count2],m->cls[k2],error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                }
                
                else if(m->sla[i][j] == RLS){
                    k3--;
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k3; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                    
                    if(k3-count == m->rls[z]->n_cl-1){
                        
                        free(error_residual);
                        error_residual = (float*)calloc(m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols,sizeof(float));
                        copy_array(error1,error_residual,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);                     
                        
                    }
                    
                    if(m->rls[z]->cls[k3-count]->activation_flag == SOFTMAX){
                        printf("Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    if(m->sla[i-1][0] == FCLS){
                        
                    
                        error2 = bp_fcl_cl(m->fcls[k1-1],m->rls[z]->cls[k3-count],error1);
                        free(error1);
                        error1 = error2;
                        
                        
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error2 = bp_cl_cl(m->cls[k2-1],m->rls[z]->cls[k3-count],error1);
                        free(error1);
                        error1 = error2;
                        
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                        
                        error2 = bp_cl_cl(m->rls[z2]->cls[k3-1-count2],m->rls[z]->cls[k3-count],error1);
                        free(error1);
                        error1 = error2;
                    }
                    
                    
                    
                    if(k3-count == 0)
                        sum1D(error2,error_residual,error2,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    
                    
                    
                }
                
            }
            
        }
    }
    
    free(error_residual);
    free(error1);
    free(temp->post_activation);
    free(temp);
    if(!bool_is_real(error2[0])){
        printf("Error: nan occurred, probably due to the exploiting gradient problem, or you just found a perfect function that match your data and you should not keep training\n");
        exit(1);
    }
    return error2;
}

/* This function can update the model of the network using the adam algorithm or the nesterov momentum
 * 
 * Input:
 * 
 *             @ model* m:= the model that must be update
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *             @ int mini_batch_size:= the batch used
 *             @ int gradient_descent_flag:= NESTEROV or ADAM (1,2)
 * 
 * */
void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag){
    if(m == NULL)
        return;
    if(gradient_descent_flag == NESTEROV){    
        update_residual_layer_nesterov(m,lr,momentum,mini_batch_size);
        update_convolutional_layer_nesterov(m,lr,momentum,mini_batch_size);
        update_fully_connected_layer_nesterov(m,lr,momentum,mini_batch_size);
    }
    
    /* we must adding else if gradient_descent_flag == ADAM*/
    
    

}

/* This function sum the partial derivatives in model m1 and m2 in m3
 * 
 * Input:
 *     
 *             @ model* m:= first input model
 *             @ model* m2:= second input model
 *             @ model* m3:= output model
 * 
 * */
void sum_model_partial_derivatives(model* m, model* m2, model* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        printf("Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    sum_fully_connected_layers_partial_derivatives(m,m2,m3);
    sum_convolutional_layers_partial_derivatives(m,m2,m3);
    sum_residual_layers_partial_derivatives(m,m2,m3);
}
