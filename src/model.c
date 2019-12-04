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
        fprintf(stderr,"Error: layers must be > 0 and at least one between n_rl, n_cl, n_fcl must be > 0\n");
        exit(1); 
    }
    
    int i,j,k, position, count, k1,k2,k3;
    
    
    /*checking if the residual layer has the right size from the input to the output*/
    for(i = 0; i < n_rl; i++){
        if(rls[i]->cls[rls[i]->n_cl-1]->pooling_flag){
            if(rls[i]->cls[rls[i]->n_cl-1]->n_kernels*rls[i]->cls[rls[i]->n_cl-1]->rows2*rls[i]->cls[rls[i]->n_cl-1]->cols2 != rls[i]->channels*rls[i]->input_rows*rls[i]->input_cols){
                fprintf(stderr,"Error: you have a residual layer (%d layer) where the input size doesn't correspond to the last convolutional layer size of the residual layer\n",rls[i]->cls[rls[i]->n_cl-1]->layer);
                exit(1);
            }
        }
        
        else{
            if(rls[i]->cls[rls[i]->n_cl-1]->n_kernels*rls[i]->cls[rls[i]->n_cl-1]->rows1*rls[i]->cls[rls[i]->n_cl-1]->cols1 != rls[i]->channels*rls[i]->input_rows*rls[i]->input_cols){
                fprintf(stderr,"Error: you have a residual layer (%d layer) where the input size doesn't correspond to the last convolutional layer size of the residual layer\n", rls[i]->cls[rls[i]->n_cl-1]->layer);
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
                fprintf(stderr,"Error: you have a residual layer with no sequential sub-convolutional-layers\n");
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
            fprintf(stderr,"Error: your layers are not sequential, missing the layer with index: %d\n",i);
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
    m->error = NULL;
    m->error_alpha = NULL;
    m->beta1_adam = BETA1_ADAM;
    m->beta2_adam = BETA2_ADAM;
    m->error_flag = NO_SET;
    
    
    for(i = 0; i < layers && sla[i][0]; i++);
    
    if(i == layers)
        i--;
    
    if(sla[i][0] == FCLS){
        if(m->fcls[m->n_fcl-1]->dropout_flag)
            m->output_layer = m->fcls[m->n_fcl-1]->dropout_temp;
        else if(m->fcls[m->n_fcl-1]->activation_flag)
            m->output_layer = m->fcls[m->n_fcl-1]->post_activation;
        else
            m->output_layer = m->fcls[m->n_fcl-1]->pre_activation;
    }
    
    else if(sla[i][0] == CLS){
        if(m->cls[m->n_cl-1]->pooling_flag)
            m->output_layer = m->cls[m->n_cl-1]->post_pooling;
        else if(m->cls[m->n_cl-1]->normalization_flag)
            m->output_layer = m->cls[m->n_cl-1]->post_normalization;
        else if(m->cls[m->n_cl-1]->activation_flag)
            m->output_layer = m->cls[m->n_cl-1]->post_activation;
        else
            m->output_layer = m->cls[m->n_cl-1]->pre_activation;
    }
    
    else{
        if(m->rls[m->n_rl-1]->cl_output->activation_flag)
            m->output_layer = m->rls[m->n_rl-1]->cl_output->post_activation;
        else
            m->output_layer = m->rls[m->n_rl-1]->cl_output->pre_activation;
    }
        
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
    free(m->error);
    free(m->error_alpha);
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
    if(m->error!=NULL)
        set_model_error(copy,m->error_flag,m->error_threshold1,m->error_threshold2,m->error_gamma,m->error_alpha,m->output_dimension);
    return copy;
}



/* This function copies a model using the paste function for the layers
 * see layers.c file
 * 
 * Input:
 *         
 *             @ model* m:= the model that must be copied
 *             @ model* copy:= the model where m is copied
 * 
 * */
void paste_model(model* m, model* copy){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_fcl; i++){
        paste_fcl(m->fcls[i],copy->fcls[i]);
    }
    for(i = 0; i < m->n_cl; i++){
        paste_cl(m->cls[i],copy->cls[i]);
    }
    for(i = 0; i < m->n_rl; i++){
        paste_rl(m->rls[i],copy->rls[i]);
    }
    return;
}

/* This function copies a model with the rule: teta_i:= teta_j*tau +(1-tau)*teta_i
 * 
 * Input:
 *         
 *             @ model* m:= the model that must be copied
 *             @ model* copy:= the model where m is copied
 *                @ float tau:= the tau param
 * 
 * */
void slow_paste_model(model* m, model* copy, float tau){
    if(m == NULL)
        return;
    int i;
    
    for(i = 0; i < m->n_fcl; i++){
        slow_paste_fcl(m->fcls[i],copy->fcls[i],tau);
    }
    for(i = 0; i < m->n_cl; i++){
        slow_paste_cl(m->cls[i],copy->cls[i],tau);
    }
    for(i = 0; i < m->n_rl; i++){
        slow_paste_rl(m->rls[i],copy->rls[i],tau);
    }
    return;
}
/* This function resets a model
 * returns a model equal to the one as input but with all resetted except for weights and biases
 * */
model* reset_model(model* m){
    if(m == NULL)
        return NULL;
    int i;
    for(i = 0; i < m->n_fcl; i++){
        reset_fcl(m->fcls[i]);
    }
    for(i = 0; i < m->n_cl; i++){
        reset_cl(m->cls[i]);
    }
    for(i = 0; i < m->n_rl; i++){
        reset_rl(m->rls[i]);
    }
    
    if(m->error != NULL){
        for(i = 0; i < m->output_dimension; i++){
            m->error[i] = 0;
        }
    }
    return m;
}


/* this function compute the space allocated by the arrays of m
 * 
 * Input:
 * 
 *             model* m:= the structure model
 * 
 * */
unsigned long long int size_of_model(model* m){
    int i;
    unsigned long long int sum = 0;
    for(i = 0; i < m->n_fcl; i++){
        sum+= size_of_fcls(m->fcls[i]);
    }
    
    
    for(i = 0; i < m->n_cl; i++){
        sum+= size_of_cls(m->cls[i]);
    }
    
    
    for(i = 0; i < m->n_rl; i++){
        sum+= size_of_rls(m->rls[i]);
    }
    
    sum+= (( unsigned long long int)(m->layers*m->layers*sizeof(int)));
    return sum;
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
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&m->layers,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fwrite(&m->n_rl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fwrite(&m->n_cl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fwrite(&m->n_fcl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the model\n");
        exit(1);
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
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
        fprintf(stderr,"Error: error during the opening of the file %s\n",file);
        exit(1);
    }
    
    int layers = 0,n_cl = 0,n_rl = 0,n_fcl = 0;
    
    i = fread(&layers,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_rl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_cl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_fcl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    

    rl** rls;
    cl** cls;
    fcl** fcls;
    
    if(!n_rl)
        rls = NULL;
    else
        rls = (rl**)malloc(sizeof(rl*)*n_rl);
    if(!n_cl)
        cls = NULL;
    else
        cls = (cl**)malloc(sizeof(cl*)*n_cl);
    if(!n_fcl)
        fcls = NULL;
    else
        fcls = (fcl**)malloc(sizeof(fcl*)*n_fcl);
    
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
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file);
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
        fprintf(stderr,"Error: the sizes between 2 fully-connected layers don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    int i;
    
    /* computing the pre-activation array for f2 from f1*/
    
    /* no activation for f1*/
    if(f1->activation_flag == NO_ACTIVATION){
        if(f1->dropout_flag == NO_DROPOUT){
            fully_connected_feed_forward(f1->pre_activation, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                fully_connected_feed_forward(f1->dropout_temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                mul_value(f2->pre_activation,f1->dropout_threshold,f1->dropout_temp,f2->input);
                fully_connected_feed_forward(f1->dropout_temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
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
                fully_connected_feed_forward(f1->dropout_temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                mul_value(f1->post_activation,f1->dropout_threshold,f1->dropout_temp,f2->input);
                fully_connected_feed_forward(f1->dropout_temp, f2->pre_activation, f2->weights,f2->biases, f2->input, f2->output);
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
    else if(f2->activation_flag == LEAKY_RELU)
        leaky_relu_array(f2->pre_activation,f2->post_activation,f2->output);
    
    /* setting the dropout mask, if dropout flag is != 0*/
    if(f2->dropout_flag){
        set_dropout_mask(f2->output, f2->dropout_mask, f2->dropout_threshold);
        if(f2->dropout_flag == DROPOUT){
            if(f2->activation_flag)
                get_dropout_array(f2->output,f2->dropout_mask,f2->pre_activation,f2->dropout_temp);
            else
                get_dropout_array(f2->output,f2->dropout_mask,f2->post_activation,f2->dropout_temp);
        }
    }

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
        fprintf(stderr,"Error: the sizes between an input fully-connected layer and an output convolutional layer don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }

    int i,j,k,z;
    /* f2 pre activation with no activation for f1*/
     if(f1->activation_flag == NO_ACTIVATION){
        if(f1->dropout_flag == NO_DROPOUT){
            if(f2->convolutional_flag == CONVOLUTION){
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_feed_forward(f1->pre_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                }
            }
            
            else{
                copy_array(f1->pre_activation,f2->pooltemp,f2->channels*f2->input_rows*f2->input_cols);
            }
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                if(f2->convolutional_flag == CONVOLUTION){
                    for(i = 0; i < f2->n_kernels; i++){
                        convolutional_feed_forward(f1->dropout_temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                    }
                }
                
                else{
                    copy_array(f1->dropout_temp,f2->pooltemp,f2->channels*f2->input_rows*f2->input_cols);
                }
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                mul_value(f1->pre_activation,f1->dropout_threshold,f1->dropout_temp,f1->output);
                if(f2->convolutional_flag == CONVOLUTION){
                    for(i = 0; i < f2->n_kernels; i++){
                        convolutional_feed_forward(f1->dropout_temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                    }
                }
                
                else{
                    copy_array(f1->dropout_temp,f2->pooltemp,f2->channels*f2->input_rows*f2->input_cols);
                }
                
            }
        }
    }
    
    /* f2 pre activation with activation for f1*/
    else{
        if(f1->dropout_flag == NO_DROPOUT){
            if(f2->convolutional_flag == CONVOLUTION){
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_feed_forward(f1->post_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                }
            }
            
            else{
                copy_array(f1->post_activation,f2->pooltemp,f2->channels*f2->input_rows*f2->input_cols);
            }
            
        }
        else{
            if(f1->dropout_flag == DROPOUT){
                if(f2->convolutional_flag == CONVOLUTION){
                    for(i = 0; i < f2->n_kernels; i++){
                        convolutional_feed_forward(f1->dropout_temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                    }
                }
                else{
                    copy_array(f1->dropout_temp,f2->pooltemp,f2->channels*f2->input_rows*f2->input_cols);
                }
            }
            
            else if(f1->dropout_flag == DROPOUT_TEST){
                mul_value(f1->post_activation,f1->dropout_threshold,f1->dropout_temp,f1->output);
                if(f2->convolutional_flag == CONVOLUTION){
                    for(i = 0; i < f2->n_kernels; i++){
                        convolutional_feed_forward(f1->dropout_temp, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
                    }
                }
                else{
                    copy_array(f1->dropout_temp,f2->pooltemp,f2->channels*f2->input_rows*f2->input_cols);
                }
            }
        }
    }
    
    if(f2->convolutional_flag == CONVOLUTION){
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
        
        else if(f2->activation_flag == LEAKY_RELU){
            if(f2->padding1_rows){
                for(i = 0; i < f2->n_kernels; i++){
                    for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                        leaky_relu_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                    }
                }
            }
            
            else
                leaky_relu_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

        }
        /* normalization for f2, if there is any normalization*/
        if(f2->normalization_flag == LOCAL_RESPONSE_NORMALIZATION){
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
        
        else if(f2->normalization_flag == GROUP_NORMALIZATION){
            if(f2->activation_flag != NO_ACTIVATION)
                group_normalization_feed_forward(f2->post_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->padding1_rows,f2->padding1_cols,f2->post_normalization);
            else
                group_normalization_feed_forward(f2->pre_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->padding1_rows,f2->padding1_cols,f2->post_normalization);
        }
    }
    
    /* pooling for f2, if there is any pooling*/
    if(f2->pooling_flag != NO_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            if(f2->convolutional_flag == NO_CONVOLUTION){
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->pooltemp[i*f2->input_rows*f2->input_cols], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->input_rows, f2->input_cols, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->pooltemp[i*f2->input_rows*f2->input_cols], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->input_rows, f2->input_cols, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
               
            else if(f2->normalization_flag != NO_NORMALIZATION){
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
        fprintf(stderr,"Error: the sizes between an input convolutional layer and an output fully-connected layer don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    else if(!f1->pooling_flag && f1->n_kernels*f1->rows1*f1->cols1 != f2->input){
        fprintf(stderr,"Error: the sizes between an input convolutional layer and an output fully-connected layer don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
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
    else if(f2->activation_flag == LEAKY_RELU)
        leaky_relu_array(f2->pre_activation,f2->post_activation,f2->output);
    

    
    /* setting the dropout mask, if dropout flag is != 0*/
    if(f2->dropout_flag){
        set_dropout_mask(f2->output, f2->dropout_mask, f2->dropout_threshold);
        if(f2->dropout_flag == DROPOUT){
            if(f2->activation_flag)
                get_dropout_array(f2->output,f2->dropout_mask,f2->pre_activation,f2->dropout_temp);
            else
                get_dropout_array(f2->output,f2->dropout_mask,f2->post_activation,f2->dropout_temp);
        }
    }
    
    
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
        fprintf(stderr,"Error: the sizes between 2 convolutional layers don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }
    
    else if(!f1->pooling_flag && f1->n_kernels*f1->rows1*f1->cols1 != f2->channels*f2->input_rows*f2->input_cols){
        fprintf(stderr,"Error: the sizes between 2 convolutional layers don't match, layer1: %d, layer2: %d\n",f1->layer,f2->layer);
        exit(1);
    }

    int i,j,k,z;
    /* pooling for f1*/
    if(f1->pooling_flag){
        if(f2->convolutional_flag == CONVOLUTION){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_feed_forward(f1->post_pooling, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
            }
        }
        
        else{
            copy_array(f1->post_pooling,f2->pooltemp,f2->input_rows*f2->input_cols*f2->channels);
        }    
    }
            
    /* no pooling for f1, but normalization*/
    else if(f1->normalization_flag){
        if(f2->convolutional_flag == CONVOLUTION){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_feed_forward(f1->post_normalization, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
            }
        }
        
        else{
            copy_array(f1->post_normalization,f2->pooltemp,f2->input_rows*f2->input_cols*f2->channels);
        }   
    }
    /* no pooling, no normalization for f1, but activation*/
    else if(f1->activation_flag){
        if(f2->convolutional_flag == CONVOLUTION){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_feed_forward(f1->post_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
            }
        }
        
        else{
            copy_array(f1->post_activation,f2->pooltemp,f2->input_rows*f2->input_cols*f2->channels);
        }
    }
    /* no pooling, no normalization, no activation for f1*/
    else{
        if(f2->convolutional_flag == CONVOLUTION){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_feed_forward(f1->pre_activation, f2->kernels[i], f2->input_rows, f2->input_cols, f2->kernel_rows, f2->kernel_cols, f2->biases[i], f2->channels, &f2->pre_activation[i*f2->rows1*f2->cols1], f2->stride1_rows, f2->padding1_rows);
            }  
        } 
        
        else{
            copy_array(f1->pre_activation,f2->pooltemp,f2->input_rows*f2->input_cols*f2->channels);
        }
    }
    
    
    if(f2->convolutional_flag == CONVOLUTION){
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
        
        else if(f2->activation_flag == LEAKY_RELU){
            if(f2->padding1_rows){
                for(i = 0; i < f2->n_kernels; i++){
                    for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                        leaky_relu_array(&f2->pre_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_rows],&f2->post_activation[i*f2->rows1*f2->cols1 + j*f2->cols1 + f2->padding1_cols],f2->cols1-2*f2->padding1_rows);
                    }
                }
            }
            
            else
                leaky_relu_array(f2->pre_activation,f2->post_activation,f2->n_kernels*f2->rows1*f2->cols1);

        }
        /* normalization for f2, if there is any normalization*/
        if(f2->normalization_flag == LOCAL_RESPONSE_NORMALIZATION){
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
        
        else if(f2->normalization_flag == GROUP_NORMALIZATION){
            if(f2->activation_flag != NO_ACTIVATION)
                group_normalization_feed_forward(f2->post_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->padding1_rows,f2->padding1_cols,f2->post_normalization);
            else
                group_normalization_feed_forward(f2->pre_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->padding1_rows,f2->padding1_cols,f2->post_normalization);
        }
    }
    
    /* pooling for f2, if there is any pooling*/
    if(f2->pooling_flag){
        for(i = 0; i < f2->n_kernels; i++){
            if(f2->convolutional_flag == NO_CONVOLUTION){
                if(f2->pooling_flag == MAX_POOLING){
                    max_pooling_feed_forward(&f2->pooltemp[i*f2->input_rows*f2->input_cols], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->input_rows, f2->input_cols, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
                else{
                    avarage_pooling_feed_forward(&f2->pooltemp[i*f2->input_rows*f2->input_cols], &f2->post_pooling[i*f2->rows2*f2->cols2], f2->input_rows, f2->input_cols, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
                }
            }
            else if(f2->normalization_flag){
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
    
    /*computing the backpropagation for f2*/
    if(f2->dropout_flag){
        dot1D(error,f2->dropout_mask,f2->temp,f2->output);
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_softmax_array(f2->pre_activation,f2->temp3,f2->post_activation,f2->temp,f2->output);
            copy_array(f2->temp3,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == LEAKY_RELU){
            derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
    }
    
    else{
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_softmax_array(f2->pre_activation,f2->temp,f2->post_activation,error,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == LEAKY_RELU){
            derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        
        else{
            copy_array(error,f2->temp,f2->output);
        }
    }
    /* computing the weight and bias derivatives for f2 applied to f1 output*/
    if(f1->dropout_flag){
        fully_connected_back_prop(f1->dropout_temp, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
    }
    
    else{
        if(f1->activation_flag){
            fully_connected_back_prop(f1->post_activation, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
        }
        
        else{
            fully_connected_back_prop(f1->pre_activation, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input,f2->output);
        }
    }
    
    return f2->error2;
    
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
    /* computing backpropagation for f2*/
    if(f2->pooling_flag == MAX_POOLING){
        if(f2->convolutional_flag == CONVOLUTION){
            for(i = 0; i < f2->n_kernels; i++){
                if(f2->normalization_flag){
                    max_pooling_back_prop(&f2->post_normalization[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                }
                else if(f2->activation_flag){
                    max_pooling_back_prop(&f2->post_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                }
                else{
                    max_pooling_back_prop(&f2->pre_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                }
            }
        }
        
        else{
            if(f1->dropout_flag){
                if(f1->activation_flag)    
                    dot1D(f1->post_activation,f1->dropout_mask,f2->temp2,f1->output);
                else    
                    dot1D(f1->pre_activation,f1->dropout_mask,f2->temp2,f1->output);
                
                for(i = 0; i < f2->n_kernels; i++){
                    max_pooling_back_prop(&f2->temp2[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                }            
            }
            
            else{
                if(f1->activation_flag){
                    for(i = 0; i < f2->n_kernels; i++){
                        max_pooling_back_prop(&f1->post_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                    }
                }
                
                else{
                    for(i = 0; i < f2->n_kernels; i++){
                        max_pooling_back_prop(&f1->pre_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                    }
                }
            }
        }
    }
    
    else if(f2->pooling_flag == AVARAGE_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            avarage_pooling_back_prop(&f2->temp[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
        }
    }
    
    else{
        copy_array(error,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
    }
    
    if(f2->convolutional_flag == CONVOLUTION){
        if(f2->normalization_flag == LOCAL_RESPONSE_NORMALIZATION){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    for(k = f2->padding1_rows; k < f2->cols1-f2->padding1_rows; k++){
                        if(f2->activation_flag)
                            local_response_normalization_back_prop(f2->post_activation,f2->temp2,f2->temp, i,j,k,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                        else
                            local_response_normalization_back_prop(f2->pre_activation,f2->temp2,f2->temp, i,j,k,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);

                    }
                }
            }
            
            if(f2->activation_flag == SIGMOID){
                derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == RELU){
                derivative_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == TANH){
                derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == NO_ACTIVATION){
                copy_array(f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            
        }
        
        else if(f2->normalization_flag == GROUP_NORMALIZATION){
            
            if(f2->activation_flag)
                group_normalization_back_propagation(f2->post_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->temp,f2->padding1_rows,f2->padding1_cols,f2->temp2);
            else
                group_normalization_back_propagation(f2->pre_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->temp,f2->padding1_rows,f2->padding1_cols,f2->temp2);

            
            if(f2->activation_flag == SIGMOID){
                derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == RELU){
                derivative_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == TANH){
                derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == NO_ACTIVATION){
                copy_array(f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            
        }
        
        else{
            if(f2->activation_flag == SIGMOID){
                derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == RELU){
                derivative_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == TANH){
                derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
        }
        
        /* computing the weight and bias derivatives for f2 applied to f1 output*/
        if(f1->dropout_flag){
            for(i = 0; i < f2->n_kernels; i++){
                convolutional_back_prop(f1->dropout_temp, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            }
       }
        
        else{
            if(f1->activation_flag){
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_back_prop(f1->post_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
                }
            }
            
            else{
                for(i = 0; i < f2->n_kernels; i++){
                    convolutional_back_prop(f1->pre_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
                }
            }
        }
        
        return f2->error2;
    }
    
    else
        return f2->temp;
    
    
    
    
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

    
    /* computing backpropagation for f2*/
    if(f2->pooling_flag == MAX_POOLING){
        if(f2->convolutional_flag == CONVOLUTION){
            for(i = 0; i < f2->n_kernels; i++){
                if(f2->normalization_flag)
                    max_pooling_back_prop(&f2->post_normalization[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                else if(f2->activation_flag)
                    max_pooling_back_prop(&f2->post_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                else
                    max_pooling_back_prop(&f2->pre_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
            }
        }
        
        else{
            for(i = 0; i < f2->n_kernels; i++){
                if(f1->pooling_flag)
                    max_pooling_back_prop(&f1->post_pooling[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                else if(f1->normalization_flag)
                    max_pooling_back_prop(&f1->post_normalization[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                else if(f1->activation_flag)
                    max_pooling_back_prop(&f1->post_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
                else
                    max_pooling_back_prop(&f1->pre_activation[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows, &f2->temp[i*f2->rows1*f2->cols1]);
            }
            
        }
    }
    
    else if(f2->pooling_flag == AVARAGE_POOLING){
        for(i = 0; i < f2->n_kernels; i++){
            avarage_pooling_back_prop(&f2->temp[i*f2->rows1*f2->cols1], &error[i*f2->rows2*f2->cols2], f2->rows1, f2->cols1, f2->pooling_rows, f2->pooling_cols, f2->stride2_rows, f2->padding2_rows);
        }
    }
    
    else{
        copy_array(error,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
    }
    
    if(f2->convolutional_flag == CONVOLUTION){
        if(f2->normalization_flag == LOCAL_RESPONSE_NORMALIZATION){
            for(i = 0; i < f2->n_kernels; i++){
                for(j = f2->padding1_rows; j < f2->rows1-f2->padding1_rows; j++){
                    for(k = f2->padding1_rows; k < f2->cols1-f2->padding1_rows; k++){
                        if(f2->activation_flag)
                            local_response_normalization_back_prop(f2->post_activation,f2->temp2,f2->temp, i,j-f2->padding1_rows,k-f2->padding1_rows,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);
                        else
                            local_response_normalization_back_prop(f2->pre_activation,f2->temp2,f2->temp, i,j,k,f2->n_kernels,f2->rows1, f2->cols1, N_NORMALIZATION,BETA_NORMALIZATION,ALPHA_NORMALIZATION,K_NORMALIZATION);

                    }
                }
            }
            
            if(f2->activation_flag == SIGMOID){
                derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == RELU){
                derivative_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == TANH){
                derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == NO_ACTIVATION){
                copy_array(f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            
        }
        
        else if(f2->normalization_flag == GROUP_NORMALIZATION){
            
            if(f2->activation_flag)
                group_normalization_back_propagation(f2->post_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->temp,f2->padding1_rows,f2->padding1_cols,f2->temp2);
            else
                group_normalization_back_propagation(f2->pre_activation,f2->n_kernels,f2->rows1,f2->cols1,f2->group_norm_channels,f2->group_norm_channels,f2->group_norm,f2->temp,f2->padding1_rows,f2->padding1_cols,f2->temp2);

            
            if(f2->activation_flag == SIGMOID){
                derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == RELU){
                derivative_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == TANH){
                derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == NO_ACTIVATION){
                copy_array(f2->temp2,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            
        }
        
        else{
            if(f2->activation_flag == SIGMOID){
                derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == RELU){
                derivative_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == TANH){
                derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
            
            if(f2->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->n_kernels*f2->rows1*f2->cols1);
                dot1D(f2->temp3,f2->temp,f2->temp,f2->n_kernels*f2->rows1*f2->cols1);
            }
        }
        
        /* computing the weight and bias derivatives for f2 applied to f1 output*/
        
        for(i = 0; i < f2->n_kernels; i++){
            if(f1->pooling_flag)
                convolutional_back_prop(f1->post_pooling, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            else if(f1->normalization_flag)
                convolutional_back_prop(f1->post_normalization, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            else if(f1->activation_flag)
                convolutional_back_prop(f1->post_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
            else
                convolutional_back_prop(f1->pre_activation, f2->kernels[i], f2->input_rows,f2->input_cols,f2->kernel_rows,f2->kernel_cols,f2->biases[i],f2->channels,&f2->temp[i*f2->rows1*f2->cols1],f2->error2,f2->d_kernels[i], &f2->d_biases[i], f2->stride1_rows, f2->padding1_rows);                
        }
        
        
        return f2->error2;
    }
    
    else
        return f2->temp;
    
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
    
    /*computing the backpropagation for f2*/
    if(f2->dropout_flag){
        dot1D(error,f2->dropout_mask,f2->temp,f2->output);
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_softmax_array(f2->pre_activation,f2->temp3,f2->post_activation,f2->temp,f2->output);
            copy_array(f2->temp3,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == LEAKY_RELU){
            derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,f2->temp,f2->temp,f2->output);
        }
    }
    
    else{
        if(f2->activation_flag == SIGMOID){
            derivative_sigmoid_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        else if(f2->activation_flag == RELU){
            derivative_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == SOFTMAX){
            derivative_softmax_array(f2->pre_activation,f2->temp,f2->post_activation,error,f2->output);
        }
        
        else if(f2->activation_flag == TANH){
            derivative_tanhh_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        
        else if(f2->activation_flag == LEAKY_RELU){
            derivative_leaky_relu_array(f2->pre_activation,f2->temp3,f2->output);
            dot1D(f2->temp3,error,f2->temp,f2->output);
        }
        
        else{
            copy_array(error,f2->temp,f2->output);
        }
    }
    /* computing the weight and bias derivatives for f2 applied to f1 output*/
        if(f1->pooling_flag)
            fully_connected_back_prop(f1->post_pooling, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
        else if(f1->normalization_flag)
            fully_connected_back_prop(f1->post_normalization, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
        else if(f1->activation_flag)
            fully_connected_back_prop(f1->post_activation, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
        else
            fully_connected_back_prop(f1->pre_activation, f2->temp, f2->weights,f2->error2, f2->d_weights,f2->d_biases, f2->input, f2->output);
    
    
    return f2->error2;
    
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
    temp->post_activation = input;
    temp->layer = -1;
    
    /* apply the feed forward to the model*/
    for(i = 0; i < m->layers; i++){
        for(j = 0; j < m->layers && m->sla[i][j] != 0; j++){
            
                
            if(!i){
                if(m->sla[i][j] == FCLS){
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    ff_cl_fcl(temp,m->fcls[k1]);
                    k1++;
                }
                
                else if(m->sla[i][j] == CLS){
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
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
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(k3-count == 0)
                        m->rls[z]->input = temp->post_activation;
                    
                    ff_cl_cl(temp,m->rls[z]->cls[k3-count]);
                    
                    if(k3-count == m->rls[z]->n_cl-1){
                        if(m->rls[z]->cls[k3-count]->pooling_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_pooling,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows2*m->rls[z]->cls[k3-count]->cols2);
                        
                        else if(m->rls[z]->cls[k3-count]->normalization_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_normalization,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        
                        else if(m->rls[z]->cls[k3-count]->activation_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        
                        else
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->pre_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        
                        if(m->rls[z]->cl_output->activation_flag == LEAKY_RELU)
                            leaky_relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == RELU)
                            relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == SIGMOID)
                            sigmoid_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == TANH)
                            tanhh_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                    }
                    
                    k3++;
                    
                    
                }
            }
            
            else{
                
                if(m->sla[i][j] == FCLS){
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
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
                    
                        
                        ff_cl_fcl(m->rls[z2]->cl_output,m->fcls[k1]);
                    }
                    
                    k1++;
                }
                
                else if(m->sla[i][j] == CLS){
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
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
                    
                        
                        ff_cl_cl(m->rls[z2]->cl_output,m->cls[k2]);
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
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    if(m->sla[i-1][0] == FCLS){
                        if(k3-count == 0){
                            if(m->fcls[k1-1]->dropout_flag){
                                if(m->fcls[k1-1]->activation_flag){
                                    dot1D(m->fcls[k1-1]->post_activation,m->fcls[k1-1]->dropout_mask,m->fcls[k1-1]->dropout_temp,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                    m->rls[z]->input = m->fcls[k1-1]->dropout_temp;
                                }
                                else{
                                    dot1D(m->fcls[k1-1]->pre_activation,m->fcls[k1-1]->dropout_mask,m->fcls[k1-1]->dropout_temp,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                    m->rls[z]->input = m->fcls[k1-1]->dropout_temp;
                                }
                            }
                            else{
                                if(m->fcls[k1-1]->activation_flag){
                                    m->rls[z]->input = m->fcls[k1-1]->post_activation;
                                }
                                else{
                                    m->rls[z]->input = m->fcls[k1-1]->pre_activation;
                                }
                            }
                        }
                    
                        ff_fcl_cl(m->fcls[k1-1],m->rls[z]->cls[k3-count]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        if(k3-count == 0){
                            if(m->cls[k2-1]->pooling_flag){
                                m->rls[z]->input = m->cls[k2-1]->post_pooling;
                            }
                            else if(m->cls[k2-1]->normalization_flag){
                                m->rls[z]->input = m->cls[k2-1]->post_normalization;
                            }
                            
                            else if(m->cls[k2-1]->activation_flag){
                                m->rls[z]->input = m->cls[k2-1]->post_activation;
                            }
                            else{
                                m->rls[z]->input = m->cls[k2-1]->pre_activation;
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
                            if(m->rls[z2]->cl_output->activation_flag)
                                m->rls[z]->input = m->rls[z2]->cl_output->post_activation;
                            else
                                m->rls[z]->input = m->rls[z2]->cl_output->pre_activation;
                        }
                        if(z2!=z){
                            ff_cl_cl(m->rls[z2]->cl_output,m->rls[z]->cls[k3-count]);

                        }
                        else{
                            ff_cl_cl(m->rls[z2]->cls[k3-1-count2],m->rls[z]->cls[k3-count]);
                        }
                    }
                    
                    if(k3-count == m->rls[z]->n_cl-1){
                        if(m->rls[z]->cls[k3-count]->pooling_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_pooling,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows2*m->rls[z]->cls[k3-count]->cols2);
                        else if(m->rls[z]->cls[k3-count]->normalization_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_normalization,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        else if(m->rls[z]->cls[k3-count]->activation_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->post_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        else
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k3-count]->pre_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k3-count]->n_kernels*m->rls[z]->cls[k3-count]->rows1*m->rls[z]->cls[k3-count]->cols1);
                        
                        if(m->rls[z]->cl_output->activation_flag == LEAKY_RELU)
                            leaky_relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == RELU)
                            relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == SIGMOID)
                            sigmoid_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == TANH)
                            tanhh_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);

                    }
                    
                    k3++;
                    
                    
                }
                
            }
            
        }
    }
    
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
    temp->post_activation = input;
    
    float* error1 = error;
         
    float* error_residual = NULL;    
    /* apply the backpropagation to the model*/
    for(i = m->layers-1; i >= 0; i--){
        for(j = 0; j < 1 && m->sla[i][j] != 0; j++){
            
            
            if(!i){
                if(m->sla[i][j] == FCLS){
                    k1--;
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    error1 = bp_cl_fcl(temp,m->fcls[k1],error1);
                    
                }
                
                else if(m->sla[i][j] == CLS){
                    k2--;
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    error1 = bp_cl_cl(temp,m->cls[k2],error1);
                    
                    
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
                        error_residual = error1; 
                    }
                    
                    if(m->rls[z]->cls[k3-count]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    
                    error1 = bp_cl_cl(temp,m->rls[z]->cls[k3-count],error1);
                    
                    
                    if(k3-count == 0)
                        sum1D(error1,error_residual,error1,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    
                    
                    
                }
            }
            
            else{
                
                if(m->sla[i][j] == FCLS){
                    k1--;
                    if(m->fcls[k1]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        error1 = bp_fcl_fcl(m->fcls[k1-1],m->fcls[k1],error1);
                        }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error1 = bp_cl_fcl(m->cls[k2-1],m->fcls[k1], error1);
                        }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        error1 = bp_cl_fcl(m->rls[z2]->cl_output,m->fcls[k1],error1);
                        if(m->rls[z2]->cl_output->activation_flag == LEAKY_RELU)
                            derivative_leaky_relu_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else if(m->rls[z2]->cl_output->activation_flag == RELU)
                            derivative_relu_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else if(m->rls[z2]->cl_output->activation_flag == SIGMOID)
                            derivative_sigmoid_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else if(m->rls[z2]->cl_output->activation_flag == TANH)
                            derivative_tanhh_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        
                        if(m->rls[z2]->cl_output->activation_flag != NO_ACTIVATION)
                            dot1D(m->rls[z2]->cl_output->temp3,error1,m->rls[z2]->cl_output->temp,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else
                            copy_array(error1,m->rls[z2]->cl_output->temp,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            
                        error1 = m->rls[z2]->cl_output->temp;
                        
                    }
                    
                    
                }
                
                else if(m->sla[i][j] == CLS){
                    k2--;
                    if(m->cls[k2]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        error1 = bp_fcl_cl(m->fcls[k1-1],m->cls[k2],error1);
                        
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error1 = bp_cl_cl(m->cls[k2-1],m->cls[k2],error1);
                        
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        error1 = bp_cl_cl(m->rls[z2]->cl_output,m->cls[k2],error1);
                        if(m->rls[z2]->cl_output->activation_flag == LEAKY_RELU)
                            derivative_leaky_relu_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else if(m->rls[z2]->cl_output->activation_flag == RELU)
                            derivative_relu_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else if(m->rls[z2]->cl_output->activation_flag == SIGMOID)
                            derivative_sigmoid_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else if(m->rls[z2]->cl_output->activation_flag == TANH)
                            derivative_tanhh_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        if(m->rls[z2]->cl_output->activation_flag != NO_ACTIVATION)
                            dot1D(m->rls[z2]->cl_output->temp3,error1,m->rls[z2]->cl_output->temp,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        else
                            copy_array(error1,m->rls[z2]->cl_output->temp,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                        error1 = m->rls[z2]->cl_output->temp;
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
                        
                        error_residual = error1;                    
                        
                    }
                    
                    if(m->rls[z]->cls[k3-count]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    if(m->sla[i-1][0] == FCLS){
                        
                    
                        error1 = bp_fcl_cl(m->fcls[k1-1],m->rls[z]->cls[k3-count],error1);
                       
                        
                        
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error1 = bp_cl_cl(m->cls[k2-1],m->rls[z]->cls[k3-count],error1);
                       
                        
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k3-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                        if(z2 == z)
                            error1 = bp_cl_cl(m->rls[z2]->cls[k3-1-count2],m->rls[z]->cls[k3-count],error1);
                        else{
                            error1 = bp_cl_cl(m->rls[z2]->cl_output,m->rls[z]->cls[k3-count],error1);
                            if(m->rls[z2]->cl_output->activation_flag == LEAKY_RELU)
                            derivative_leaky_relu_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            else if(m->rls[z2]->cl_output->activation_flag == RELU)
                                derivative_relu_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            else if(m->rls[z2]->cl_output->activation_flag == SIGMOID)
                                derivative_sigmoid_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            else if(m->rls[z2]->cl_output->activation_flag == TANH)
                                derivative_tanhh_array(m->rls[z2]->cl_output->pre_activation,m->rls[z2]->cl_output->temp3,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            if(m->rls[z2]->cl_output->activation_flag != NO_ACTIVATION)
                                dot1D(m->rls[z2]->cl_output->temp3,error1,m->rls[z2]->cl_output->temp,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            else
                                copy_array(error1,m->rls[z2]->cl_output->temp,m->rls[z2]->cl_output->n_kernels*m->rls[z2]->cl_output->rows1*m->rls[z2]->cl_output->cols1);
                            error1 = m->rls[z2]->cl_output->temp;
                        }
                        
                        
                    }
                    
                    
                    
                    if(k3-count == 0)
                        sum1D(error1,error_residual,error1,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    
                    
                    
                }
                
            }
            
        }
    }

    free(temp);
    if(!bool_is_real(error1[0])){
        fprintf(stderr,"Error: nan occurred, probably due to the exploiting gradient problem, or you just found a perfect function that match your data and you should not keep training\n");
        exit(1);
    }
    return error1;
}

/* This function returs the total number of weights in the model m
 * 
 * Input
 * 
 *             @ model* m:= the model
 * 
 * */
int count_weights(model* m){
    int i,j;
    int sum = 0;
    for(i = 0; i < m->n_fcl; i++){
        sum+=m->fcls[i]->input*m->fcls[i]->output;
    }
    
    for(i = 0; i < m->n_cl; i++){
        if(m->cls[i]->convolutional_flag == CONVOLUTION){
            sum+=m->cls[i]->n_kernels*m->cls[i]->channels*m->cls[i]->kernel_rows*m->cls[i]->kernel_cols;
            if(m->cls[i]->normalization_flag == GROUP_NORMALIZATION){
                sum+=m->cls[i]->n_kernels/m->cls[i]->group_norm_channels*m->cls[i]->group_norm[0]->vector_dim;
            }
        }
    }
    
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            sum+=m->rls[i]->cls[j]->n_kernels*m->rls[i]->cls[j]->channels*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols;
            if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                sum+=m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels*m->rls[i]->cls[j]->group_norm[0]->vector_dim;
            }
        }
    }
    
    return sum;
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
 *                @ float* b1:= the hyper parameter b1 of adam algorithm
 *                @ float* b2:= the hyper parameter b2 of adam algorithm
 *                @ int regularization:= NO_REGULARIZATION or L2 (0,1)
 *                @ int total_number_weights:= the number of total weights of the network (for l2 regularization)
 *                @ float lambda:= a float value for l2 regularization
 *                @ unsigned long long int* t:= the number of time that radam has been used
 * */
void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t){
    if(m == NULL)
        return;
    
    lambda*=(float)mini_batch_size;
    
    if(regularization == L2_REGULARIZATION){
        add_l2_residual_layer(m,total_number_weights,lambda);
        add_l2_convolutional_layer(m,total_number_weights,lambda);
        add_l2_fully_connected_layer(m,total_number_weights,lambda);
    }
    
    
    if(gradient_descent_flag == NESTEROV){    
        update_residual_layer_nesterov(m,lr,momentum,mini_batch_size);
        update_convolutional_layer_nesterov(m,lr,momentum,mini_batch_size);
        update_fully_connected_layer_nesterov(m,lr,momentum,mini_batch_size);
    }
    
    else if(gradient_descent_flag == ADAM){
        update_residual_layer_adam(m,lr,mini_batch_size, (*b1), (*b2));
        update_convolutional_layer_adam(m,lr,mini_batch_size, (*b1), (*b2));
        update_fully_connected_layer_adam(m,lr,mini_batch_size, (*b1), (*b2));
        (*b1)*=m->beta1_adam;
        (*b2)*=m->beta2_adam;
    }
    
    else if(gradient_descent_flag == RADAM){
        update_residual_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t);
        update_convolutional_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t);
        update_fully_connected_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t);
        (*b1)*=m->beta1_adam;
        (*b2)*=m->beta2_adam;
        (*t)++;
    }     
    

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
        fprintf(stderr,"Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    sum_fully_connected_layers_partial_derivatives(m,m2,m3);
    sum_convolutional_layers_partial_derivatives(m,m2,m3);
    sum_residual_layers_partial_derivatives(m,m2,m3);
}


/* this function gives the number of float params for biases and weights in a model
 * 
 * Input:
 * 
 * 
 *                 @ rl* f:= the residual layer
 * */
int get_array_size_params_model(model* f){
    int sum = 0,i;
    for(i = 0; i < f->n_fcl; i++){
        sum+=get_array_size_params(f->fcls[i]);
    }
    
    for(i = 0; i < f->n_cl; i++){
        sum+=get_array_size_params_cl(f->cls[i]);
    }
    
    for(i = 0; i < f->n_rl; i++){
        sum+=get_array_size_params_rl(f->rls[i]);
    }
    
    return sum;
}

/* this function paste the weights and biases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ model* f:= the model
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_params_model(model* f, float* vector){
    int sum = 0,i;
    for(i = 0; i < f->n_fcl; i++){
        memcopy_vector_to_params(f->fcls[i],&vector[sum]);
        sum += get_array_size_params(f->fcls[i]);
    }
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_params_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
    for(i = 0; i < f->n_rl; i++){
        memcopy_vector_to_params_rl(f->rls[i],&vector[sum]);
        sum += get_array_size_params_rl(f->rls[i]);
    }
}


/* this function paste the vector in the weights and biases of the model
 * 
 * Inputs:
 * 
 * 
 *                 @ model* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_params_to_vector_model(model* f, float* vector){
    int sum = 0,i;
    for(i = 0; i < f->n_fcl; i++){
        memcopy_params_to_vector(f->fcls[i],&vector[sum]);
        sum += get_array_size_params(f->fcls[i]);
    }
    for(i = 0; i < f->n_cl; i++){
        memcopy_params_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
    for(i = 0; i < f->n_rl; i++){
        memcopy_params_to_vector_rl(f->rls[i],&vector[sum]);
        sum += get_array_size_params_rl(f->rls[i]);
    }
}

/* this function paste the dweights and dbiases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ model* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_derivative_params_model(model* f, float* vector){
    int sum = 0,i;
    for(i = 0; i < f->n_fcl; i++){
        memcopy_vector_to_derivative_params(f->fcls[i],&vector[sum]);
        sum += get_array_size_params(f->fcls[i]);
    }
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_derivative_params_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
    for(i = 0; i < f->n_rl; i++){
        memcopy_vector_to_derivative_params_rl(f->rls[i],&vector[sum]);
        sum += get_array_size_params_rl(f->rls[i]);
    }
}


/* this function paste the vector in the dweights and dbiases of the model
 * 
 * Inputs:
 * 
 * 
 *                 @ model* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_derivative_params_to_vector_model(model* f, float* vector){
    int sum = 0,i;
    for(i = 0; i < f->n_fcl; i++){
        memcopy_derivative_params_to_vector(f->fcls[i],&vector[sum]);
        sum += get_array_size_params(f->fcls[i]);
    }
    for(i = 0; i < f->n_cl; i++){
        memcopy_derivative_params_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
    for(i = 0; i < f->n_rl; i++){
        memcopy_derivative_params_to_vector_rl(f->rls[i],&vector[sum]);
        sum += get_array_size_params_rl(f->rls[i]);
    }
}

/* Setting the params for the loss computation
 * 
 * Inputs:
 *             @ model* m:= the model
 *             @ int error_flag:= the loss computed
 *             @ float threshold1:= the threshold for some errors see math_functions.c (huber loss)
 *             @ float threshold2:= the threshold for some errors see math_functions.c (huber loss)
 *             @ float gamma:= the gamma factor for focal loss
 *             @ float* alpha:= the param for unbalanced dataset (dimension: output_dimension)
 *             @ int output_dimension:= the error dimension
 * */
 
 
void set_model_error(model* m, int error_flag, float threshold1, float threshold2, float gamma, float* alpha, int output_dimension){
    m->error_flag = error_flag;
    m->error_threshold1 = threshold1;
    m->error_threshold2 = threshold2;
    m->error_gamma = gamma;
    if(m->error != NULL)
    free(m->error);
    m->error = (float*)calloc(output_dimension,sizeof(float));
    if(alpha != NULL){
        if(m->error_alpha != NULL)
            free(m->error_alpha);
        m->error_alpha = (float*)malloc(sizeof(float)*output_dimension);
        copy_array(alpha,m->error_alpha,output_dimension);
    }
    m->output_dimension = output_dimension;
}

/* Given a model and an output, this function computes the mse derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void mse_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in mse model error something is null\n");
        exit(1);
    }
    
    derivative_mse_array(m->output_layer,output,m->error,m->output_dimension);     
}

/* Given a model and an output, this function computes the cross entropy derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void cross_entropy_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in cross entropy model error something is null\n");
        exit(1);
    }
    
    derivative_cross_entropy_array(m->output_layer,output,m->error,m->output_dimension);     
}

/* Given a model and an output, this function computes the focal loss derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void focal_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in focal model error something is null\n");
        exit(1);
    }
    
    derivative_focal_loss_array(m->output_layer,output,m->error,m->error_gamma,m->output_dimension);     
}

/* Given a model and an output, this function computes the huber loss derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void huber_one_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in huber one model error something is null\n");
        exit(1);
    }
    
    derivative_huber_loss_array(m->output_layer,output,m->error,m->error_threshold1,m->output_dimension);     
}

/* Given a model and an output, this function computes the modified huber loss derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void huber_two_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in huber two model error something is null\n");
        exit(1);
    }
    
    derivative_modified_huber_loss_array(m->output_layer,output,m->error_threshold1,m->error,m->error_threshold2,m->output_dimension);     
}

/* Given a model and an output, this function computes the kl divergence derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void kl_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in kl model error something is null\n");
        exit(1);
    }
    
    derivative_kl_divergence(m->output_layer,output,m->error,m->output_dimension);     
}

/* Given a model and an output, this function computes the entropy derivative in m->error_vector
 * 
 * Inputs:
 * 
 *             @ model* m:= the model
 *             @ float* output:= the output (dimension:m->output_dimension)
 * 
 * */
void entropy_model_error(model* m, float* output){
    if(m == NULL || m->error == NULL || output == NULL){
        fprintf(stderr,"Error: in entropy model error something is null\n");
        exit(1);
    }
    
    derivative_entropy_array(m->output_layer,m->error,m->output_dimension);     
}


/* compute the error of a model
 * 
 * Inputs:
 * 
 *             @ model* m:= model
 *             @ float* output:= the output
 * */
void compute_model_error(model* m, float* output){
    if(m->error_flag == NO_LOSS)
        copy_array(output,m->error,m->output_dimension);
    else if(m->error_flag == MSE_LOSS)
        mse_model_error(m,output);
    else if(m->error_flag == CROSS_ENTROPY_LOSS)
        cross_entropy_model_error(m,output);
    else if(m->error_flag == FOCAL_LOSS)
        focal_model_error(m,output);
    else if(m->error_flag == HUBER1_LOSS)
        huber_one_model_error(m,output);
    else if(m->error_flag == HUBER2_LOSS)
        huber_two_model_error(m,output);
    else if(m->error_flag == KL_DIVERGENCE_LOSS)
        kl_model_error(m,output);
    else if(m->error_flag == ENTROPY_LOSS)
        entropy_model_error(m,output);            
}


/* computing the feed forward, the error and the back propagation of a model given an input and output
 * 
 * Input:
 *             
 *             @ model* m:= the model with the layers
 *             @ int tensor_depth:= the depth of the input tensor
 *             @ int tensor_i:= the number of rows of the tensor
 *             @ int tensor_j:= the number of columns of the tensor
 *             @ float* input:= your input array
 *             @ float* output:= your output array
 * 
 * */
float* ff_error_bp_model_once(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* output){
    model_tensor_input_ff(m,tensor_depth,tensor_i,tensor_j,input);
    compute_model_error(m,output);
    if(m->error_alpha != NULL)
        dot1D(m->error_alpha,output,output,m->output_dimension);
    return model_tensor_input_bp(m,tensor_depth,tensor_i,tensor_j,input, m->error,m->output_dimension);
}

/*sum partial derivatives of batch sizes in 1 unique model
 * 
 * input:
 * 
 *             @ model* sum_m:= where are summed up the partial derivatives
 *             @ model** models:= the models (dimension: n_models)
 *             @ int n_models:= the number of models
 * 
 * */
void sum_models_partial_derivatives(model* sum_m, model** models, int n_models){
    int i;
    for(i = 0; i < n_models; i++){
        sum_model_partial_derivatives(models[i],sum_m,sum_m);
    }
}
