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

/* This function builds a bmodel* structure which can be used to train the network
 * 
 * Input:
 *             
 *             @ int layers:= number of total layers, this means that if you have 2 layers with the same layer id 
 *                            then layers = 2. For example if you have 2 fully-connected layers with same layer id = 0
 *                            then layers param must be set to 2. if you have 3 layers, 2 with same layer id and 1 with another
 *                            layer id, then layers = 3 and so on
 *             @ int n_rl:= same as layers but only for residual layers
 *             @ int n_cl:= same as layer but only for convolutional layers. (the convolutional layers inside residual layer must not be count)
 *             @ int n_fcl:= same as layer, but only for fully-connected layers
 *             @ int n_bnl:= same as layer, but only for batch normalized layers
 *             @ rl** rls:= your residual layers
 *             @ cl** cls:= your convolutional layers
 *             @ fcl** fcls:= your fully-connected layers
 *             @ bn** bns:= your batch normalized layers
 * 
 * */
bmodel* batch_network(int layers, int n_rl, int n_cl, int n_fcl, int n_bnl, rl** rls, cl** cls, fcl** fcls, bn** bnls){
    if(!layers || (!n_rl && !n_cl && !n_fcl && !n_bnl) || (!n_rl && rls != NULL) || (!n_cl && cls!= NULL) || (!n_fcl && fcls != NULL) || (!n_bnl && bnls != NULL)){
        fprintf(stderr,"Error: layers must be > 0 and at least one between n_rl, n_cl, n_fcl, n_bnl must be > 0\n");
        exit(1); 
    }
    
    int i,j,k, position, count, k1,k2,k3;
    
    
    /*checking if the residual layer has the right size from the input to the output*/
    for(i = 0; i < n_rl; i++){
        if(rls[i]->cls[rls[i]->n_cl-1]->pooling_flag){
            if(rls[i]->cls[rls[i]->n_cl-1]->n_kernels*rls[i]->cls[rls[i]->n_cl-1]->rows2*rls[i]->cls[rls[i]->n_cl-1]->cols2 != rls[i]->channels*rls[i]->input_rows*rls[i]->input_cols){
                fprintf(stderr,"Error: you have a residual layer where the input size doesn't correspond to the last convolutional layer size of the residual layer\n");
                exit(1);
            }
        }
        
        else{
            if(rls[i]->cls[rls[i]->n_cl-1]->n_kernels*rls[i]->cls[rls[i]->n_cl-1]->rows1*rls[i]->cls[rls[i]->n_cl-1]->cols1 != rls[i]->channels*rls[i]->input_rows*rls[i]->input_cols){
                fprintf(stderr,"Error: you have a residual layer where the input size doesn't correspond to the last convolutional layer size of the residual layer\n");
                exit(1);
            }
        }
    }
    
    cl* temp = NULL;
    fcl* temp2 = NULL;
    rl* temp3 = NULL;
    bn* temp4 = NULL;
    int** sla = (int**)malloc(sizeof(int*)*layers);
    for(i = 0; i < layers; i++){
        sla[i] = (int*)calloc(layers,sizeof(int));
    }
    
    bmodel* m = (bmodel*)malloc(sizeof(bmodel));
    
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
        /*there is no need to check if conv layers inside residual layer are sequential 'cause can there be batch normalized layer inside*/
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
    
    /* sorting batch-normalized layers*/
    for(i = 0; i < n_bnl; i++){
        j = 0;
        temp4 = bnls[j];
        position = j;
        
        for(k = 1; k < n_bnl; k++){
            if(bnls[position]->layer > bnls[k]->layer){
                bnls[position] = bnls[k];
                bnls[k] = temp4;
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
        
        for(j = 0; j < n_bnl; j++){
            if(bnls[j]->layer == i){
                sla[i][k] = BNS;
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
    m->n_bn = n_bnl;
    m->sla = sla;
    m->rls = rls;
    m->cls = cls;
    m->fcls = fcls;
    m->bns = bnls;
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
        m->output_layer_bn_training_mode = NULL;
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
        m->output_layer_bn_training_mode = NULL;
    }
    
    else if(sla[i][0] == BNS)
        m->output_layer_bn_training_mode = m->bns[m->n_bn-1]->outputs;
    
    
    else{
        if(m->rls[m->n_rl-1]->cl_output->activation_flag)
            m->output_layer = m->rls[m->n_rl-1]->cl_output->post_activation;
        else
            m->output_layer = m->rls[m->n_rl-1]->cl_output->pre_activation;
        m->output_layer_bn_training_mode = NULL;
    }
    
    return m;
}

/* This function frees the space allocated by a bmodel structure
 * 
 * Input:
 *             @ bmodel* m:= the structure
 * 
 * */
void free_bmodel(bmodel* m){
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
    
    for(i = 0; i < m->n_bn; i++){
        free_batch_normalization(m->bns[i]);
    }
    free(m->bns);
    for(i = 0; i < m->layers; i++){
        free(m->sla[i]);
    }
    free(m->sla);
    free(m);
}


/* This function copies a model using the copy function for the layers
 * see layers.c files
 * 
 * Input:
 *         
 *             @ bmodel* m:= the bmodel that must be copied
 * 
 * */
bmodel* copy_bmodel(bmodel* m){
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
    
    bn** bns = NULL;
    if(m->bns!=NULL)
        bns = (bn**)malloc(sizeof(bn*)*m->n_bn);
        
    for(i = 0; i < m->n_fcl; i++){
        fcls[i] = copy_fcl(m->fcls[i]);
    }
    for(i = 0; i < m->n_cl; i++){
        cls[i] = copy_cl(m->cls[i]);
    }
    for(i = 0; i < m->n_rl; i++){
        rls[i] = copy_rl(m->rls[i]);
    }
    
    for(i = 0; i < m->n_bn; i++){
        bns[i] = copy_bn(m->bns[i]);
    }
    bmodel* copy = batch_network(m->layers, m->n_rl, m->n_cl, m->n_fcl,m->n_bn, rls, cls, fcls, bns);
    return copy;
}



/* This function copies a bmodel using the paste function for the layers
 * see layers.c files (are copied biases and weights)
 * 
 * Input:
 *         
 *             @ bmodel* m:= the model that must be copied
 *             @ bmodel* copy:= the model where m is copied
 * 
 * */
void paste_bmodel(bmodel* m,bmodel* copy){
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
    for(i = 0; i < m->n_bn; i++){
        paste_bn(m->bns[i],copy->bns[i]);
    }
    return;
}

/* This function copies a bmodel with the rule: teta_i:= teta_j*tau +(1-tau)*teta_i for biases and weights
 * 
 * Input:
 *         
 *             @ bmodel* m:= the model that must be copied
 *             @ bmodel* copy:= the model where m is copied
 *             @ float tau:= the tau param
 * 
 * */
void slow_paste_bmodel(bmodel* m, bmodel* copy, float tau){
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
    
    for(i = 0; i < m->n_rl; i++){
        slow_paste_bn(m->bns[i],copy->bns[i],tau);
    }
    return;
}
/* This function resets a bmodel using the copy bmodel function
 * returns a bmodel equal to the one as input but with all resetted except for weights and biases
 * 
 * Input:
 *             
 *             @ bmodel* m:= the bmodel m that must be resetted
 * */
bmodel* reset_bmodel(bmodel* m){
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
    
    for(i = 0; i < m->n_bn; i++){
        reset_bn(m->bns[i]);
    }
    return m;
}


/* this function returns the space allocated by the arrays of m
 * 
 * Input:
 * 
 *             bmodel* m:= the structure bmodel
 * 
 * */
unsigned long long int size_of_bmodel(bmodel* m){
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
    
    for(i = 0; i < m->n_bn; i++){
        sum+= size_of_bn(m->bns[i]);
    }
    
    sum+= (( unsigned long long int)(m->layers*m->layers*sizeof(int)));
    return sum;
}




/* This function saves a bmodel(network) on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ bmodel* m:= the actual network that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_bmodel(bmodel* m, int n){
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
        fprintf(stderr,"Error: an error occurred saving the bmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->n_rl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the bmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->n_cl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the bmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->n_fcl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the bmodel\n");
        exit(1);
    }
    
    i = fwrite(&m->n_bn,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the bmodel\n");
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
    
    for(i = 0; i < m->n_bn; i++){
        save_bn(m->bns[i],n);
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
bmodel* load_bmodel(char* file){
    if(file == NULL)
        return NULL;
    int i;
    FILE* fr = fopen(file,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",file);
        exit(1);
    }
    
    int layers = 0,n_cl = 0,n_rl = 0,n_fcl = 0,n_bn;
    
    i = fread(&layers,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the bmodel\n");
        exit(1);
    }
    
    i = fread(&n_rl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the bmodel\n");
        exit(1);
    }
    
    i = fread(&n_cl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the bmodel\n");
        exit(1);
    }
    
    i = fread(&n_fcl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the bmodel\n");
        exit(1);
    }
    
    i = fread(&n_bn,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the bmodel\n");
        exit(1);
    }
    

    rl** rls;
    cl** cls;
    fcl** fcls;
    bn** bns;
    
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
    if(!n_bn)
        bns = NULL;
    else
        bns = (bn**)malloc(sizeof(bn*)*n_bn);
    
    for(i = 0; i < n_rl; i++){
        rls[i] = load_rl(fr);
    }
    
    for(i = 0; i < n_cl; i++){
        cls[i] = load_cl(fr);
    }
    
    for(i = 0; i < n_fcl; i++){
        fcls[i] = load_fcl(fr);
    }
    
    for(i = 0; i < n_bn; i++){
        bns[i] = load_bn(fr);
    }
    
    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file);
        exit(1);
    }
    
    bmodel* m = batch_network(layers,n_rl,n_cl,n_fcl,n_bn,rls,cls,fcls,bns);
    
    return m;
    
}


/* This function returns the total number of weights in the bmodel m
 * 
 * Input
 * 
 *             @ bmodel* m:= the bmodel
 * 
 * */
int count_bmodel_weights(bmodel* m){
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
    
    for(i = 0; i < m->n_bn; i++){
        sum+=m->bns[i]->vector_dim;
    }
    
    return sum;
}


/* This function can update the bmodel of the network using the adam algorithm, radam algorithm or the nesterov momentum
 * 
 * Input:
 * 
 *             @ bmodel* m:= the bmodel that must be update
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *             @ int mini_batch_size:= the batch used
 *             @ int gradient_descent_flag:= NESTEROV or ADAM (1,2)
 *                @ float* b1:= the hyper parameter b1 of adam algorithm
 *                @ float* b2:= the hyper parameter b2 of adam algorithm
 *                @ int regularization:= NO_REGULARIZATION or L2 (0,1)
 *                @ int total_number_weights:= the number of total weights of the network (for l2 regularization)
 *                @ float lambda:= a float value for l2 regularization
 *                 @ unsigned long long int *t:= the t param used by the radam algorithm
 * 
 * */
void update_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t){
    if(m == NULL)
        return;
    
    lambda*=mini_batch_size;
    
    if(regularization == L2_REGULARIZATION){
        // l2 regularization with batch normalization doesn't make sense! i can't add it to batch norm layers, so in the case
        // i presume that you are using this model just to keep all togheter and you want to merge this bmodel to another one
        // later, i hope that you didn't used batch norm layers in this case, otherwise is a mess
        add_l2_residual_layer_bmodel(m,total_number_weights,lambda);
        add_l2_convolutional_layer_bmodel(m,total_number_weights,lambda);
        add_l2_fully_connected_layer_bmodel(m,total_number_weights,lambda);
    }
    
    
    if(gradient_descent_flag == NESTEROV){    
        update_residual_layer_nesterov_bmodel(m,lr,momentum,mini_batch_size);
        update_convolutional_layer_nesterov_bmodel(m,lr,momentum,mini_batch_size);
        update_fully_connected_layer_nesterov_bmodel(m,lr,momentum,mini_batch_size);
        update_batch_normalized_layer_nesterov_bmodel(m,lr,momentum,mini_batch_size);
    }
    
    else if(gradient_descent_flag == ADAM){
        update_residual_layer_adam_bmodel(m,lr,mini_batch_size, (*b1), (*b2));
        update_convolutional_layer_adam_bmodel(m,lr,mini_batch_size, (*b1), (*b2));
        update_fully_connected_layer_adam_bmodel(m,lr,mini_batch_size, (*b1), (*b2));
        update_batch_normalized_layer_adam_bmodel(m,lr,mini_batch_size, (*b1), (*b2));
        (*b1)*=m->beta1_adam;
        (*b2)*=m->beta2_adam;
    } 
    
    else if(gradient_descent_flag == RADAM){
        update_residual_layer_radam_bmodel(m,lr,mini_batch_size, (*b1), (*b2), *t);
        update_convolutional_layer_radam_bmodel(m,lr,mini_batch_size, (*b1), (*b2), *t);
        update_fully_connected_layer_radam_bmodel(m,lr,mini_batch_size, (*b1), (*b2), *t);
        update_batch_normalized_layer_radam_bmodel(m,lr,mini_batch_size,(*b1),(*b2),*t);
        (*b1)*=m->beta1_adam;
        (*b2)*=m->beta2_adam;
        (*t)++;
    } 
    

}

/* This function sums the partial derivatives in bmodel m1 and m2 in m3
 * 
 * Input:
 *     
 *             @ bmodel* m:= first input model
 *             @ bmodel* m2:= second input model
 *             @ bmodel* m3:= output model
 * 
 * */
void sum_model_partial_derivatives_bmodel(bmodel* m, bmodel* m2, bmodel* m3){
    if(m == NULL || m2 == NULL || m3 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_model_partial_derivatives\n");
        exit(1);
    }
    sum_fully_connected_layers_partial_derivatives_bmodel(m,m2,m3);
    sum_convolutional_layers_partial_derivatives_bmodel(m,m2,m3);
    sum_residual_layers_partial_derivatives_bmodel(m,m2,m3);
}

/* This function computes the feed-forward for a bmodel m. each layer at the index l makes the feed-forward
 * for the first layer at the index l-1. if the input is a 1d array then you should split its dimension
 * in 3 dimension to turn the input in a tensor, for example:
 * I have an input array of length 59, then i can split this in 3 dimensions: depth = 1, row = 1, cols = 59
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model with the layers
 *             @ int tensor_depth:= the depth of the input tensor
 *             @ int tensor_i:= the number of rows of the tensor
 *             @ int tensor_j:= the number of columns of the tensor
 *             @ float* input:= your input array
 *             @ int* k1k2k3k4:= is an array of input of dimension = 4, where k1 keeps the fully connected layers that have been reached
 *                               k2 the convolutional layers, k3 the residual layers and k4 the batch normalized layers(size : 4)
 * 
 * */
int bmodel_tensor_input_ff(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, int* k1k2k3k4){
    if(m == NULL)
        return;
    int i,j,z,w,count,count2,z2, layers = k1k2k3k4[0]+k1k2k3k4[1]+k1k2k3k4[2]+k1k2k3k4[3];
    
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
    for(i = layers; i < m->layers; i++){
        for(j = 0; j < m->layers && m->sla[i][j] != 0; j++){
            
                
            if(!i){
                if(m->sla[i][j] == FCLS){
                    if(m->fcls[k1k2k3k4[0]]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    ff_cl_fcl(temp,m->fcls[k1k2k3k4[0]]);
                    k1k2k3k4[0]++;
                }
                
                else if(m->sla[i][j] == CLS){
                    if(m->cls[k1k2k3k4[1]]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    ff_cl_cl(temp,m->cls[k1k2k3k4[1]]);
                    k1k2k3k4[1]++;
                }
                
                else if(m->sla[i][j] == RLS){
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k1k2k3k4[2]; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                
                    
                    if(m->rls[z]->cls[k1k2k3k4[2]-count]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(k1k2k3k4[2]-count == 0)
                        m->rls[z]->input = temp->post_activation;
                    
                    ff_cl_cl(temp,m->rls[z]->cls[k1k2k3k4[2]-count]);
                    
                    if(k1k2k3k4[2]-count == m->rls[z]->n_cl-1){
                        if(m->rls[z]->cls[k1k2k3k4[2]-count]->pooling_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->post_pooling,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows2*m->rls[z]->cls[k1k2k3k4[2]-count]->cols2);
                        
                        else if(m->rls[z]->cls[k1k2k3k4[2]-count]->normalization_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->post_normalization,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows1*m->rls[z]->cls[k1k2k3k4[2]-count]->cols1);
                        
                        else if(m->rls[z]->cls[k1k2k3k4[2]-count]->activation_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->post_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows1*m->rls[z]->cls[k1k2k3k4[2]-count]->cols1);
                        
                        else
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->pre_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows1*m->rls[z]->cls[k1k2k3k4[2]-count]->cols1);
                        
                        if(m->rls[z]->cl_output->activation_flag == LEAKY_RELU)
                            leaky_relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == RELU)
                            relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == SIGMOID)
                            sigmoid_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == TANH)
                            tanhh_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                    }
                    
                    k1k2k3k4[2]++;
                    
                    
                }
                
                else if(m->sla[i][j] == BNS){
                    k1k2k3k4[3]++;
                    free(temp);
                    return i;
                }
            }
            
            else{
                
                if(m->sla[i][j] == FCLS){
                    if(m->fcls[k1k2k3k4[0]]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        ff_fcl_fcl(m->fcls[k1k2k3k4[0]-1],m->fcls[k1k2k3k4[0]]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        ff_cl_fcl(m->cls[k1k2k3k4[1]-1],m->fcls[k1k2k3k4[0]]);
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k1k2k3k4[2]-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        ff_cl_fcl(m->rls[z2]->cl_output,m->fcls[k1k2k3k4[0]]);
                    }
                    
                    k1k2k3k4[0]++;
                }
                
                else if(m->sla[i][j] == CLS){
                    if(m->cls[k1k2k3k4[1]]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        ff_fcl_cl(m->fcls[k1k2k3k4[0]-1],m->cls[k1k2k3k4[1]]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        ff_cl_cl(m->cls[k1k2k3k4[1]-1],m->cls[k1k2k3k4[1]]);
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k1k2k3k4[2]-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        ff_cl_cl(m->rls[z2]->cl_output,m->cls[k1k2k3k4[1]]);
                    }
                    k1k2k3k4[1]++;
                }
                
                else if(m->sla[i][j] == RLS){
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k1k2k3k4[2]; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                
                    
                    if(m->rls[z]->cls[k1k2k3k4[2]-count]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    if(m->sla[i-1][0] == FCLS){
                        if(k1k2k3k4[2]-count == 0){
                            if(m->fcls[k1k2k3k4[0]-1]->dropout_flag){
                                if(m->fcls[k1k2k3k4[0]-1]->activation_flag){
                                    dot1D(m->fcls[k1k2k3k4[0]-1]->post_activation,m->fcls[k1k2k3k4[0]-1]->dropout_mask,m->fcls[k1k2k3k4[0]-1]->dropout_temp,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                    m->rls[z]->input = m->fcls[k1k2k3k4[0]-1]->dropout_temp;
                                }
                                else{
                                    dot1D(m->fcls[k1k2k3k4[0]-1]->pre_activation,m->fcls[k1k2k3k4[0]-1]->dropout_mask,m->fcls[k1k2k3k4[0]-1]->dropout_temp,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                                    m->rls[z]->input = m->fcls[k1k2k3k4[0]-1]->dropout_temp;
                                }
                            }
                            else{
                                if(m->fcls[k1k2k3k4[0]-1]->activation_flag){
                                    m->rls[z]->input = m->fcls[k1k2k3k4[0]-1]->post_activation;
                                }
                                else{
                                    m->rls[z]->input = m->fcls[k1k2k3k4[0]-1]->pre_activation;
                                }
                            }
                        }
                    
                        ff_fcl_cl(m->fcls[k1k2k3k4[0]-1],m->rls[z]->cls[k1k2k3k4[2]-count]);
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        if(k1k2k3k4[2]-count == 0){
                            if(m->cls[k1k2k3k4[1]-1]->pooling_flag){
                                m->rls[z]->input = m->cls[k1k2k3k4[1]-1]->post_pooling;
                            }
                            else if(m->cls[k1k2k3k4[1]-1]->normalization_flag){
                                m->rls[z]->input = m->cls[k1k2k3k4[1]-1]->post_normalization;
                            }
                            
                            else if(m->cls[k1k2k3k4[1]-1]->activation_flag){
                                m->rls[z]->input = m->cls[k1k2k3k4[1]-1]->post_activation;
                            }
                            else{
                                m->rls[z]->input = m->cls[k1k2k3k4[1]-1]->pre_activation;
                            }
                        }
                        ff_cl_cl(m->cls[k1k2k3k4[1]-1],m->rls[z]->cls[k1k2k3k4[2]-count]);
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k1k2k3k4[2]-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                        
                        if(k1k2k3k4[2]-count == 0){
                            if(m->rls[z2]->cl_output->activation_flag)
                                m->rls[z]->input = m->rls[z2]->cl_output->post_activation;
                            else
                                m->rls[z]->input = m->rls[z2]->cl_output->pre_activation;
                        }
                        if(z2!=z){
                            ff_cl_cl(m->rls[z2]->cl_output,m->rls[z]->cls[k1k2k3k4[2]-count]);

                        }
                        else{
                            ff_cl_cl(m->rls[z2]->cls[k1k2k3k4[2]-1-count2],m->rls[z]->cls[k1k2k3k4[2]-count]);
                        }
                    }
                    
                    if(k1k2k3k4[2]-count == m->rls[z]->n_cl-1){
                        if(m->rls[z]->cls[k1k2k3k4[2]-count]->pooling_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->post_pooling,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows2*m->rls[z]->cls[k1k2k3k4[2]-count]->cols2);
                        else if(m->rls[z]->cls[k1k2k3k4[2]-count]->normalization_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->post_normalization,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows1*m->rls[z]->cls[k1k2k3k4[2]-count]->cols1);
                        else if(m->rls[z]->cls[k1k2k3k4[2]-count]->activation_flag)
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->post_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows1*m->rls[z]->cls[k1k2k3k4[2]-count]->cols1);
                        else
                            sum1D(m->rls[z]->input,m->rls[z]->cls[k1k2k3k4[2]-count]->pre_activation,m->rls[z]->cl_output->pre_activation,m->rls[z]->cls[k1k2k3k4[2]-count]->n_kernels*m->rls[z]->cls[k1k2k3k4[2]-count]->rows1*m->rls[z]->cls[k1k2k3k4[2]-count]->cols1);
                        
                        if(m->rls[z]->cl_output->activation_flag == LEAKY_RELU)
                            leaky_relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == RELU)
                            relu_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == SIGMOID)
                            sigmoid_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);
                        else if(m->rls[z]->cl_output->activation_flag == TANH)
                            tanhh_array(m->rls[z]->cl_output->pre_activation,m->rls[z]->cl_output->post_activation, m->rls[z]->cl_output->n_kernels*m->rls[z]->cl_output->rows1*m->rls[z]->cl_output->cols1);

                    }
                    
                    k1k2k3k4[2]++;
                    
                    
                }
                
                else if(m->sla[i][j] == BNS){
                    k1k2k3k4[3]++;
                    free(temp);
                    return i;
                }
                
            }
            
        }
    }
    
    free(temp);
    return i;
}


/* This function computes the back-propagation for a bmodel m. each first layer at the index l makes the backprop
 * from the first layer at the index l+1. if the input is a 1d array then you should split its dimension
 * in 3 dimension to turn the input in a tensor, for example:
 * I have an input array of legth 59, then i can split this in 3 dimensions: depth = 1, row = 1, cols = 59
 * 
 * Input:
 *             
 *             @ bmodel* m:= the model with the layers
 *             @ int tensor_depth:= the depth of the input tensor
 *             @ int tensor_i:= the number of rows of the tensor
 *             @ int tensor_j:= the number of columns of the tensor
 *             @ float* input:= your input array
 *                @ float* error:= the error of the last layer of the last function computed
 *                @ int error_dimension:= the dimension of the float* error vector
 *             @ int* k1k2k3k4:= is an array of input of dimension = 4, where k1 keeps the fully connected layers that have been reached
 *                               k2 the convolutional layers, k3 the residual layers and k4 the batch normalized layers (size : 4)
 * 
 * 
 * */
float* bmodel_tensor_input_bp(bmodel* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension, int* k1k2k3k4){
    if(m == NULL)
        return NULL;
        
    int i,j,z,w,count,count2,z2,layers = k1k2k3k4[0]+k1k2k3k4[1]+k1k2k3k4[2]+k1k2k3k4[3]-1;
 
    
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
    for(i = layers; i >= 0; i--){
        for(j = 0; j < 1 && m->sla[i][j] != 0; j++){
            
            
            if(!i){
                if(m->sla[i][j] == FCLS){
                    k1k2k3k4[0]--;
                    if(m->fcls[k1k2k3k4[0]]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    error1 = bp_cl_fcl(temp,m->fcls[k1k2k3k4[0]],error1);
                    
                }
                
                else if(m->sla[i][j] == CLS){
                    k1k2k3k4[1]--;
                    if(m->cls[k1k2k3k4[1]]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    error1 = bp_cl_cl(temp,m->cls[k1k2k3k4[1]],error1);
                    
                    
                }
                
                else if(m->sla[i][j] == RLS){
                    k1k2k3k4[2]--;
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k1k2k3k4[2]; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                    
                    if(k1k2k3k4[2]-count == m->rls[z]->n_cl-1){
                        error_residual = error1; 
                    }
                    
                    if(m->rls[z]->cls[k1k2k3k4[2]-count]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    
                    error1 = bp_cl_cl(temp,m->rls[z]->cls[k1k2k3k4[2]-count],error1);
                    
                    
                    if(k1k2k3k4[2]-count == 0)
                        sum1D(error1,error_residual,error1,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    
                    
                    
                }
                
                else if(m->sla[i][j] == BNS){
                    k1k2k3k4[3]--;
                    free(temp);
                    return error1;
                }
            }
            
            else{
                
                if(m->sla[i][j] == FCLS){
                    k1k2k3k4[0]--;
                    if(m->fcls[k1k2k3k4[0]]->activation_flag == SOFTMAX && i != m->layers-1 && m->sla[i+1][0] != 0){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        error1 = bp_fcl_fcl(m->fcls[k1k2k3k4[0]-1],m->fcls[k1k2k3k4[0]],error1);
                        }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error1 = bp_cl_fcl(m->cls[k1k2k3k4[1]-1],m->fcls[k1k2k3k4[0]], error1);
                        }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k1k2k3k4[2]-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        error1 = bp_cl_fcl(m->rls[z2]->cl_output,m->fcls[k1k2k3k4[0]],error1);
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
                    k1k2k3k4[1]--;
                    if(m->cls[k1k2k3k4[1]]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    if(m->sla[i-1][0] == FCLS){
                        error1 = bp_fcl_cl(m->fcls[k1k2k3k4[0]-1],m->cls[k1k2k3k4[1]],error1);
                        
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error1 = bp_cl_cl(m->cls[k1k2k3k4[1]-1],m->cls[k1k2k3k4[1]],error1);
                        
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k1k2k3k4[2]-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                    
                        
                        error1 = bp_cl_cl(m->rls[z2]->cl_output,m->cls[k1k2k3k4[1]],error1);
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
                    k1k2k3k4[2]--;
                    count = 0;
                    for(z = 0; z < m->n_rl && count <= k1k2k3k4[2]; z++){
                        count+=m->rls[z]->n_cl;
                    }
                    
                    
                    z--;
                    count-=m->rls[z]->n_cl;
                    
                    if(k1k2k3k4[2]-count == m->rls[z]->n_cl-1){
                        
                        error_residual = error1;                    
                        
                    }
                    
                    if(m->rls[z]->cls[k1k2k3k4[2]-count]->activation_flag == SOFTMAX){
                        fprintf(stderr,"Error: the softmax can be applied only on the last fully-connected layers\n");
                        exit(1);
                    }
                    
                    
                    if(m->sla[i-1][0] == FCLS){
                        
                    
                        error1 = bp_fcl_cl(m->fcls[k1k2k3k4[0]-1],m->rls[z]->cls[k1k2k3k4[2]-count],error1);
                       
                        
                        
                    }
                    
                    else if(m->sla[i-1][0] == CLS){
                        error1 = bp_cl_cl(m->cls[k1k2k3k4[1]-1],m->rls[z]->cls[k1k2k3k4[2]-count],error1);
                       
                        
                    }
                    
                    if(m->sla[i-1][0] == RLS){
                        count2 = 0;
                        for(z2 = 0; z2 < m->n_rl && count2 <= k1k2k3k4[2]-1; z2++){
                            count2+=m->rls[z2]->n_cl;
                        }
                        
                        z2--;
                        count2-=m->rls[z2]->n_cl;
                        if(z2 == z)
                            error1 = bp_cl_cl(m->rls[z2]->cls[k1k2k3k4[2]-1-count2],m->rls[z]->cls[k1k2k3k4[2]-count],error1);
                        else{
                            error1 = bp_cl_cl(m->rls[z2]->cl_output,m->rls[z]->cls[k1k2k3k4[2]-count],error1);
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
                    
                    
                    
                    if(k1k2k3k4[2]-count == 0)
                        sum1D(error1,error_residual,error1,m->rls[z]->channels*m->rls[z]->input_rows*m->rls[z]->input_cols);
                    
                    
                    
                    
                }
                
                 else if(m->sla[i][j] == BNS){
                    k1k2k3k4[3]--;
                    free(temp);
                    return error1;
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


