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
 * see layers.c files
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

/* This function copies a bmodel with the rule: teta_i:= teta_j*tau +(1-tau)*teta_i
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


/* this function compute the space allocated by the arrays of m
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
        sum+=m->cls[i]->n_kernels*m->cls[i]->channels*m->cls[i]->kernel_rows*m->cls[i]->kernel_cols;
    }
    
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            sum+=m->rls[i]->cls[j]->n_kernels*m->rls[i]->cls[j]->channels*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols;
        }
    }
    
    for(i = 0; i < m->n_bn; i++){
        sum+=m->bns[i]->vector_dim;
    }
    
    return sum;
}


/* This function can update the bmodel of the network using the adam algorithm or the nesterov momentum
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
 * 
 * */
void update_bmodel(bmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda){
    if(m == NULL)
        return;
    
    lambda*=mini_batch_size;
    
    if(regularization == L2_REGULARIZATION){
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
        (*b1)*=BETA1_ADAM;
        (*b2)*=BETA2_ADAM;
    }    
    

}


/* This function sum the partial derivatives in bmodel m1 and m2 in m3
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



