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


/* This function creates a variational auto encoder filled with amodel encoder and a decoder model
 * pay attention: the vaemodel store in encoder and decoder the copies of encoder and decoder param
 * the final array of encoder is considered the first half as the mean and the second half as the std
 * Inputs:
 * 
 *     
 *                 @ model* encoder:= the encoder
 *                 @ model* decoder:= the decoder
 *                 @ int latent_size:= the latent size of the std, mean and input
 * */
vaemodel* variational_auto_encoder_model(model* encoder, model* decoder, int latent_size){
    if(encoder == NULL || decoder == NULL || !latent_size){
        fprintf(stderr,"Error: encorder must be != NULL, decoder must be != NULL, latent_size must be > 0\n");
        exit(1);
    }
    vaemodel* vm = (vaemodel*)malloc(sizeof(vaemodel));
    vm->z = (float*)calloc(latent_size,sizeof(float));
    vm->input = (float*)calloc(latent_size,sizeof(float));
    vm->dmean = (float*)calloc(latent_size,sizeof(float));
    vm->dstd = (float*)calloc(latent_size,sizeof(float));
    vm->latent_size = latent_size;
    vm->encoder = copy_model(encoder);
    vm->decoder = copy_model(decoder);
    
    return vm;
}


/* This function frees the space allocated by a vaemodel* structure
 * 
 * Input:
 *             
 *                 @ vaemodel* vm:= the variational autoencoder that must be freed
 * */
void free_vae_model(vaemodel* vm){
    if(vm == NULL)
        return
    
    free(vm->z);
    free(vm->input);
    free(vm->dmean);
    free(vm->dstd);
    free_model(vm->encoder);
    free_model(vm->decoder);
    free(vm);
}

/* This function copies a vaemodel
 * 
 * Inputs:
 * 
 *                 @ vaemodel* vm:= the vaemodel that must be copied
 * 
 * */    
vaemodel* copy_vae_model(vaemodel* vm){
    vaemodel* vm2 = variational_auto_encoder_model(vm->encoder,vm->decoder,vm->latent_size);
    return vm2;
}


/* this function paste the encoder and decoder models from vm1 to vm2
 * 
 * Inputs:
 * 
 *             @ vaemodel* vm1:= what must be pasted
 *             @ vaemodel* vm2:= where is pasted
 * 
 * */
void paste_vae_model(vaemodel* vm1, vaemodel* vm2){
    if(vm1 == NULL || vm2 == NULL)
        return;
    
    paste_model(vm1->encoder,vm2->encoder);
    paste_model(vm1->decoder,vm2->decoder);
}

/* this function paste the encoder and decoder models from vm1 to vm2 with the rule: teta_i:= teta_j*tau +(1-tau)*teta_i
 * 
 * Inputs:
 * 
 *             @ vaemodel* vm1:= what must be pasted
 *             @ vaemodel* vm2:= where is pasted
 * 
 * */
void slow_paste_vae_model(vaemodel* vm1, vaemodel* vm2, float tau){
    if(vm1 == NULL || vm2 == NULL)
        return;
    slow_paste_model(vm1->encoder,vm2->encoder, tau);
    slow_paste_model(vm1->decoder,vm2->decoder, tau);
}

/* This function resets a vaemodel
 * returns a vaemodel equal to the one as input but with all resetted except for weights and biases of encoder and decoder
 * */
void reset_vae_model(vaemodel* vm){
    if(vm == NULL)
        return;
    int i;
    reset_model(vm->encoder);
    reset_model(vm->decoder);
    for(i = 0; i < vm->latent_size; i++){
        vm->dstd[i] = 0;
        vm->dmean[i] = 0;
        vm->z[i] = 0;
        vm->input[i] = 0;
    }
}

/* this function computes the space allocated by this structure
 * 
 * */
unsigned long long int size_of_vae_model(vaemodel* vm){
    return size_of_model(vm->encoder)+size_of_model(vm->decoder) + vm->latent_size*4;
}

/* this function save a variational auto encoder in a 2 .bin file
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ vaemodel* vm:= the vaemodel that must be saved
 *             @ int n:= the encoder is saved in n.bin file
 *             @ int m:= the decoder is saved in m.bin file
 * */
void save_vae_model(vaemodel* vm, int n, int m){
    save_model(vm->encoder,n);
    save_model(vm->decoder,m);
}

/* this function load a vaemodel given 2 files as input, the first one contains the encoder, the second one the decoder*/
vaemodel* load_vae_model(char* file1, char* file2){
    model* encoder = load_model(file1);
    model* decoder = load_model(file2);
    int latent_size;
    if(decoder->sla[0][0] == FCLS)
        latent_size = decoder->fcls[0]->input;
    else if(decoder->sla[0][0] == CLS)
        latent_size = decoder->cls[0]->channels*decoder->cls[0]->input_rows*decoder->cls[0]->input_cols;
    else if(decoder->sla[0][0] == RLS)
        latent_size = decoder->rls[0]->channels*decoder->rls[0]->input_rows*decoder->rls[0]->input_cols;
    
    vaemodel* vm = variational_auto_encoder_model(encoder,decoder,latent_size);
    free_model(encoder);
    free_model(decoder);
    
    return vm;
}



/* this function computes the feed forward for a variational auto encoder
 * 
 * Inputs:
 * 
 * 
 *                 @ vaemodel* vm:= the vae
 *                 @ int tensor_depth:= the input has seen as a tensor, number of channels
 *                 @ int tensor_i:= number of rows of the tensor
 *                 @ int tensor_j:= number of cols of the tensor
 *                 @ float* input:= the input
 * */
void vae_model_tensor_input_ff(vaemodel* vm, int tensor_depth, int tensor_i, int tensor_j,float* input){
    if(vm == NULL)
        return;
        
    model_tensor_input_ff(vm->encoder,tensor_depth,tensor_i,tensor_j,input);
    int i;
    for(i = 0; i < vm->latent_size; i++){
        vm->input[i] = random_normal();
    }
    
    for(i = 0; i < vm->encoder->layers-1; i++){
        if(vm->encoder->sla[i][0] == 0){
            i--;
            break;
        }
    }
    
    if(vm->encoder->sla[i][0] == FCLS){
        if(vm->encoder->fcls[vm->encoder->n_fcl-1]->dropout_flag){
            fprintf(stderr,"Error: is not a good practice using dropout for the encoder of vae model, 'cause you cannot shift the output for the decoder in the Test time\n");
            exit(1);
        }
        if(vm->encoder->fcls[vm->encoder->n_fcl-1]->activation_flag){
            if(vm->encoder->fcls[vm->encoder->n_fcl-1]->activation_flag == SOFTMAX){
                fprintf(stderr,"Error: cannot be computed with softmax function for the final layer of the encoder\n");
                exit(1);
            }
            dot1D(vm->input,&vm->encoder->fcls[vm->encoder->n_fcl-1]->post_activation[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->fcls[vm->encoder->n_fcl-1]->post_activation,vm->z,vm->latent_size);
        }
        
        else{
            dot1D(vm->input,&vm->encoder->fcls[vm->encoder->n_fcl-1]->pre_activation[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->fcls[vm->encoder->n_fcl-1]->pre_activation,vm->z,vm->latent_size);
        }
    }
    
    else if(vm->encoder->sla[i][0] == CLS){
        if(vm->encoder->cls[vm->encoder->n_cl-1]->pooling_flag){
            dot1D(vm->input,&vm->encoder->cls[vm->encoder->n_cl-1]->post_pooling[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->cls[vm->encoder->n_cl-1]->post_pooling,vm->z,vm->latent_size);
        }
        
        else if(vm->encoder->cls[vm->encoder->n_cl-1]->normalization_flag){
            dot1D(vm->input,&vm->encoder->cls[vm->encoder->n_cl-1]->post_normalization[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->cls[vm->encoder->n_cl-1]->post_normalization,vm->z,vm->latent_size);
        }
        
        else if(vm->encoder->cls[vm->encoder->n_cl-1]->activation_flag){
            dot1D(vm->input,&vm->encoder->cls[vm->encoder->n_cl-1]->post_activation[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->cls[vm->encoder->n_cl-1]->post_activation,vm->z,vm->latent_size);
        }
        
        else {
            dot1D(vm->input,&vm->encoder->cls[vm->encoder->n_cl-1]->pre_activation[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->cls[vm->encoder->n_cl-1]->pre_activation,vm->z,vm->latent_size);
        }
    }
    
    else if(vm->encoder->sla[i][0] == RLS){
        if(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->activation_flag){
            dot1D(vm->input,&vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->post_activation[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->post_activation,vm->z,vm->latent_size);
        }
        
        else{
            dot1D(vm->input,&vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation[vm->latent_size],vm->z,vm->latent_size);
            sum1D(vm->z,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation,vm->z,vm->latent_size);
        }
    }
    
    model_tensor_input_ff(vm->decoder,vm->latent_size,1,1,vm->z);
    
}


/* this function computes the partial derivatives of a variational auto encoder
 * */
float* vae_model_tensor_input_bp(vaemodel* vm, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension){
    
    float* temp = model_tensor_input_bp(vm->decoder,vm->latent_size,1,1,vm->z,error,error_dimension);
    copy_array(temp,vm->dmean,vm->latent_size);
    dot1D(temp,vm->input,vm->dstd,vm->latent_size);
    
    float* temp2 = (float*)calloc(vm->latent_size*2,sizeof(float));
    float* temp3;
    
    int i,j;
    
    for(i = 0; i < vm->encoder->layers-1; i++){
        if(vm->encoder->sla[i][0] == 0){
            i--;
            break;
        }
    }
    
    if(vm->encoder->sla[i][0] == FCLS){
        if(vm->encoder->fcls[vm->encoder->n_fcl-1]->dropout_flag){
            fprintf(stderr,"Error: is not a good practice using dropout for the encoder of vae model, 'cause you cannot shift the output for the decoder in the Test time\n");
            exit(1);
        }
        if(vm->encoder->fcls[vm->encoder->n_fcl-1]->activation_flag){
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->fcls[vm->encoder->n_fcl-1]->post_activation[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->fcls[vm->encoder->n_fcl-1]->post_activation[j];
            }
        }
        
        else{
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->fcls[vm->encoder->n_fcl-1]->pre_activation[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->fcls[vm->encoder->n_fcl-1]->pre_activation[j];
            }
        }
    }
    
    else if(vm->encoder->sla[i][0] == CLS){
        if(vm->encoder->cls[vm->encoder->n_cl-1]->pooling_flag){
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->cls[vm->encoder->n_cl-1]->post_pooling[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->cls[vm->encoder->n_cl-1]->post_pooling[j];
            }
        }
        
        else if(vm->encoder->cls[vm->encoder->n_cl-1]->normalization_flag){
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->cls[vm->encoder->n_cl-1]->post_normalization[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->cls[vm->encoder->n_cl-1]->post_normalization[j];
            }
        }
        
        else if(vm->encoder->cls[vm->encoder->n_cl-1]->activation_flag){
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->cls[vm->encoder->n_cl-1]->post_activation[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->cls[vm->encoder->n_cl-1]->post_activation[j];
            }
        }
        
        else {
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->cls[vm->encoder->n_cl-1]->pre_activation[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->cls[vm->encoder->n_cl-1]->pre_activation[j];
            }
        }
    }
    
    else if(vm->encoder->sla[i][0] == RLS){
        if(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->activation_flag){
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->post_activation[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->post_activation[j];
            }
            
            if(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->activation_flag == SIGMOID)
                derivative_sigmoid_array(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->temp3,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->n_kernels*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->rows1*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->cols1);
            else if(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->activation_flag == TANH)
                derivative_tanhh_array(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->temp3,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->n_kernels*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->rows1*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->cols1);
            else if(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->activation_flag == LEAKY_RELU)
                derivative_leaky_relu_array(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->temp3,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->n_kernels*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->rows1*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->cols1);
            else if(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->activation_flag == RELU)
                derivative_relu_array(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->temp3,vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->n_kernels*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->rows1*vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->cols1);
            
            dot1D(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->temp3,vm->dmean,vm->dmean,vm->latent_size);
            dot1D(&vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->temp3[vm->latent_size],vm->dstd,vm->dstd,vm->latent_size);
        }
        
        else{
            for(j = 0; j < vm->latent_size; j++){
                vm->dstd[j] += (exp(vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation[vm->latent_size+j]) - 1)/2;
                vm->dmean[j] += vm->encoder->rls[vm->encoder->n_rl-1]->cl_output->pre_activation[j];
            }
        }
    }
    
    sum1D(vm->dmean,temp,temp2,vm->latent_size);
    dot1D(temp,vm->input,temp2+vm->latent_size,vm->latent_size);
    sum1D(vm->dstd,temp2+vm->latent_size,temp2+vm->latent_size,vm->latent_size);
    
    temp3 = model_tensor_input_bp(vm->encoder,tensor_depth,tensor_i,tensor_j,input,temp2,vm->latent_size*2);
    
    free(temp2);
    
    return temp3;
    
    
}

/* number of weights of a cae model*/
int count_weights_vae_model(vaemodel* vm){
    return count_weights(vm->encoder) + count_weights(vm->decoder);
}

void update_vae_model(vaemodel* vm, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, int total_number_weights, float lambda, unsigned long long int* t){
    update_model(vm->encoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
    if(gradient_descent_flag == ADAM){
        (*b1)/=BETA1_ADAM;
        (*b2)/=BETA2_ADAM;
    }
    
    else if(gradient_descent_flag == RADAM){
        (*b1)/=BETA1_ADAM;
        (*b2)/=BETA2_ADAM;
        (*t)--;
    }
    update_model(vm->decoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
}


void sum_vae_model_partial_derivatives(vaemodel* vm, vaemodel* vm2, vaemodel* vm3){
    if(vm == NULL || vm2 == NULL || vm3 == NULL){
        fprintf(stderr,"Error: passed NULL pointer as values in sum_vae_model_partial_derivatives\n");
        exit(1);
    }
    sum_model_partial_derivatives(vm->encoder,vm2->encoder,vm3->encoder);
    sum_model_partial_derivatives(vm->decoder,vm2->decoder,vm3->decoder);
}

