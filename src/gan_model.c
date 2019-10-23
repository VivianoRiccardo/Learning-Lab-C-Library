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



ganmodel* gan_network(model* generator, model* discriminator,float g_lr, float g_momentum, int mini_batch_size, int g_gradient_descent_flag, float g_b1, float g_b2, int g_regularization, float g_lambda,float d_lr, float d_momentum, int d_gradient_descent_flag, float d_b1, float d_b2, int d_regularization, float d_lambda){
    if(generator == NULL || discriminator == NULL){
        fprintf(stderr,"Error: you have to put a generator and a discriminator in the ganmodel\n");
        exit(1);
    }
    ganmodel* gm = (ganmodel*)malloc(sizeof(ganmodel));
    gm->generator = generator;
    gm->discriminator = discriminator;
    gm->discriminator2 = copy_model(discriminator);
    gm->mini_batch_size = mini_batch_size;
    gm->generator_b1 = g_b1;
    gm->generator_b2 = g_b2;
    gm->discriminator_b1 = d_b1;
    gm->discriminator_b2 = d_b2;
    gm->generator_gradient_descent_flag = g_gradient_descent_flag;
    gm->discriminator_gradient_descent_flag = d_gradient_descent_flag;
    gm->generator_lambda = g_lambda;
    gm->discriminator_lambda = d_lambda;
    gm->generator_lr = g_lr;
    gm->discriminator_lr = d_lr;
    gm->generator_momentum = g_momentum;
    gm->discriminator_momentum = d_momentum;
    gm->generator_regularization = g_regularization;
    gm->discriminator_regularization = d_regularization;
    gm->generator_t = 1;
    gm->discriminator_t = 1;
    gm->generator_total_number_weights = count_weights(generator);
    gm->discriminator_total_number_weights = count_weights(discriminator);
    
    return gm;
}

void free_ganmodel(ganmodel* gm){
    free_model(gm->generator);
    free_model(gm->discriminator);
    free_model(gm->discriminator2);
    free(gm);
}

ganmodel* copy_ganmodel(ganmodel* gm){
    return gan_network(copy_model(gm->generator),copy_model(gm->discriminator),gm->generator_lr,gm->generator_momentum,gm->mini_batch_size,gm->generator_gradient_descent_flag,gm->generator_b1,gm->generator_b2,gm->generator_regularization,gm->generator_lambda,gm->discriminator_lr,gm->discriminator_momentum,gm->discriminator_gradient_descent_flag,gm->discriminator_b1,gm->discriminator_b2,gm->discriminator_regularization,gm->discriminator_lambda);
}

void paste_ganmodel(ganmodel* m1, ganmodel* m2){
    paste_model(m1->generator,m2->generator);
    paste_model(m1->discriminator,m2->discriminator);
    paste_model(m1->discriminator2,m2->discriminator2);
}

void slow_paste_ganmodel(ganmodel* m1, ganmodel* m2, float tau){    
    slow_paste_model(m1->generator,m2->generator,tau);
    slow_paste_model(m1->discriminator,m2->discriminator,tau);
    slow_paste_model(m1->discriminator2,m2->discriminator2,tau);
}

void reset_ganmodel(ganmodel* m1){
    reset_model(m1->generator);
    reset_model(m1->discriminator);
    reset_model(m1->discriminator2);
}

unsigned long long int size_of_ganmodel(ganmodel* m1){
    return size_of_model(m1->generator)+2*size_of_model(m1->discriminator);
}

void save_ganmodel(ganmodel* gm, int n, int m){
    save_model(gm->generator,n);
    save_model(gm->discriminator,m);
}

ganmodel* load_ganmodel(char* file1, char* file2){
    model* g = load_model(file1);
    model* d = load_model(file2);
    return gan_network(g,d,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);//you have to reset the update params
}


void discriminator_feed_forward(ganmodel* gm, float* real_input, float* noise_input, int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j){
    model_tensor_input_ff(gm->generator,tensor_input_g_depth,tensor_input_g_i,tensor_input_g_j,noise_input);
    model_tensor_input_ff(gm->discriminator,tensor_input_d_depth,tensor_input_d_i,tensor_input_d_j,real_input);
    int i;
    float* output;
    for(i = 0; i < gm->generator->layers-1 && gm->generator->sla[i][0] != 0; i++);
    if(gm->generator->sla[i][0] == 0)
        i--;
    if(gm->generator->sla[i][0] == FCLS){
        if(gm->generator->fcls[gm->generator->n_fcl-1]->activation_flag)
            output = gm->generator->fcls[gm->generator->n_fcl-1]->post_activation;
        else    
            output = gm->generator->fcls[gm->generator->n_fcl-1]->pre_activation;
    }
    else if(gm->generator->sla[i][0] == CLS){
        if(gm->generator->cls[gm->generator->n_cl-1]->pooling_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_pooling;
        else if(gm->generator->cls[gm->generator->n_cl-1]->normalization_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_normalization;
        else if(gm->generator->cls[gm->generator->n_cl-1]->activation_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_activation;
        else
            output = gm->generator->cls[gm->generator->n_cl-1]->pre_activation;
        
    }
    else if(gm->generator->sla[i][0] == RLS){
        if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag)
            output = gm->generator->rls[gm->generator->n_rl-1]->cl_output->post_activation;
        else    
            output = gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation;
    }
    model_tensor_input_ff(gm->discriminator2,tensor_input_d_depth,tensor_input_d_i,tensor_input_d_j,output);
}

void discriminator_back_propagation(ganmodel* gm, float* real_input, float* noise_input,  int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j){
    float* error1 = (float*)malloc(sizeof(float));
    float* error2 = (float*)malloc(sizeof(float));
    int i;
    for(i = 0; i < gm->discriminator->layers-1 && gm->discriminator->sla[i][0] != 0; i++);
    if(gm->discriminator->sla[i][0] == 0)
        i--;
    if(gm->discriminator->sla[i][0] == FCLS){
        if(gm->discriminator->fcls[gm->discriminator->n_fcl-1]->activation_flag){
            error1[0] = -log((double)gm->discriminator->fcls[gm->discriminator->n_fcl-1]->post_activation[0]);
            error2[0] = -log((double)(1-gm->discriminator2->fcls[gm->discriminator->n_fcl-1]->post_activation[0]));
        }
        else{
            error1[0] = -log((double)gm->discriminator->fcls[gm->discriminator->n_fcl-1]->pre_activation[0]);
            error2[0] = -log((double)(1-gm->discriminator2->fcls[gm->discriminator->n_fcl-1]->pre_activation[0]));
            
        }
    }
    else if(gm->discriminator->sla[i][0] == CLS){
        if(gm->discriminator->cls[gm->discriminator->n_cl-1]->pooling_flag){
            error1[0] = -log((double)gm->discriminator->cls[gm->discriminator->n_cl-1]->post_pooling[0]);
            error2[0] = -log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->post_pooling[0]));
            
        }
        else if(gm->discriminator->cls[gm->discriminator->n_cl-1]->normalization_flag){
            error1[0] = -log((double)gm->discriminator->cls[gm->discriminator->n_cl-1]->post_normalization[0]);
            error2[0] = -log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->post_normalization[0]));
            
        }
        else if(gm->discriminator->cls[gm->discriminator->n_cl-1]->activation_flag){
            error1[0] = -log((double)gm->discriminator->cls[gm->discriminator->n_cl-1]->post_activation[0]);
            error2[0] = -log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->post_activation[0]));
            
        }
        else{    
            error1[0] = -log((double)gm->discriminator->cls[gm->discriminator->n_cl-1]->pre_activation[0]);
            error2[0] = -log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->pre_activation[0]));
            
        }
    }
    
    else if(gm->discriminator->sla[i][0] == RLS){
        if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag){
            error1[0] = -log((double)gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->post_activation[0]);
            error2[0] = -log((double)(1-gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->post_activation[0]));
            if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == SIGMOID){
                derivative_sigmoid_array(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->temp3,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->n_kernels*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->rows1*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->cols1);
                derivative_sigmoid_array(gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == TANH){
                derivative_tanhh_array(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->temp3,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->n_kernels*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->rows1*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->cols1);
                derivative_tanhh_array(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->temp3,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->n_kernels*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->rows1*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->cols1);
                derivative_leaky_relu_array(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == RELU){
                derivative_relu_array(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->temp3,gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->n_kernels*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->rows1*gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->cols1);
                derivative_relu_array(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            dot1D(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->temp3,error1,error1,1);
            dot1D(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,error2,error2,1);
        }
        else{
            error1[0] = -log((double)gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation[0]);
            error2[0] = -log((double)(1-gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation[0]));
            
        }
    }
    
    model_tensor_input_bp(gm->discriminator,tensor_input_d_depth,tensor_input_d_i,tensor_input_d_j,real_input,error1,1);
    model_tensor_input_bp(gm->discriminator2,tensor_input_d_depth,tensor_input_d_i,tensor_input_d_j,noise_input,error2,1);
    
    sum_model_partial_derivatives(gm->discriminator,gm->discriminator2,gm->discriminator);
    
    free(error1);
    free(error2);
        
}

void generator_feed_forward(ganmodel* gm, float* noise_input,  int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j){
    model_tensor_input_ff(gm->generator,tensor_input_g_depth,tensor_input_g_i,tensor_input_g_j,noise_input);
    int i;
    float* output;
    for(i = 0; i < gm->generator->layers-1 && gm->generator->sla[i][0] != 0; i++);
    if(gm->generator->sla[i][0] == 0)
        i--;
    if(gm->generator->sla[i][0] == FCLS){
        if(gm->generator->fcls[gm->generator->n_fcl-1]->activation_flag)
            output = gm->generator->fcls[gm->generator->n_fcl-1]->post_activation;
        else    
            output = gm->generator->fcls[gm->generator->n_fcl-1]->pre_activation;
    }
    else if(gm->generator->sla[i][0] == CLS){
        if(gm->generator->cls[gm->generator->n_cl-1]->pooling_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_pooling;
        else if(gm->generator->cls[gm->generator->n_cl-1]->normalization_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_normalization;
        else if(gm->generator->cls[gm->generator->n_cl-1]->activation_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_activation;
        else
            output = gm->generator->cls[gm->generator->n_cl-1]->pre_activation;
        
    }
    else if(gm->generator->sla[i][0] == RLS){
        if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag)
            output = gm->generator->rls[gm->generator->n_rl-1]->cl_output->post_activation;
        else    
            output = gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation;
    }
    model_tensor_input_ff(gm->discriminator2,tensor_input_d_depth,tensor_input_d_i,tensor_input_d_j,output);
}

float* generator_back_propagation(ganmodel* gm, float* noise_input,  int tensor_input_g_depth, int tensor_input_g_i, int tensor_input_g_j,int tensor_input_d_depth, int tensor_input_d_i, int tensor_input_d_j, int generator_output_size){
    float* error1;
    float* error2 = (float*)malloc(sizeof(float));
    int i;
    
    float* output;
    for(i = 0; i < gm->generator->layers-1 && gm->generator->sla[i][0] != 0; i++);
    if(gm->generator->sla[i][0] == 0)
        i--;
    if(gm->generator->sla[i][0] == FCLS){
        if(gm->generator->fcls[gm->generator->n_fcl-1]->activation_flag)
            output = gm->generator->fcls[gm->generator->n_fcl-1]->post_activation;
        else    
            output = gm->generator->fcls[gm->generator->n_fcl-1]->pre_activation;
    }
    else if(gm->generator->sla[i][0] == CLS){
        if(gm->generator->cls[gm->generator->n_cl-1]->pooling_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_pooling;
        else if(gm->generator->cls[gm->generator->n_cl-1]->normalization_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_normalization;
        else if(gm->generator->cls[gm->generator->n_cl-1]->activation_flag)
            output = gm->generator->cls[gm->generator->n_cl-1]->post_activation;
        else
            output = gm->generator->cls[gm->generator->n_cl-1]->pre_activation;
        
    }
    else if(gm->generator->sla[i][0] == RLS){
        if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag)
            output = gm->generator->rls[gm->generator->n_rl-1]->cl_output->post_activation;
        else    
            output = gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation;
    }
    
    for(i = 0; i < gm->discriminator->layers-1 && gm->discriminator->sla[i][0] != 0; i++);
    if(gm->discriminator->sla[i][0] == 0)
        i--;
    if(gm->discriminator->sla[i][0] == FCLS){
        if(gm->discriminator->fcls[gm->discriminator->n_fcl-1]->activation_flag){
            error2[0] = log((double)(1-gm->discriminator2->fcls[gm->discriminator->n_fcl-1]->post_activation[0]));
        }
        else{
            error2[0] = log((double)(1-gm->discriminator2->fcls[gm->discriminator->n_fcl-1]->pre_activation[0]));
            
        }
    }
    else if(gm->discriminator->sla[i][0] == CLS){
        if(gm->discriminator->cls[gm->discriminator->n_cl-1]->pooling_flag){
            error2[0] = log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->post_pooling[0]));
            
        }
        else if(gm->discriminator->cls[gm->discriminator->n_cl-1]->normalization_flag){
            error2[0] = log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->post_normalization[0]));
            
        }
        else if(gm->discriminator->cls[gm->discriminator->n_cl-1]->activation_flag){
            error2[0] = log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->post_activation[0]));
            
        }
        else{    
            error2[0] = log((double)(1-gm->discriminator2->cls[gm->discriminator->n_cl-1]->pre_activation[0]));
            
        }
    }
    
    else if(gm->discriminator->sla[i][0] == RLS){
        if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag){
            error2[0] = log((double)(1-gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->post_activation[0]));
            if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == SIGMOID){
                derivative_sigmoid_array(gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == TANH){
                derivative_tanhh_array(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->discriminator->rls[gm->discriminator->n_rl-1]->cl_output->activation_flag == RELU){
                derivative_relu_array(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->pre_activation,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->n_kernels*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->rows1*gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->cols1);
            
            }
            dot1D(gm->discriminator2->rls[gm->discriminator2->n_rl-1]->cl_output->temp3,error2,error2,1);
        }
        else{
            error2[0] = log((double)(1-gm->discriminator2->rls[gm->discriminator->n_rl-1]->cl_output->pre_activation[0]));
            
        }
    }
    
    error1 = model_tensor_input_bp(gm->discriminator2,tensor_input_d_depth,tensor_input_d_i,tensor_input_d_j,output,error2,1);
    
    for(i = 0; i < gm->generator->layers-1 && gm->generator->sla[i][0] != 0; i++);
    if(gm->generator->sla[i][0] == 0)
        i--;
    
    if(gm->generator->sla[i][0] == RLS){
        if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag){
            if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag == SIGMOID){
                derivative_sigmoid_array(gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation,gm->generator->rls[gm->generator->n_rl-1]->cl_output->temp3,gm->generator->rls[gm->generator->n_rl-1]->cl_output->n_kernels*gm->generator->rls[gm->generator->n_rl-1]->cl_output->rows1*gm->generator->rls[gm->generator->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag == TANH){
                derivative_tanhh_array(gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation,gm->generator->rls[gm->generator->n_rl-1]->cl_output->temp3,gm->generator->rls[gm->generator->n_rl-1]->cl_output->n_kernels*gm->generator->rls[gm->generator->n_rl-1]->cl_output->rows1*gm->generator->rls[gm->generator->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag == LEAKY_RELU){
                derivative_leaky_relu_array(gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation,gm->generator->rls[gm->generator->n_rl-1]->cl_output->temp3,gm->generator->rls[gm->generator->n_rl-1]->cl_output->n_kernels*gm->generator->rls[gm->generator->n_rl-1]->cl_output->rows1*gm->generator->rls[gm->generator->n_rl-1]->cl_output->cols1);
            
            }
            else if(gm->generator->rls[gm->generator->n_rl-1]->cl_output->activation_flag == RELU){
                derivative_relu_array(gm->generator->rls[gm->generator->n_rl-1]->cl_output->pre_activation,gm->generator->rls[gm->generator->n_rl-1]->cl_output->temp3,gm->generator->rls[gm->generator->n_rl-1]->cl_output->n_kernels*gm->generator->rls[gm->generator->n_rl-1]->cl_output->rows1*gm->generator->rls[gm->generator->n_rl-1]->cl_output->cols1);
            
            }
            dot1D(gm->generator->rls[gm->generator->n_rl-1]->cl_output->temp3,error1,error1,gm->generator->rls[gm->generator->n_rl-1]->channels*gm->generator->rls[gm->generator->n_rl-1]->input_rows*gm->generator->rls[gm->generator->n_rl-1]->input_cols);
        }
    }
    free(error2);
    return model_tensor_input_bp(gm->generator,tensor_input_g_depth,tensor_input_g_i,tensor_input_g_j,noise_input,error1,generator_output_size);
    
}


void update_discriminator(ganmodel* gm){
    update_model(gm->discriminator,gm->discriminator_lr,gm->discriminator_momentum,gm->mini_batch_size,gm->discriminator_gradient_descent_flag,&gm->discriminator_b1,&gm->discriminator_b2,gm->discriminator_regularization,gm->discriminator_total_number_weights,gm->discriminator_lambda,&gm->discriminator_t);
}

void update_generator(ganmodel* gm){
    update_model(gm->generator,gm->generator_lr,gm->generator_momentum,gm->mini_batch_size,gm->generator_gradient_descent_flag,&gm->generator_b1,&gm->generator_b2,gm->generator_regularization,gm->generator_total_number_weights,gm->generator_lambda,&gm->generator_t);
}
