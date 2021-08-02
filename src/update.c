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

/* Given a model, this function update the params of the residual layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->training_mode != FREEZE_TRAINING){
                if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION || m->rls[i]->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                    if(m->rls[i]->cls[j]->training_mode == GRADIENT_DESCENT || m->rls[i]->cls[j]->training_mode == FREEZE_BIASES){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                                for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                    for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                        nesterov_momentum(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w]);
                                    }
                                }
                            }
                            if(m->rls[i]->cls[j]->training_mode != FREEZE_BIASES)
                            nesterov_momentum(&m->rls[i]->cls[j]->biases[k],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_biases[k],&m->rls[i]->cls[j]->d1_biases[k]);
                            if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                                update_batch_normalized_layer_nesterov(m->rls[i]->cls[j]->group_norm,m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels,lr,momentum,mini_batch_size);
                            }
                        }
                    }
                    else if(m->rls[i]->cls[j]->training_mode == EDGE_POPUP){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            
                            nesterov_momentum(&m->rls[i]->cls[j]->scores[k],lr,momentum,mini_batch_size, m->rls[i]->cls[j]->d_scores[k],&m->rls[i]->cls[j]->d1_scores[k]);
                                    
                        }
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->training_mode != FREEZE_TRAINING){
                if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION || m->rls[i]->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                    if(m->rls[i]->cls[j]->training_mode == GRADIENT_DESCENT || m->rls[i]->cls[j]->training_mode == FREEZE_BIASES){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                                for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                    for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                        adam_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                                    }
                                }
                            }
                            if(m->rls[i]->cls[j]->training_mode != FREEZE_BIASES)
                            adam_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                            if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                                update_batch_normalized_layer_adam(m->rls[i]->cls[j]->group_norm,m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam);
                            }
                        }
                    }
                    else if(m->rls[i]->cls[j]->training_mode == EDGE_POPUP){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            
                            adam_algorithm(&m->rls[i]->cls[j]->scores[k],&m->rls[i]->cls[j]->d1_scores[k],&m->rls[i]->cls[j]->d2_scores[k],m->rls[i]->cls[j]->d_scores[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                             
                        }
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_residual_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->training_mode != FREEZE_TRAINING){
                if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION || m->rls[i]->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                    if(m->rls[i]->cls[j]->training_mode == GRADIENT_DESCENT || m->rls[i]->cls[j]->training_mode == FREEZE_BIASES){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                                for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                    for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                        adamod(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size, beta3_adamod,&m->rls[i]->cls[j]->d3_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w]);
                                    }
                                }
                            }
                            if(m->rls[i]->cls[j]->training_mode != FREEZE_BIASES)
                            adamod(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod, &m->rls[i]->cls[j]->d3_biases[k]);
                            if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                                update_batch_normalized_layer_adamod(m->rls[i]->cls[j]->group_norm,m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam,beta3_adamod);
                            }
                        }
                    }
                    else if(m->rls[i]->cls[j]->training_mode == EDGE_POPUP){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            
                            adamod(&m->rls[i]->cls[j]->scores[k],&m->rls[i]->cls[j]->d1_scores[k],&m->rls[i]->cls[j]->d2_scores[k],m->rls[i]->cls[j]->d_scores[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->rls[i]->cls[j]->d3_scores[k]);
                                    
                        }
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the residual layers of the model with the adam diff grad optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_residual_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->training_mode != FREEZE_TRAINING){
                if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION || m->rls[i]->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                    if(m->rls[i]->cls[j]->training_mode == GRADIENT_DESCENT || m->rls[i]->cls[j]->training_mode == FREEZE_BIASES){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                                for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                    for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                        adam_diff_grad_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->rls[i]->cls[j]->d3_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w]);
                                    }
                                }
                            }
                            if(m->rls[i]->cls[j]->training_mode != FREEZE_BIASES)
                            adam_diff_grad_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->rls[i]->cls[j]->d3_biases[k]);
                            if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                                update_batch_normalized_layer_adam_diff_grad(m->rls[i]->cls[j]->group_norm,m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam);
                            }
                        }
                    }
                    else if(m->rls[i]->cls[j]->training_mode == EDGE_POPUP){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            adam_diff_grad_algorithm(&m->rls[i]->cls[j]->scores[k],&m->rls[i]->cls[j]->d1_scores[k],&m->rls[i]->cls[j]->d2_scores[k],m->rls[i]->cls[j]->d_scores[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->rls[i]->cls[j]->d3_scores[k]);
                                    
                        }
                    }
                }
            }
        }
    }
}
/* Given a model, this function update the params of the residual layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * 
 * */
void update_residual_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long t, float beta1_adam, float beta2_adam){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->training_mode != FREEZE_TRAINING){
                if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION || m->rls[i]->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                    if(m->rls[i]->cls[j]->training_mode == GRADIENT_DESCENT || m->rls[i]->cls[j]->training_mode == FREEZE_BIASES){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                                for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                    for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                        radam_algorithm(&m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d1_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],&m->rls[i]->cls[j]->d2_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                                    }
                                }
                            }
                            if(m->rls[i]->cls[j]->training_mode != FREEZE_BIASES)
                            radam_algorithm(&m->rls[i]->cls[j]->biases[k],&m->rls[i]->cls[j]->d1_biases[k],&m->rls[i]->cls[j]->d2_biases[k],m->rls[i]->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                            if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                                update_batch_normalized_layer_radam(m->rls[i]->cls[j]->group_norm,m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,t,beta1_adam,beta2_adam);
                            }
                        }
                    }
                    else if(m->rls[i]->cls[j]->training_mode == EDGE_POPUP){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                            radam_algorithm(&m->rls[i]->cls[j]->scores[k],&m->rls[i]->cls[j]->d1_scores[k],&m->rls[i]->cls[j]->d2_scores[k],m->rls[i]->cls[j]->d_scores[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                                    
                        }
                    }
                }
            }
        }
    }
}



/* Given a model, this function update the params of the convolutional layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_convolutional_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->training_mode != FREEZE_TRAINING){
            if(m->cls[j]->convolutional_flag == CONVOLUTION || m->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                if(m->cls[j]->training_mode == GRADIENT_DESCENT || m->cls[j]->training_mode == FREEZE_BIASES){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->cls[j]->channels; u++){
                            for(z = 0; z < m->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                    nesterov_momentum(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr,momentum,mini_batch_size, m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w]);
                                }
                                    
                            }
                        }
                        if(m->cls[j]->training_mode != FREEZE_BIASES)
                        nesterov_momentum(&m->cls[j]->biases[k],lr,momentum,mini_batch_size, m->cls[j]->d_biases[k],&m->cls[j]->d1_biases[k]);
                        if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                            update_batch_normalized_layer_nesterov(m->cls[j]->group_norm,m->cls[j]->n_kernels/m->cls[j]->group_norm_channels,lr,momentum,mini_batch_size);
                        }
                    }
                }
                else if(m->cls[j]->training_mode == EDGE_POPUP){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        
                        nesterov_momentum(&m->cls[j]->scores[k],lr,momentum,mini_batch_size, m->cls[j]->d_scores[k],&m->cls[j]->d1_scores[k]);
                                
                    }
                }
            }
        }
    }
}


/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_convolutional_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->training_mode != FREEZE_TRAINING){
            if(m->cls[j]->convolutional_flag == CONVOLUTION || m->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                if(m->cls[j]->training_mode == GRADIENT_DESCENT || m->cls[j]->training_mode == FREEZE_BIASES){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->cls[j]->channels; u++){
                            for(z = 0; z < m->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                    adam_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                                }
                                    
                            }
                        }
                        if(m->cls[j]->training_mode != FREEZE_BIASES)
                        adam_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                        if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                            update_batch_normalized_layer_adam(m->cls[j]->group_norm,m->cls[j]->n_kernels/m->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam);
                        }
                    }
                }
                else if(m->cls[j]->training_mode == EDGE_POPUP){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        
                        adam_algorithm(&m->cls[j]->scores[k], &m->cls[j]->d1_scores[k],&m->cls[j]->d2_scores[k], m->cls[j]->d_scores[k],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                                
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_convolutional_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->training_mode != FREEZE_TRAINING){
            if(m->cls[j]->convolutional_flag == CONVOLUTION || m->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                if(m->cls[j]->training_mode == GRADIENT_DESCENT || m->cls[j]->training_mode == FREEZE_BIASES){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->cls[j]->channels; u++){
                            for(z = 0; z < m->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                    adamod(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->cls[j]->d3_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w]);
                                }
                                    
                            }
                        }
                        if(m->cls[j]->training_mode != FREEZE_BIASES)
                        adamod(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->cls[j]->d3_biases[k]);
                        if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                            update_batch_normalized_layer_adamod(m->cls[j]->group_norm,m->cls[j]->n_kernels/m->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam,beta3_adamod);
                        }
                    }
                }
                else if(m->cls[j]->training_mode == EDGE_POPUP){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        
                        adamod(&m->cls[j]->scores[k], &m->cls[j]->d1_scores[k],&m->cls[j]->d2_scores[k], m->cls[j]->d_scores[k],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size, beta3_adamod,&m->cls[j]->d3_scores[k]);
                               
                    }
                }
            }
        }
    }
}

/* Given a model, this function update the params of the convolutional layers of the model with the adam diff grad optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_convolutional_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->training_mode != FREEZE_TRAINING){
            if(m->cls[j]->convolutional_flag == CONVOLUTION || m->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                if(m->cls[j]->training_mode == GRADIENT_DESCENT || m->cls[j]->training_mode == FREEZE_BIASES){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->cls[j]->channels; u++){
                            for(z = 0; z < m->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                    adam_diff_grad_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->cls[j]->d3_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w]);
                                }
                                    
                            }
                        }
                        if(m->cls[j]->training_mode != FREEZE_BIASES)
                        adam_diff_grad_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,m->cls[j]->d3_kernels[k]);
                        if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                            update_batch_normalized_layer_adam_diff_grad(m->cls[j]->group_norm,m->cls[j]->n_kernels/m->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam);
                        }
                    }
                }
                else if(m->cls[j]->training_mode == EDGE_POPUP){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                       
                        adam_diff_grad_algorithm(&m->cls[j]->scores[k], &m->cls[j]->d1_scores[k],&m->cls[j]->d2_scores[k], m->cls[j]->d_scores[k],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->cls[j]->d3_scores[k]);
                                
                    }
                }
            }
        }
    }
}
/* Given a model, this function update the params of the convolutional layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                   @ unsigned long long int t:= the number of time radam has been used
 * */
void update_convolutional_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->training_mode != FREEZE_TRAINING){
            if(m->cls[j]->convolutional_flag == CONVOLUTION || m->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                if(m->cls[j]->training_mode == GRADIENT_DESCENT  || m->cls[j]->training_mode == FREEZE_BIASES){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->cls[j]->channels; u++){
                            for(z = 0; z < m->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                    radam_algorithm(&m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], &m->cls[j]->d1_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],&m->cls[j]->d2_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w], m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                                }
                                    
                            }
                        }
                        if(m->cls[j]->training_mode != FREEZE_BIASES)
                        radam_algorithm(&m->cls[j]->biases[k],&m->cls[j]->d1_biases[k],&m->cls[j]->d2_biases[k], m->cls[j]->d_biases[k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                        if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                            update_batch_normalized_layer_radam(m->cls[j]->group_norm,m->cls[j]->n_kernels/m->cls[j]->group_norm_channels,lr,mini_batch_size,b1,b2,t,beta1_adam,beta2_adam);
                        }
                    }
                }
                
                else if(m->cls[j]->training_mode == EDGE_POPUP){
                    for(k = 0; k < m->cls[j]->n_kernels; k++){             
                        radam_algorithm(&m->cls[j]->scores[k], &m->cls[j]->d1_scores[k],&m->cls[j]->d2_scores[k], m->cls[j]->d_scores[k],lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);                             
                    }
                }
            }
        }
    }
}



/* Given a model, this function update the params of the fully-connected layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_fully_connected_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(m->fcls[i]->training_mode != FREEZE_TRAINING && m->fcls[i]->feed_forward_flag != ONLY_DROPOUT){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    if(m->fcls[i]->training_mode == GRADIENT_DESCENT || m->fcls[i]->training_mode == FREEZE_BIASES)
                    nesterov_momentum(&m->fcls[i]->weights[j*m->fcls[i]->input+k], lr, momentum, mini_batch_size, m->fcls[i]->d_weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k]);
                    if(m->fcls[i]->training_mode == EDGE_POPUP)
                    nesterov_momentum(&m->fcls[i]->scores[j*m->fcls[i]->input+k], lr, momentum, mini_batch_size, m->fcls[i]->d_scores[j*m->fcls[i]->input+k],&m->fcls[i]->d1_scores[j*m->fcls[i]->input+k]);
                }
                if(m->fcls[i]->training_mode == GRADIENT_DESCENT)
                nesterov_momentum(&m->fcls[i]->biases[j], lr, momentum, mini_batch_size, m->fcls[i]->d_biases[j],&m->fcls[i]->d1_biases[j]);
            }
            if(m->fcls[i]->normalization_flag == LAYER_NORMALIZATION)
                update_batch_normalized_layer_nesterov(&m->fcls[i]->layer_norm,1,lr,momentum,mini_batch_size);
        }
    }
}

void update_scaled_l2_norm_nesterov(scaled_l2_norm* l, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    if (l->training_mode == GRADIENT_DESCENT)
        nesterov_momentum(&l->learned_g, lr, momentum, mini_batch_size, l->d_learned_g,&l->d1_learned_g);
    
}


/* Given tot bns layers, this function update the params of the batch-normalized layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ bn** bns:= the model that must be updated
 *                @ int n_bn:= the number of bns
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 * 
 * */
void update_batch_normalized_layer_nesterov(bn** bns,int n_bn, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < n_bn; i++){
        if(bns[i]->training_mode == GRADIENT_DESCENT){
            for(j = 0; j < bns[i]->vector_dim; j++){
                nesterov_momentum(&bns[i]->gamma[j], lr, momentum, 1, bns[i]->d_gamma[j],&bns[i]->d1_gamma[j]);
                nesterov_momentum(&bns[i]->beta[j], lr, momentum, 1, bns[i]->d_beta[j],&bns[i]->d1_beta[j]);
            }
        }
    }
}


/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_fully_connected_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(m->fcls[i]->training_mode != FREEZE_TRAINING && m->fcls[i]->feed_forward_flag != ONLY_DROPOUT){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    if(m->fcls[i]->training_mode == GRADIENT_DESCENT || m->fcls[i]->training_mode == FREEZE_BIASES)
                    adam_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                    else if(m->fcls[i]->training_mode == EDGE_POPUP)
                    adam_algorithm(&m->fcls[i]->scores[j*m->fcls[i]->input+k],&m->fcls[i]->d1_scores[j*m->fcls[i]->input+k], &m->fcls[i]->d2_scores[j*m->fcls[i]->input+k], m->fcls[i]->d_scores[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                }
                if(m->fcls[i]->training_mode == GRADIENT_DESCENT)
                adam_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
            }
        
            if(m->fcls[i]->normalization_flag == LAYER_NORMALIZATION)
                update_batch_normalized_layer_adam(&m->fcls[i]->layer_norm,1,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam);
        }
    }
}

void update_scaled_l2_norm_adam(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    if (l->training_mode == GRADIENT_DESCENT)
        adam_algorithm(&l->learned_g,&l->d1_learned_g, &l->d2_learned_g, l->d_learned_g, lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
}
void update_scaled_l2_norm_radam(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam){
    int i,j,k;
    if (l->training_mode == GRADIENT_DESCENT)
        radam_algorithm(&l->learned_g,&l->d1_learned_g, &l->d2_learned_g, l->d_learned_g, lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
}

/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_fully_connected_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(m->fcls[i]->training_mode != FREEZE_TRAINING && m->fcls[i]->feed_forward_flag != ONLY_DROPOUT){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    if(m->fcls[i]->training_mode == GRADIENT_DESCENT  || m->fcls[i]->training_mode == FREEZE_BIASES)
                    adamod(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->fcls[i]->d3_weights[j*m->fcls[i]->input+k]);
                    else if(m->fcls[i]->training_mode == EDGE_POPUP)
                    adamod(&m->fcls[i]->scores[j*m->fcls[i]->input+k],&m->fcls[i]->d1_scores[j*m->fcls[i]->input+k], &m->fcls[i]->d2_scores[j*m->fcls[i]->input+k], m->fcls[i]->d_scores[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->fcls[i]->d3_scores[j*m->fcls[i]->input+k]);
                }
                if(m->fcls[i]->training_mode == GRADIENT_DESCENT)
                adamod(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->fcls[i]->d3_biases[j]);
            }
        
            if(m->fcls[i]->normalization_flag == LAYER_NORMALIZATION)
                update_batch_normalized_layer_adamod(&m->fcls[i]->layer_norm,1,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam,beta3_adamod);
        }
    }
}

void update_scaled_l2_norm_adamod(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod){
    int i,j,k;
    if (l->training_mode == GRADIENT_DESCENT)
        adamod(&l->learned_g,&l->d1_learned_g, &l->d2_learned_g, l->d_learned_g,lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&l->d3_learned_g);
}
/* Given a model, this function update the params of the fully-connected layers of the model with the adam diff grad optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_fully_connected_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(m->fcls[i]->training_mode != FREEZE_TRAINING && m->fcls[i]->feed_forward_flag != ONLY_DROPOUT){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    if(m->fcls[i]->training_mode == GRADIENT_DESCENT || m->fcls[i]->training_mode == FREEZE_BIASES)
                    adam_diff_grad_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->fcls[i]->d3_weights[j*m->fcls[i]->input+k]);
                    else if(m->fcls[i]->training_mode == EDGE_POPUP)
                    adam_diff_grad_algorithm(&m->fcls[i]->scores[j*m->fcls[i]->input+k],&m->fcls[i]->d1_scores[j*m->fcls[i]->input+k], &m->fcls[i]->d2_scores[j*m->fcls[i]->input+k], m->fcls[i]->d_scores[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->fcls[i]->d3_scores[j*m->fcls[i]->input+k]);
                }
                if(m->fcls[i]->training_mode == GRADIENT_DESCENT)
                adam_diff_grad_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->fcls[i]->d3_biases[j]);
            }
        
            if(m->fcls[i]->normalization_flag == LAYER_NORMALIZATION)
                update_batch_normalized_layer_adam_diff_grad(&m->fcls[i]->layer_norm,1,lr,mini_batch_size,b1,b2,beta1_adam,beta2_adam);
        }
    }
}

void update_scaled_l2_norm_adam_diff_grad(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    if (l->training_mode == GRADIENT_DESCENT)
        adam_diff_grad_algorithm(&l->learned_g,&l->d1_learned_g, &l->d2_learned_g, l->d_learned_g,lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&l->d3_learned_g);

}

/* Given a model, this function update the params of the fully-connected layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ model* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_fully_connected_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(m->fcls[i]->training_mode != FREEZE_TRAINING && m->fcls[i]->feed_forward_flag != ONLY_DROPOUT){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    if(m->fcls[i]->training_mode == GRADIENT_DESCENT || m->fcls[i]->training_mode == FREEZE_BIASES)
                        radam_algorithm(&m->fcls[i]->weights[j*m->fcls[i]->input+k],&m->fcls[i]->d1_weights[j*m->fcls[i]->input+k], &m->fcls[i]->d2_weights[j*m->fcls[i]->input+k], m->fcls[i]->d_weights[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                    else if(m->fcls[i]->training_mode == EDGE_POPUP)
                        radam_algorithm(&m->fcls[i]->scores[j*m->fcls[i]->input+k],&m->fcls[i]->d1_scores[j*m->fcls[i]->input+k], &m->fcls[i]->d2_scores[j*m->fcls[i]->input+k], m->fcls[i]->d_scores[j*m->fcls[i]->input+k], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);

                }
                if(m->fcls[i]->training_mode == GRADIENT_DESCENT)
                radam_algorithm(&m->fcls[i]->biases[j],&m->fcls[i]->d1_biases[j], &m->fcls[i]->d2_biases[j], m->fcls[i]->d_biases[j],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
            }
            
            if(m->fcls[i]->normalization_flag == LAYER_NORMALIZATION)
                update_batch_normalized_layer_radam(&m->fcls[i]->layer_norm,1,lr,mini_batch_size,b1,b2,t,beta1_adam,beta2_adam);
        }
    }
}




/* Given a bns** layers, this function update the params of the batch-normalized layers of the model with the adam diff grad optimization algorithm
 * 
 * Input:
 *             
 *             @ bn** bns:= batch_normalized layers
 *             @ int n_bn:= number of bn
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_batch_normalized_layer_adam_diff_grad(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < n_bn; i++){
        if(bns[i]->training_mode == GRADIENT_DESCENT){
            for(j = 0; j < bns[i]->vector_dim; j++){
                adam_diff_grad_algorithm(&bns[i]->gamma[j],&bns[i]->d1_gamma[j], &bns[i]->d2_gamma[j], bns[i]->d_gamma[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&bns[i]->d3_gamma[j]);
                adam_diff_grad_algorithm(&bns[i]->beta[j],&bns[i]->d1_beta[j], &bns[i]->d2_beta[j], bns[i]->d_beta[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&bns[i]->d3_beta[j]);  
            }
        }
    }
}
/* Given a bns** layers, this function update the params of the batch-normalized layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bn** bns:= batch_normalized layers
 *             @ int n_bn:= number of bn
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_batch_normalized_layer_adam(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < n_bn; i++){
        if(bns[i]->training_mode == GRADIENT_DESCENT){
            for(j = 0; j < bns[i]->vector_dim; j++){
                adam_algorithm(&bns[i]->gamma[j],&bns[i]->d1_gamma[j], &bns[i]->d2_gamma[j], bns[i]->d_gamma[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                adam_algorithm(&bns[i]->beta[j],&bns[i]->d1_beta[j], &bns[i]->d2_beta[j], bns[i]->d_beta[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);  
            }
        }
    }
}

/* Given a bns** layers, this function update the params of the batch-normalized layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bn** bns:= batch_normalized layers
 *             @ int n_bn:= number of bn
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 * 
 * */
void update_batch_normalized_layer_adamod(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod){
    int i,j,k;
    for(i = 0; i < n_bn; i++){
        if(bns[i]->training_mode == GRADIENT_DESCENT){
            for(j = 0; j < bns[i]->vector_dim; j++){
                adamod(&bns[i]->gamma[j],&bns[i]->d1_gamma[j], &bns[i]->d2_gamma[j], bns[i]->d_gamma[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size, beta3_adamod,&bns[i]->d3_gamma[j]);
                adamod(&bns[i]->beta[j],&bns[i]->d1_beta[j], &bns[i]->d2_beta[j], bns[i]->d_beta[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&bns[i]->d3_beta[j]);  
            }
        }
    }
}
/* Given tot bns layers, this function update the params of the batch-normalized layers of the model with the adam optimization algorithm
 * 
 * Input:
 *             
 *             @ bn** bns:= the bns layers
 *             @ int n_bn:= number of bns
 *             @ float lr:= the learning rate
 *             @ int mini_batch_size:= the size of the mini_batch
 *                @ float b1:= BETA1_ADAM^t
 *                @ float b2:= BETA2_ADAM^t
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_batch_normalized_layer_radam(bn** bns, int n_bn, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < n_bn; i++){
        if(bns[i]->training_mode == GRADIENT_DESCENT){
            for(j = 0; j < bns[i]->vector_dim; j++){
                radam_algorithm(&bns[i]->gamma[j],&bns[i]->d1_gamma[j], &bns[i]->d2_gamma[j], bns[i]->d_gamma[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                radam_algorithm(&bns[i]->beta[j],&bns[i]->d1_beta[j], &bns[i]->d2_beta[j], bns[i]->d_beta[j], lr, beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);  
            }
        }
    }
}



/* Given a rmodel, this function update the params of the lstm layers of the model with the nesterov momentum
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 * 
 * */
void update_lstm_layer_nesterov(rmodel* m, float lr, float momentum, int mini_batch_size){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->input_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    nesterov_momentum(&m->lstms[i]->w[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_w[j][k],&m->lstms[i]->d1_w[j][k]);      
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    nesterov_momentum(&m->lstms[i]->u_scores[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_u_scores[j][k],&m->lstms[i]->d1_u[j][k]);
                }
            }
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->output_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    nesterov_momentum(&m->lstms[i]->u[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_u[j][k],&m->lstms[i]->d1_u[j][k]);
                    if(k < m->lstms[i]->output_size && m->lstms[i]->training_mode != FREEZE_BIASES)
                        nesterov_momentum(&m->lstms[i]->biases[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_biases[j][k],&m->lstms[i]->d1_biases[j][k]);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    nesterov_momentum(&m->lstms[i]->u_scores[j][k],lr,momentum,mini_batch_size,m->lstms[i]->d_u_scores[j][k],&m->lstms[i]->d1_u[j][k]);
                }
            }
        }
    }
}

/* Given a rmodel, this function update the params of the lstm layers of the model with the adam diff grad algorithm
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 * 
 * */
void update_lstm_layer_adam_diff_grad(rmodel* m,float lr,int mini_batch_size,float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->input_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    adam_diff_grad_algorithm(&m->lstms[i]->w[j][k],&m->lstms[i]->d1_w[j][k],&m->lstms[i]->d2_w[j][k],m->lstms[i]->d_w[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->lstms[i]->d3_w[j][k]);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    adam_diff_grad_algorithm(&m->lstms[i]->w_scores[j][k],&m->lstms[i]->d1_w_scores[j][k],&m->lstms[i]->d2_w_scores[j][k],m->lstms[i]->d_w_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->lstms[i]->d3_w_scores[j][k]);
                }
            }
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->output_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    adam_diff_grad_algorithm(&m->lstms[i]->u[j][k],&m->lstms[i]->d1_u[j][k],&m->lstms[i]->d2_u[j][k],m->lstms[i]->d_u[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->lstms[i]->d3_u[j][k]);
                    if(k < m->lstms[i]->output_size && m->lstms[i]->training_mode != FREEZE_BIASES)
                        adam_diff_grad_algorithm(&m->lstms[i]->biases[j][k],&m->lstms[i]->d1_biases[j][k],&m->lstms[i]->d2_biases[j][k],m->lstms[i]->d_biases[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->lstms[i]->d3_biases[j][k]);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    adam_diff_grad_algorithm(&m->lstms[i]->u_scores[j][k],&m->lstms[i]->d1_u_scores[j][k],&m->lstms[i]->d2_u_scores[j][k],m->lstms[i]->d_u_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,&m->lstms[i]->d3_u_scores[j][k]);    
                }
            }
        }
    }
}

/* Given a rmodel, this function update the params of the lstm layers of the model with the adam algorithm
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 * 
 * */
void update_lstm_layer_adam(rmodel* m,float lr,int mini_batch_size,float b1, float b2, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->input_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    adam_algorithm(&m->lstms[i]->w[j][k],&m->lstms[i]->d1_w[j][k],&m->lstms[i]->d2_w[j][k],m->lstms[i]->d_w[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    adam_algorithm(&m->lstms[i]->w_scores[j][k],&m->lstms[i]->d1_w_scores[j][k],&m->lstms[i]->d2_w_scores[j][k],m->lstms[i]->d_w_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                }
            }
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->output_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    adam_algorithm(&m->lstms[i]->u[j][k],&m->lstms[i]->d1_u[j][k],&m->lstms[i]->d2_u[j][k],m->lstms[i]->d_u[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                    if(k < m->lstms[i]->output_size && m->lstms[i]->training_mode != FREEZE_BIASES)
                        adam_algorithm(&m->lstms[i]->biases[j][k],&m->lstms[i]->d1_biases[j][k],&m->lstms[i]->d2_biases[j][k],m->lstms[i]->d_biases[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    adam_algorithm(&m->lstms[i]->u_scores[j][k],&m->lstms[i]->d1_u_scores[j][k],&m->lstms[i]->d2_u_scores[j][k],m->lstms[i]->d_u_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size);
                }
            }
        }
    }
}

/* Given a rmodel, this function update the params of the lstm layers of the model with the adam algorithm
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 * 
 * */
void update_lstm_layer_adamod(rmodel* m,float lr,int mini_batch_size,float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->input_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    adamod(&m->lstms[i]->w[j][k],&m->lstms[i]->d1_w[j][k],&m->lstms[i]->d2_w[j][k],m->lstms[i]->d_w[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->lstms[i]->d3_w[j][k]);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    adamod(&m->lstms[i]->w_scores[j][k],&m->lstms[i]->d1_w_scores[j][k],&m->lstms[i]->d2_w_scores[j][k],m->lstms[i]->d_w_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->lstms[i]->d3_w[j][k]);
                }
            }
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->output_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    adamod(&m->lstms[i]->u[j][k],&m->lstms[i]->d1_u[j][k],&m->lstms[i]->d2_u[j][k],m->lstms[i]->d_u[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->lstms[i]->d3_u[j][k]);
                    if(k < m->lstms[i]->output_size && m->lstms[i]->training_mode != FREEZE_BIASES)
                        adamod(&m->lstms[i]->biases[j][k],&m->lstms[i]->d1_biases[j][k],&m->lstms[i]->d2_biases[j][k],m->lstms[i]->d_biases[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->lstms[i]->d3_biases[j][k]);
                }
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    adamod(&m->lstms[i]->u_scores[j][k],&m->lstms[i]->d1_u_scores[j][k],&m->lstms[i]->d2_u_scores[j][k],m->lstms[i]->d_u_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,beta3_adamod,&m->lstms[i]->d3_u[j][k]);
                }
            }
        }
    }
}
/* Given a rmodel, this function update the params of the lstm layers of the model with the adam algorithm
 * 
 * Input:
 *             
 *             @ rmodel* m:= the model that must be updated
 *             @ float lr:= the learning rate
 *             @ float momentum:= the momentum
 *                @ int mini_batch_size:= the mini batch dimensions
 *                @ unsigned long long int t:= the number of time radam has been used
 * */
void update_lstm_layer_radam(rmodel* m,float lr,int mini_batch_size,float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam){
    int i,j,k;
    for(i = 0; i < m->n_lstm; i++){
        for(j = 0; j < 4; j++){
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->input_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    radam_algorithm(&m->lstms[i]->w[j][k],&m->lstms[i]->d1_w[j][k],&m->lstms[i]->d2_w[j][k],m->lstms[i]->d_w[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                }
                
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    radam_algorithm(&m->lstms[i]->w_scores[j][k],&m->lstms[i]->d1_w_scores[j][k],&m->lstms[i]->d2_w_scores[j][k],m->lstms[i]->d_w_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                }
            }
            for(k = 0; k < m->lstms[i]->output_size*m->lstms[i]->output_size; k++){
                if(m->lstms[i]->training_mode == GRADIENT_DESCENT){
                    radam_algorithm(&m->lstms[i]->u[j][k],&m->lstms[i]->d1_u[j][k],&m->lstms[i]->d2_u[j][k],m->lstms[i]->d_u[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                    if(k < m->lstms[i]->output_size && m->lstms[i]->training_mode != FREEZE_BIASES)
                        radam_algorithm(&m->lstms[i]->biases[j][k],&m->lstms[i]->d1_biases[j][k],&m->lstms[i]->d2_biases[j][k],m->lstms[i]->d_biases[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                }
                
                else if(m->lstms[i]->training_mode == EDGE_POPUP){
                    radam_algorithm(&m->lstms[i]->u_scores[j][k],&m->lstms[i]->d1_u_scores[j][k],&m->lstms[i]->d2_u_scores[j][k],m->lstms[i]->d_u_scores[j][k],lr,beta1_adam,beta2_adam,b1,b2,EPSILON_ADAM,mini_batch_size,t);
                }
            }
        }
    }
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
void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda, unsigned long long int* t){
    if(m == NULL)
        return;
    
    lambda*=(float)mini_batch_size;
    
    if(regularization == L2_REGULARIZATION){
        add_l2_residual_layer(m,(double)total_number_weights,lambda);
        add_l2_convolutional_layer(m,(double)total_number_weights,lambda);
        add_l2_fully_connected_layer(m,(double)total_number_weights,lambda);
    }
    
    
    if(gradient_descent_flag == NESTEROV){    
        update_residual_layer_nesterov(m,lr,momentum,mini_batch_size);
        update_convolutional_layer_nesterov(m,lr,momentum,mini_batch_size);
        update_fully_connected_layer_nesterov(m,lr,momentum,mini_batch_size);
    }
    
    else if(gradient_descent_flag == ADAM){
        update_residual_layer_adam(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
        update_convolutional_layer_adam(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
        update_fully_connected_layer_adam(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
    }
    
    else if(gradient_descent_flag == RADAM){
        update_residual_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t,m->beta1_adam,m->beta2_adam);
        update_convolutional_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t,m->beta1_adam,m->beta2_adam);
        update_fully_connected_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t,m->beta1_adam,m->beta2_adam);
    }
    
    else if(gradient_descent_flag == DIFF_GRAD){
        update_residual_layer_adam_diff_grad(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
        update_convolutional_layer_adam_diff_grad(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
        update_fully_connected_layer_adam_diff_grad(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
    }
    
    else if(gradient_descent_flag == ADAMOD){
        update_residual_layer_adamod(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam,m->beta3_adamod);
        update_convolutional_layer_adamod(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam,m->beta3_adamod);
        update_fully_connected_layer_adamod(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam,m->beta3_adamod);

    }
    

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
void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda, unsigned long long int* t){
    if(m == NULL)
        return;
    
    int i,count = 0,count2 = 0,j,k = 0;
    
    for(i = 0; i < m->layers; i++){
        if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
            count++;
            count2+=m->lstms[i]->window/m->lstms[i]->n_grouped_cell;
        }
    }
    
    bn** bns = NULL;
    if(count){
        int n_bn = count2;
        bns = (bn**)malloc(sizeof(bn*)*count2);
        for(i = 0, k = 0; i < m->layers; i++){
            if(m->lstms[i]->norm_flag == GROUP_NORMALIZATION){
                for(j = 0; j < m->lstms[i]->window/m->lstms[i]->n_grouped_cell; j++, k++){
                    bns[k] = m->lstms[i]->bns[j];
                }
            }
        }
        
        if(gradient_descent_flag == NESTEROV)
            update_batch_normalized_layer_nesterov(bns,n_bn,lr,momentum,mini_batch_size);
        else if(gradient_descent_flag == ADAM)
            update_batch_normalized_layer_adam(bns,n_bn,lr,mini_batch_size,(*b1),(*b2),m->beta1_adam,m->beta2_adam);
        else if(gradient_descent_flag == RADAM)
            update_batch_normalized_layer_radam(bns,n_bn,lr,mini_batch_size,(*b1),(*b2),m->beta1_adam,m->beta2_adam,(*t));
        else if(gradient_descent_flag == DIFF_GRAD)
            update_batch_normalized_layer_adam_diff_grad(bns,n_bn,lr,mini_batch_size,(*b1),(*b2),m->beta1_adam,m->beta2_adam);
        else if(gradient_descent_flag == ADAMOD)
            update_batch_normalized_layer_adamod(bns,n_bn,lr,mini_batch_size,(*b1),(*b2),m->beta1_adam,m->beta2_adam,m->beta3_adamod);
        
        
        free(bns);
    }
    
    lambda*=(float)mini_batch_size;
    
    if(regularization == L2_REGULARIZATION)
        add_l2_lstm_layer(m,(double)total_number_weights,lambda);
    
    
    
    if(gradient_descent_flag == NESTEROV)
        update_lstm_layer_nesterov(m,lr,momentum,mini_batch_size);
    
    
    else if(gradient_descent_flag == ADAM){
        update_lstm_layer_adam(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
    }
    
    else if(gradient_descent_flag == RADAM){
        update_lstm_layer_radam(m,lr,mini_batch_size, (*b1), (*b2), *t,m->beta1_adam,m->beta2_adam);
    } 
    
    else if(gradient_descent_flag == DIFF_GRAD){
        update_lstm_layer_adam_diff_grad(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam);
    }
    
    else if(gradient_descent_flag == ADAMOD){
        update_lstm_layer_adamod(m,lr,mini_batch_size, (*b1), (*b2),m->beta1_adam,m->beta2_adam,m->beta3_adamod);
    }    
    

}

/* This function can update the model of the transformer using the adam algorithm or the nesterov momentum
 * 
 * Input:
 * 
 *             @ transformer* t:= the model that must be updated
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
void update_transformer(transformer* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda, unsigned long long int* time){
    int i;
    for(i = 0; i < t->n_te; i++){
        update_transformer_encoder(t->te[i],lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    }
    for(i = 0; i < t->n_td; i++){
        update_transformer_decoder(t->td[i],lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    }
    
    return;
}

/* This function can update the model of the encoder transformer using the adam algorithm or the nesterov momentum
 * 
 * Input:
 * 
 *             @ transformer_encoder* t:= the model that must be updated
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
void update_transformer_decoder(transformer_decoder* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda, unsigned long long int* time){
    fcl** fcls = t->e->m->fcls;
    cl** cls = t->e->m->cls;
    rl** rls = t->e->m->rls;
    int n_fcl = t->e->m->n_fcl, n_cl = t->e->m->n_cl, n_rl = t->e->m->n_rl, l = t->e->m->layers,i;
    
    update_model(t->e->m,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    
    if(gradient_descent_flag == NESTEROV){
        for(i = 0; i < t->n_l2+t->e->n_l2; i++){
            update_scaled_l2_norm_nesterov(t->l2[i],lr,momentum,mini_batch_size);
        }
    }
    
    if(gradient_descent_flag == ADAM){
        
        for(i = 0; i < t->n_l2+t->e->n_l2; i++){
            update_scaled_l2_norm_adam(t->l2[i],lr,mini_batch_size,*b1,*b2,t->e->m->beta1_adam,t->e->m->beta2_adam);
        }
    }
    
    else if(gradient_descent_flag == RADAM){

        for(i = 0; i < t->n_l2+t->e->n_l2; i++){
            update_scaled_l2_norm_radam(t->l2[i],lr,mini_batch_size,*b1,*b2,*time,t->e->m->beta1_adam,t->e->m->beta2_adam);
        }
    }
    
    else if(gradient_descent_flag == DIFF_GRAD){
        

        for(i = 0; i < t->n_l2+t->e->n_l2; i++){
            update_scaled_l2_norm_adam_diff_grad(t->l2[i],lr,mini_batch_size,*b1,*b2,t->e->m->beta1_adam,t->e->m->beta2_adam);
        }
    }
    
    else if(gradient_descent_flag == ADAMOD){
        

        for(i = 0; i < t->n_l2+t->e->n_l2; i++){
            update_scaled_l2_norm_adamod(t->l2[i],lr,mini_batch_size,*b1,*b2,t->e->m->beta1_adam,t->e->m->beta2_adam,t->e->m->beta3_adamod);
        }
    }
    
    t->e->m->fcls = t->fcls;
    t->e->m->cls = NULL; 
    t->e->m->rls = NULL;
    t->e->m->layers = 3*(t->n_head+t->e->n_head);
    t->e->m->n_cl = 0;
    t->e->m->n_rl = 0; 
    t->e->m->n_fcl = 3*(t->n_head+t->e->n_head);
    
    update_model(t->e->m,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    
    
    
    t->e->m->fcls = fcls;
    t->e->m->cls = cls; 
    t->e->m->rls = rls;
    t->e->m->layers = l;
    t->e->m->n_cl = n_cl;
    t->e->m->n_rl = n_rl; 
    t->e->m->n_fcl = n_fcl;
    
    update_model(t->e->linear_after_attention,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    
    
    
    
    update_model(t->linear_after_attention,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    
    
    
    
    return;
}

/* This function can update the model of the encoder transformer using the adam algorithm or the nesterov momentum
 * 
 * Input:
 * 
 *             @ transformer_encoder* t:= the model that must be updated
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
void update_transformer_encoder(transformer_encoder* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda, unsigned long long int* time){
    fcl** fcls = t->m->fcls;
    cl** cls = t->m->cls;
    rl** rls = t->m->rls;
    int n_fcl = t->m->n_fcl, n_cl = t->m->n_cl, n_rl = t->m->n_rl, l = t->m->layers,i;
    update_model(t->m,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    update_model(t->linear_after_attention,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    t->m->fcls = t->fcls;
    t->m->cls = NULL; 
    t->m->rls = NULL;
    t->m->layers = 3*t->n_head;
    t->m->n_cl = 0;
    t->m->n_rl = 0; 
    t->m->n_fcl = 3*t->n_head;
    
    update_model(t->m,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,time);
    
    if(gradient_descent_flag == NESTEROV){
        for(i = 0; i < t->n_l2; i++){
            update_scaled_l2_norm_nesterov(t->l2[i],lr,momentum,mini_batch_size);
        }
    }
    
    if(gradient_descent_flag == ADAM){
        (*b1)/=t->m->beta1_adam;
        (*b2)/=t->m->beta2_adam;
        for(i = 0; i < t->n_l2; i++){
            update_scaled_l2_norm_adam(t->l2[i],lr,mini_batch_size,*b1,*b2,t->m->beta1_adam,t->m->beta2_adam);
        }
    }
    
    else if(gradient_descent_flag == RADAM){

        for(i = 0; i < t->n_l2; i++){
            update_scaled_l2_norm_radam(t->l2[i],lr,mini_batch_size,*b1,*b2,*time,t->m->beta1_adam,t->m->beta2_adam);
        }
    }
    
    else if(gradient_descent_flag == DIFF_GRAD){
        

        for(i = 0; i < t->n_l2; i++){
            update_scaled_l2_norm_adam_diff_grad(t->l2[i],lr,mini_batch_size,*b1,*b2,t->m->beta1_adam,t->m->beta2_adam);
        }
    }
    
    else if(gradient_descent_flag == ADAMOD){

        for(i = 0; i < t->n_l2; i++){
            update_scaled_l2_norm_adamod(t->l2[i],lr,mini_batch_size,*b1,*b2,t->m->beta1_adam,t->m->beta2_adam,t->m->beta3_adamod);
        }
    }
    
    t->m->fcls = fcls;
    t->m->cls = cls; 
    t->m->rls = rls;
    t->m->layers = l;
    t->m->n_cl = n_cl;
    t->m->n_rl = n_rl; 
    t->m->n_fcl = n_fcl;
    
    return;
}

void update_vae_model(vaemodel* vm, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda, unsigned long long int* t){
    update_model(vm->encoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
    update_model(vm->decoder,lr,momentum,mini_batch_size,gradient_descent_flag,b1,b2,regularization,total_number_weights,lambda,t);
}


void update_training_parameters(float* beta1, float* beta2, long long unsigned int* time_step, float start_beta1, float start_beta2){
    (*beta1)*=start_beta1;
    (*beta2)*=start_beta2;
    (*time_step)++;
}
