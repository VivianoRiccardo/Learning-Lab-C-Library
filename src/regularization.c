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

/* This function add the l2 regularization to the partial derivative of the weights for residual layers of m
 * 
 * 
 * Input:
 *         
 *             @ model* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_residual_layer(model* m,double total_number_weights,float lambda){
    int i,j,k,u,z,w;
    for(i = 0; i < m->n_rl; i++){
        for(j = 0; j < m->rls[i]->n_cl; j++){
            if(m->rls[i]->cls[j]->convolutional_flag == CONVOLUTION || m->rls[i]->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
                if(m->rls[i]->cls[j]->training_mode == GRADIENT_DESCENT){
                    for(k = 0; k < m->rls[i]->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                            for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                    ridge_regression(&m->rls[i]->cls[j]->d_kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lambda, total_number_weights);
                                }
                            }
                        }
                    }
                    if(m->rls[i]->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                        for(k = 0; k < m->rls[i]->cls[j]->n_kernels/m->rls[i]->cls[j]->group_norm_channels; k++){
                            for(u = 0; u < m->rls[i]->cls[j]->group_norm[k]->vector_dim; u++){
                                ridge_regression(&(m->rls[i]->cls[j]->group_norm[k]->d_gamma[u]),m->rls[i]->cls[j]->group_norm[k]->gamma[u],lambda,total_number_weights);
                            }
                        }
                    }
                }
                else if(m->rls[i]->cls[j]->training_mode == EDGE_POPUP){
                    for(k = m->rls[i]->cls[j]->n_kernels*m->rls[i]->cls[j]->k_percentage; k < m->rls[i]->cls[j]->n_kernels; k++){
                        for(u = 0; u < m->rls[i]->cls[j]->channels; u++){
                            for(z = 0; z < m->rls[i]->cls[j]->kernel_rows; z++){
                                for(w = 0; w < m->rls[i]->cls[j]->kernel_cols; w++){
                                    ridge_regression(&m->rls[i]->cls[j]->d_scores[k],m->rls[i]->cls[j]->kernels[k][u*m->rls[i]->cls[j]->kernel_rows*m->rls[i]->cls[j]->kernel_cols + z*m->rls[i]->cls[j]->kernel_cols + w],lambda, total_number_weights);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


/* This function add the l2 regularization to the partial derivative of the weights for convolutional layers of m
 * 
 * 
 * Input:
 *         
 *             @ model* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_convolutional_layer(model* m,double total_number_weights,float lambda){
    int j,k,u,z,w;
    for(j = 0; j < m->n_cl; j++){
        if(m->cls[j]->convolutional_flag == CONVOLUTION || m->cls[j]->convolutional_flag == TRANSPOSED_CONVOLUTION){
            if(m->cls[j]->training_mode == GRADIENT_DESCENT){
                for(k = 0; k < m->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->cls[j]->channels; u++){
                        for(z = 0; z < m->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                ridge_regression(&m->cls[j]->d_kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lambda, total_number_weights);

                            }
                                
                        }
                    }
                }
                if(m->cls[j]->normalization_flag == GROUP_NORMALIZATION){
                    for(k = 0; k < m->cls[j]->n_kernels/m->cls[j]->group_norm_channels; k++){
                        for(u = 0; u < m->cls[j]->group_norm[k]->vector_dim; u++){
                            ridge_regression(&(m->cls[j]->group_norm[k]->d_gamma[u]),m->cls[j]->group_norm[k]->gamma[u],lambda,total_number_weights);
                        }
                    }
                }
            }
            else if(m->cls[j]->training_mode == EDGE_POPUP){
                for(k = 0; k < m->cls[j]->n_kernels; k++){
                    for(u = 0; u < m->cls[j]->channels; u++){
                        for(z = 0; z < m->cls[j]->kernel_rows; z++){
                            for(w = 0; w < m->cls[j]->kernel_cols; w++){
                                ridge_regression(&m->cls[j]->d_scores[k],m->cls[j]->kernels[k][u*m->cls[j]->kernel_rows*m->cls[j]->kernel_cols + z*m->cls[j]->kernel_cols + w],lambda, total_number_weights);

                            }
                                
                        }
                    }
                }
            }
        }
    }
}


/* This function add the l2 regularization to the partial derivative of the weights for fully-connected layers of m
 * 
 * 
 * Input:
 *         
 *             @ model* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_fully_connected_layer(model* m,double total_number_weights,float lambda){
    int i,j,k;
    for(i = 0; i < m->n_fcl; i++){
        if(m->fcls[i]->training_mode == GRADIENT_DESCENT){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    ridge_regression(&m->fcls[i]->d_weights[j*m->fcls[i]->input+k],m->fcls[i]->weights[j*m->fcls[i]->input+k],lambda, total_number_weights);

                }
            }
        }
        else if(m->fcls[i]->training_mode == EDGE_POPUP){
            for(j = 0; j < m->fcls[i]->output; j++){
                for(k = 0; k < m->fcls[i]->input; k++){
                    ridge_regression(&m->fcls[i]->d_scores[j*m->fcls[i]->input+k],m->fcls[i]->weights[j*m->fcls[i]->input+k],lambda, total_number_weights);

                }
            }
        }
    }
}


/* This function add the l2 regularization to the partial derivative of the weights for lstm layers of m
 * 
 * 
 * Input:
 *         
 *             @ rmodel* m:= the model
 *             @ int toal_number_weights:= the number of total weights
 *             @ float lambda an hyper param
 * 
 * */
void add_l2_lstm_layer(rmodel* m,double total_number_weights,float lambda){
    int j,k,u,z,w;
    for(j = 0; j < m->n_lstm; j++){
        if(m->lstms[j]->training_mode == GRADIENT_DESCENT){
            for(k = 0; k < 4; k++){
                for(u = 0; u < m->lstms[j]->output_size*m->lstms[j]->input_size; u++){
                    ridge_regression(&m->lstms[j]->d_w[k][u],m->lstms[j]->w[k][u],lambda,total_number_weights);
                }
                for(u = 0; u < m->lstms[j]->output_size*m->lstms[j]->output_size; u++){
                    ridge_regression(&m->lstms[j]->d_u[k][u],m->lstms[j]->u[k][u],lambda,total_number_weights);
                }
            }
        }
        else if(m->lstms[j]->training_mode == EDGE_POPUP){
            for(k = 0; k < 4; k++){
                for(u = 0; u < m->lstms[j]->output_size*m->lstms[j]->input_size; u++){
                    ridge_regression(&m->lstms[j]->d_w_scores[k][u],m->lstms[j]->w[k][u],lambda,total_number_weights);
                }
                for(u = 0; u < m->lstms[j]->output_size*m->lstms[j]->output_size; u++){
                    ridge_regression(&m->lstms[j]->d_u_scores[k][u],m->lstms[j]->u[k][u],lambda,total_number_weights);
                }
            }
        }
    }
}

