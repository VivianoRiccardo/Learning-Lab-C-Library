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


/* This function computes the feed forward of the wrapped encoder transformer inside the decoder transformer
 * indeed the last (possible) attention + reisudal + normalization + ff + residual + norm of the decoder can be seen
 * as an encoder inside the decoder, however the feed forward should be a little bit different at the beginning
 * because the attention mechanism takes the keys and queries from a precise input and the values from another one,
 * morover the first residual layer of this part only takes from the output of the first part of the decoder.
 * The difference in this sense is just that 
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ float* inputs2:= the inputs coming from the encoders for the attention, the input dimension1
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension:= the dimension of the inputs1
 *             @ int input_dimension1:= the dimension of the inputs2
 * */
void wrapped_encoder_transformer_decoder_ff(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension1,int input_dimension){
    int i;
    for(i = 0; i < t->n_head; i++){
        if(input_dimension1 != t->fcls[i*3]->input || input_dimension1 != t->fcls[i*3+1]->input || input_dimension != t->fcls[i*3+2]->input){
            fprintf(stderr,"Error: your fully connected layers don't match the inputs - inputs given: %d, %d, %d, inputs fcls: %d,%d,%d\n",input_dimension1,input_dimension1,input_dimension,t->fcls[i*3]->input,t->fcls[i*3+1]->input,t->fcls[i*3+2]->input);
            exit(0);
        }
        fully_connected_feed_forward(inputs2,&t->q[i*t->dimension],t->fcls[i*3]->weights,t->fcls[i*3]->biases,input_dimension1,t->dimension);
        fully_connected_feed_forward(inputs2,&t->k[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[i*3+1]->biases,input_dimension1,t->dimension);
        fully_connected_feed_forward(inputs1,&t->v[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[i*3+2]->biases,input_dimension,t->dimension);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        sum1D(inputs1,t->attention_output,t->residual1_output,input_dimension);
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(input_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual1_output,t->l2[0]->output);
            model_tensor_input_ff(t->m,1,input_dimension,1,t->l2[0]->output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                sum1D(t->l2[0]->output,t->m->output_layer,t->residual2_output,input_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
        
        else{
            model_tensor_input_ff(t->m,1,input_dimension,1,t->residual1_output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                sum1D(t->residual1_output,t->m->output_layer,t->residual2_output,input_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[0]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[0]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
    }
    
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(input_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->attention_output,t->l2[0]->output);
            model_tensor_input_ff(t->m,1,input_dimension,1,t->l2[0]->output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                sum1D(t->l2[0]->output,t->m->output_layer,t->residual2_output,input_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
        
        else{
            model_tensor_input_ff(t->m,1,input_dimension,1,t->attention_output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                sum1D(t->attention_output,t->m->output_layer,t->residual2_output,input_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[0]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(input_dimension,t->l2[0]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
    }
}


/* This function computes the back propagation of our encoder transformer network insde the decoder (see the description for the
 * ff wrapped endoer)
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ float* inputs2:= the inputs coming from the encoders, the input dimension1
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension:= the dimension of the inputs1
 *             @ int input_dimension1:= the dimension of the inputs2
 *                @ float* output_error:= the error for the output
 *               @float* encoder_error:= where will be stored the error for the encoders that fed the decoder
 * it returns the error for the inputs
 * */
float* wrapped_encoder_transformer_decoder_bp(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension1,int input_dimension,float* output_error,float* encoder_error){
    int i;
    float* temp;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
        if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
            back_propagation_scaled_l2_norm(t->l2[t->n_l2-1]->input_dimension,t->l2[t->n_l2-1]->learned_g,&t->l2[t->n_l2-1]->d_learned_g,t->l2[t->n_l2-1]->norm,t->residual2_output,output_error,t->l2[t->n_l2-1]->output_error);
        else    
            back_propagation_scaled_l2_norm(t->l2[t->n_l2-1]->input_dimension,t->l2[t->n_l2-1]->learned_g,&t->l2[t->n_l2-1]->d_learned_g,t->l2[t->n_l2-1]->norm,t->m->output_layer,output_error,t->l2[t->n_l2-1]->output_error);    
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->l2[0]->output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->input_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->attention_output,temp,t->l2[0]->output_error);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->l2[0]->output_error,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                fully_connected_back_prop(inputs2,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,encoder_error,t->fcls[i*3]->d_weights,t->fcls[i*3]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs2,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,encoder_error,t->fcls[i*3+1]->d_weights,t->fcls[i*3+1]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs1,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->attention_output_error,t->fcls[i*3+2]->d_weights,t->fcls[i*3+2]->d_biases,input_dimension,t->dimension);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->attention_output_error,t->l2[0]->output_error,t->attention_output_error,t->input_dimension);
            return t->attention_output_error;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->residual1_output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->attention_output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->input_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                copy_array(t->residual1_output_error,temp,t->input_dimension);
                            
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,temp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                fully_connected_back_prop(inputs2,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,encoder_error,t->fcls[i*3]->d_weights,t->fcls[i*3]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs2,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,encoder_error,t->fcls[i*3+1]->d_weights,t->fcls[i*3+1]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs1,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->attention_output_error,t->fcls[i*3+2]->d_weights,t->fcls[i*3+2]->d_biases,input_dimension,t->dimension);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->attention_output_error,t->residual1_output_error,t->attention_output_error,t->input_dimension);
            return t->attention_output_error;
        }
        
    }
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->l2[0]->output,output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->input_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->attention_output,temp,t->l2[0]->output_error);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->l2[0]->output_error,t->dimension,t->n_head,t->input_dimension,t->attention_flag);

            for(i = 0; i < t->n_head; i++){
                fully_connected_back_prop(inputs2,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,encoder_error,t->fcls[i*3]->d_weights,t->fcls[i*3]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs2,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,encoder_error,t->fcls[i*3+1]->d_weights,t->fcls[i*3+1]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs1,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->attention_output_error,t->fcls[i*3+2]->d_weights,t->fcls[i*3+2]->d_biases,input_dimension,t->dimension);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->attention_output_error,t->l2[0]->output_error,t->attention_output_error,t->input_dimension);
            return t->attention_output_error;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->residual1_output,output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->attention_output,output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->input_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                copy_array(t->residual1_output_error,temp,t->input_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,temp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);

            for(i = 0; i < t->n_head; i++){
                fully_connected_back_prop(inputs2,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,encoder_error,t->fcls[i*3]->d_weights,t->fcls[i*3]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs2,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,encoder_error,t->fcls[i*3+1]->d_weights,t->fcls[i*3+1]->d_biases,input_dimension1,t->dimension);
                fully_connected_back_prop(inputs1,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,encoder_error,t->fcls[i*3+2]->d_weights,t->fcls[i*3+2]->d_biases,input_dimension,t->dimension);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->attention_output_error,t->residual1_output_error,t->attention_output_error,t->input_dimension);
            return t->attention_output_error;
        }
        
    }
    
    
}
