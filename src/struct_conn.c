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



//rmodel stuff filled with all -1 where there is no need to connect
/* this function allocates space for a connection structure between 2 different strctures
 * we can also have 3 structures if concatenate flag is set in that case the v3 should be allocated
 * 
 * Inputs:
 *             // all the outputs except for v3 could be treated as input in case of cancatenation
 *             @ int id:= the indentifier of the structure
 *             @ model* m1:= the model in input
 *             @ model* m2:= the model in output
 *             @ rmodel* r1:= the rmodel in input
 *             @ rmodel* r2:= the rmodel in output
 *             @ transformer_encoder* e1: the encoder in input
 *             @ transformer_encoder* e2: the encoder in output
 *             @ transformer_decoder* d1: the decoder in input
 *             @ transformer_decoder* d2: the decoder in output
 *             @ transformer_encoder* t1: the transf in input
 *             @ transformer_encoder* t2: the transf in output    
 *             @ scaled_l2_norm* l1:= the scaled l2 norm in input
 *             @ scaled_l2_norm* l2:= the scaled l2 norm in output
 *             @ vector_struct* v1:= the vector struct in input
 *             @ vector_struct* v2:= the vector struct in output
 *             @ vector_struct* v3:= the vector struct in output in case of concatenation
 *             @ int input1_type:= is the input type in input(MODEL, RMODEL, TRANSFORMER_ENCODER, TRANSFORMER_DECODER, TRANSFORMER, L2_NORM_CONN, VECTOR, TEMPORAL_ENCODING_MODEL)
 *             @ int input2_type:= is the input type in input (if there is concatenate flag set to concatenate, in output otherwise)
 *             @ int output_type:= is used only if concatenate has been set (output must be set to VECTOR)
 *             @ int* input_temporal_index:= informations of indeces for the temporal model 
 *             @ int* input_encoder_indices:= informations of indeces for the encoder
 *             @ int* input_decoder_indices_left:= informations of indeces for the decoder for the left input
 *             @ int* input_decoder_indices_down:= informations of indeces for the decoder for the input below
 *             @ int* input_transf_encoder_indeces:= informations of indeces for the encoder of the transformer
 *             @ int* input_trasnf_decoder_indeces:= informations of indeces for the decoder of the transformer
 *             @ int* rmodel_input_left:= is an array of informations in case you want specifics inputs for the h in case you are using an rmodel in input2_type without concatenate
 *             @ int* rmodel_input_down:= same of left, but for the input down of the rmodel
 *             @ int decoder_left_input:= is the size of the left input for the decoder (in case of input2_type = DECODER, no concatenate)
 *             @ int decoder_down_input:= is the size of the input down of the decoder (in case of input2_type = DECODER, no concatenate)
 *             @ int transf_dec_input:= the input size for the decoder part in case input2_type = TRANSFORMER
 *             @ int transf_enc_input:= same of above but for encoder part
 *             @ int concatenate_flag:= (CONCATENATE | NO_CONCATENATE)
 *             @ int input_size:= the size of the model input
 *             @ int model_input_index:= the information for the input of the model respect the input1_type
 *             @ int temporal_encoding_model_size:= the size in case you must apply temporal encoding 
 *                @ int vector_index:= used in case of concatenation for the information regardings the input2_type
 * IN CASE TEMPORAL IS SET FOR INPUT1 TYPE THE **M MUST BE ASSIGNED BY A PREVIOSULY ALLOCATED MEMORY OF ANOTHER STRUCTURE_CONNECTION
 * */
struct_conn* structure_connection(int id, model* m1, model* m2, rmodel* r1, rmodel* r2, transformer_encoder* e1, transformer_encoder* e2, transformer_decoder* d1, transformer_decoder* d2, transformer* t1, transformer* t2, scaled_l2_norm* l1, scaled_l2_norm* l2, vector_struct* v1, vector_struct* v2, vector_struct* v3, int input1_type, int input2_type, int output_type, int* input_temporal_index,int* input_encoder_indeces, int* input_decoder_indeces_left, int* input_decoder_indeces_down, int* input_transf_encoder_indeces, int* input_transf_decoder_indeces, int* rmodel_input_left, int* rmodel_input_down, int decoder_left_input, int decoder_down_input, int transf_dec_input, int transf_enc_input, int concatenate_flag, int input_size, int model_input_index, int temporal_encoding_model_size, int vector_index){
    int i;
    
    struct_conn* s = (struct_conn*)malloc(sizeof(struct_conn));
    
    if(input1_type == MODEL && m1 == NULL){
        fprintf(stderr,"Error: you have a model as input you must pass a model too!\n");
        exit(1);
    }
    if(input2_type == MODEL && m2 == NULL){
        fprintf(stderr,"Error: you have a model as output you must pass a model too!\n");
        exit(1);
    }
    if(input1_type == RMODEL && r1 == NULL){
        fprintf(stderr,"Error: you have a rmodel as input you must pass a rmodel too!\n");
        exit(1);
    }
    if(input2_type == RMODEL && r2 == NULL){
        fprintf(stderr,"Error: you have a model as output you must pass a rmodel too!\n");
        exit(1);
    }
    if(input1_type == TRANSFORMER_ENCODER && e1 == NULL){
        fprintf(stderr,"Error: you have an encoder as input you must pass an encoder too!\n");
        exit(1);
    }
    if(input2_type == TRANSFORMER_ENCODER && e2 == NULL){
        fprintf(stderr,"Error: you have a encoder as output you must pass an encoder too!\n");
        exit(1);
    }
    if(input1_type == TRANSFORMER_DECODER && d1 == NULL){
        fprintf(stderr,"Error: you have a decoder as input you must pass a decoder too!\n");
        exit(1);
    }
    if(input2_type == TRANSFORMER_DECODER && d2 == NULL){
        fprintf(stderr,"Error: you have a DECODER as output you must pass a decoder too!\n");
        exit(1);
    }
    if(input1_type == TRANSFORMER && t1 == NULL){
        fprintf(stderr,"Error: you have a transformer as input you must pass a transformer too!\n");
        exit(1);
    }
    if(input2_type == TRANSFORMER && t2 == NULL){
        fprintf(stderr,"Error: you have a transformer as output you must pass a transformer too!\n");
        exit(1);
    }
    if(input1_type == L2_NORM_CONN && l1 == NULL){
        fprintf(stderr,"Error: you have a scaled l2 norm as input you must pass a l2 norm struct too!\n");
        exit(1);
    }
    if(input2_type == L2_NORM_CONN && l2 == NULL){
        fprintf(stderr,"Error: you have a scaled l2 norm as output you must pass a l2 norm struct too!\n");
        exit(1);
    }
    if(input1_type == VECTOR && v1 == NULL){
        fprintf(stderr,"Error: you have a scaled l2 norm as input you must pass a vector too!\n");
        exit(1);
    }
    if(input2_type == VECTOR && v2 == NULL){
        fprintf(stderr,"Error: you have a scaled l2 norm as input you must pass a vector too!\n");
        exit(1);
    }
    
    if(input2_type == RMODEL && (rmodel_input_down == NULL || rmodel_input_left == NULL)){
        fprintf(stderr,"Error: you have a rmodel here you must pass also the informations about the inputs\n");
        exit(1);
    }
    
    
    if(input2_type == TEMPORAL_ENCODING_MODEL && (m2 == NULL || !temporal_encoding_model_size)){
        fprintf(stderr,"Error: you must set m2 or temporal_encoding_model_size > 0\n");
        exit(1);
    }
    
    if(input1_type == TEMPORAL_ENCODING_MODEL && (m1 == NULL || !temporal_encoding_model_size)){
        fprintf(stderr,"Error: you must set m1 or temporal_encoding_model_size > 0\n");
        exit(1);
    }

    s->m1 = m1;
    s->m2 = m2;
    s->r1 = r1;
    s->r2 = r2;
    s->e1 = e1;
    s->e2 = e2;
    s->d1 = d1;
    s->d2 = d2;
    s->t1 = t1;
    s->t2 = t2;
    s->l1 = l1;
    s->l2 = l2;
    s->v1 = v1;
    s->v2 = v2;
    s->v3 = v3;
    
    s->concatenate_flag = concatenate_flag;
    s->id = id;
    s->input1_type = input1_type;
    s->input2_type = input2_type;
    s->output_type = output_type;
    s->vector_index = vector_index;
    
    s->decoder_left_input = decoder_left_input;
    s->decoder_down_input = decoder_down_input;
    s->transf_dec_input = transf_dec_input;
    s->transf_enc_input = transf_enc_input;
    s->rmodel_input_left = s->rmodel_input_left;// pointer
    s->rmodel_input_down = s->rmodel_input_down;// pointer
    s->model_input_index = model_input_index;
    s->input_size = input_size;// for model
    s->temporal_encoding_model_size = temporal_encoding_model_size;
    s->input_temporal_index = input_temporal_index;// pointer
    s->input_encoder_indeces = input_encoder_indeces;// pointer
    s->input_decoder_indeces_down = input_decoder_indeces_down;//pointer
    s->input_decoder_indeces_left = s->input_decoder_indeces_left;//pointer
    s->input_transf_encoder_indeces = input_transf_encoder_indeces;
    s->input_transf_decoder_indeces = input_transf_decoder_indeces;
    
    if(input2_type == RMODEL && concatenate_flag != CONCATENATE){
        s->h = (float**)malloc(sizeof(float*)*r2->n_lstm);
        s->c = (float**)malloc(sizeof(float*)*r2->n_lstm);
        s->inputs = (float**)malloc(sizeof(float*)*r2->lstms[0]->window);
        for(i = 0; i < r2->n_lstm; i++){
            s->h[i] = (float*)calloc(r2->lstms[i]->output_size,sizeof(float));
            s->c[i] = (float*)calloc(r2->lstms[i]->output_size,sizeof(float));
        }
        for(i = 0; i < r2->lstms[0]->window; i++){
            s->inputs[i] = (float*)calloc(r2->lstms[0]->input_size,sizeof(float));
        }
        s->encoder_input = NULL;
        s->temporal_m = NULL;
        s->temporal_m2 = NULL;
        s->decoder_input_down = NULL;
        s->decoder_input_left = NULL;
        s->transformer_input_encoder = NULL;
        s->transformer_input_decoder = NULL;
    }
    
    else if(input2_type == TRANSFORMER_ENCODER && concatenate_flag != CONCATENATE){
        s->encoder_input = (float*)calloc(s->transf_enc_input,sizeof(float));
        s->h = NULL;
        s->c = NULL;
        s->inputs = NULL;
        s->temporal_m = NULL;
        s->temporal_m2 = NULL;
        s->decoder_input_down = NULL;
        s->decoder_input_left = NULL;
        s->transformer_input_encoder = NULL;
        s->transformer_input_decoder = NULL;
    }
    
    else if(input2_type == TRANSFORMER_DECODER && concatenate_flag != CONCATENATE){
        s->encoder_input = NULL;
        s->h = NULL;
        s->c = NULL;
        s->inputs = NULL;
        s->temporal_m = NULL;
        s->temporal_m2 = NULL;
        s->transformer_input_encoder = NULL;
        s->transformer_input_decoder = NULL;
        s->decoder_input_down = (float*)calloc(s->decoder_down_input,sizeof(float));
        s->decoder_input_left = (float*)calloc(s->decoder_left_input,sizeof(float));
    }
    
    else if(input2_type == TRANSFORMER && concatenate_flag != CONCATENATE){
        s->encoder_input = NULL;
        s->h = NULL;
        s->c = NULL;
        s->inputs = NULL;
        s->temporal_m = NULL;
        s->temporal_m2 = NULL;
        s->transformer_input_encoder = (float*)calloc(transf_enc_input,sizeof(float));
        s->transformer_input_decoder = (float*)calloc(transf_dec_input,sizeof(float));
        s->decoder_input_down = NULL;
        s->decoder_input_left = NULL;
    }
    
        
    else if(input2_type == TEMPORAL_ENCODING_MODEL && concatenate_flag != CONCATENATE){
        s->temporal_m = (model**)malloc(sizeof(model*)*temporal_encoding_model_size);
        for(i = 0; i < temporal_encoding_model_size; i++){
            s->temporal_m[i] = copy_model_without_learning_parameters(m2);
        }
        s->temporal_m2 = NULL;
        s->h = NULL;
        s->c = NULL;
        s->inputs = NULL;
        s->encoder_input = NULL;
        s->decoder_input_down = NULL;
        s->decoder_input_left = NULL;
        s->transformer_input_encoder = NULL;
        s->transformer_input_decoder = NULL;
    }
    
    else{
        s->encoder_input = NULL;
        s->h = NULL;
        s->c = NULL;
        s->inputs = NULL;
        s->temporal_m = NULL;
        s->temporal_m2 = NULL;
        s->transformer_input_encoder = NULL;
        s->transformer_input_decoder = NULL;
        s->decoder_input_down = NULL;
        s->decoder_input_left = NULL;
    }
    return s;
}

void free_struct_conn(struct_conn* s){
    int i;
    if(s->input2_type == RMODEL && s->concatenate_flag != CONCATENATE){
        free_matrix((void**)s->h,s->r2->n_lstm);
        free_matrix((void**)s->c,s->r2->n_lstm);
        free_matrix((void**)s->inputs,s->r2->lstms[0]->window);
    }
    
    else if(s->input2_type ==TRANSFORMER_ENCODER && s->concatenate_flag != CONCATENATE){
        free(s->encoder_input);
    }
    else if(s->input2_type ==TRANSFORMER_DECODER && s->concatenate_flag != CONCATENATE){
        free(s->decoder_input_down);
        free(s->decoder_input_left);
    }
    
    else if(s->input2_type ==TRANSFORMER && s->concatenate_flag != CONCATENATE){
        free(s->transformer_input_decoder);
        free(s->transformer_input_encoder);
    }
    
    else if(s->input2_type == TEMPORAL_ENCODING_MODEL && s->concatenate_flag != CONCATENATE){
        for(i = 0; i < s->temporal_encoding_model_size; i++){
            free_model_without_learning_parameters(s->temporal_m[i]);
        }
        free(s->temporal_m);
    }
    free(s->rmodel_input_left);
    free(s->rmodel_input_down);
    free(s->input_temporal_index);
    free(s->input_encoder_indeces);
    free(s->input_decoder_indeces_down);
    free(s->input_decoder_indeces_left);
    free(s->input_transf_encoder_indeces);
    free(s->input_transf_decoder_indeces);
    free(s);
    return;
}

void reset_struct_conn(struct_conn* s){
    int i;
    if(s->input2_type == RMODEL && s->concatenate_flag != CONCATENATE){
        for(i = 0; i < s->r2->n_lstm; i++){
            set_vector_with_value(0,s->h[i],s->r2->lstms[i]->output_size);
            set_vector_with_value(0,s->c[i],s->r2->lstms[i]->output_size);
        }
        for(i = 0; i < s->r2->lstms[0]->window; i++){
            set_vector_with_value(0,s->inputs[i],s->r2->lstms[0]->input_size);
        }
    }
    
    else if(s->input2_type == TRANSFORMER_ENCODER && s->concatenate_flag != CONCATENATE){
        set_vector_with_value(0,s->encoder_input,s->transf_enc_input);
    }
    else if(s->input2_type == TRANSFORMER_DECODER && s->concatenate_flag != CONCATENATE){
        set_vector_with_value(0,s->decoder_input_down,s->decoder_down_input);
        set_vector_with_value(0,s->decoder_input_left,s->decoder_left_input);
    }
    
    else if(s->input2_type == TRANSFORMER && s->concatenate_flag != CONCATENATE){
        set_vector_with_value(0,s->transformer_input_decoder,s->transf_dec_input);
        set_vector_with_value(0,s->transformer_input_encoder,s->transf_enc_input);
    }
    
    else if(s->input2_type == TEMPORAL_ENCODING_MODEL && s->concatenate_flag != CONCATENATE){
        for(i = 0; i < s->temporal_encoding_model_size; i++){
            reset_model_without_learning_parameters(s->temporal_m[i]);
        }
    }    
}

// this function just copy the output from some structure in the input tensor of the rmodel
void struct_connection_input_arrays(struct_conn* s){
    if(s->input2_type == RMODEL && s->concatenate_flag != CONCATENATE){
        // this is the case when we have a rmodel in output and something else in input (can also be another rmodel)
        int i;
        if(s->input1_type == MODEL){
            int out_size = s->m1->output_dimension;
            float* output = s->m1->output_layer;
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->h[i],min(out_size,s->r2->lstms[i]->output_size));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(&output[s->rmodel_input_down[i]],s->inputs[i],min(out_size,s->r2->lstms[0]->input_size));
                }
            }
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(s->temporal_m[s->rmodel_input_left[i]]->output_layer,s->h[i],min(s->r2->lstms[i]->output_size,s->temporal_m[s->rmodel_input_left[i]]->output_dimension));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1)
                    copy_array(s->temporal_m[s->rmodel_input_left[i]]->output_layer,s->inputs[i],min(s->r2->lstms[0]->input_size,s->temporal_m[s->rmodel_input_left[i]]->output_dimension));
                
            }
        }
        
        else if(s->input1_type == RMODEL){
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(get_ith_output_cell(s->r1,s->rmodel_input_left[i]),s->h[i],min(s->r2->lstms[i]->output_size,s->r1->lstms[s->r1->n_lstm-1]->output_size));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(get_ith_output_cell(s->r1,i),s->inputs[i],min(s->r2->lstms[0]->input_size,s->r1->lstms[s->r1->n_lstm-1]->output_size));
                }
            }
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            float* output = get_output_layer_from_encoder_transf(s->e1);
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->h[i],min(s->r2->lstms[i]->output_size,s->e1->m->output_dimension));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->inputs[i],min(s->r2->lstms[0]->input_size,s->e1->m->output_dimension));
                }
            }
        }
        
        else if(s->input1_type == TRANSFORMER_DECODER){
            float* output = get_output_layer_from_encoder_transf(s->d1->e);
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->h[i],min(s->r2->lstms[i]->output_size,s->d1->e->m->output_dimension));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->inputs[i],min(s->r2->lstms[0]->input_size,s->d1->e->m->output_dimension));
                }
            }
        }
        else if(s->input1_type == TRANSFORMER){
            float* output = get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e);
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->h[i],min(s->r2->lstms[i]->output_size,s->t1->td[s->t1->n_td-1]->e->m->output_dimension));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->inputs[i],min(s->r2->lstms[0]->input_size,s->t1->td[s->t1->n_td-1]->e->m->output_dimension));
                }
            }
        }
        else if(s->input1_type == L2_NORM_CONN){
            float* output = s->l1->output;
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->h[i],min(s->r2->lstms[i]->output_size,s->l1->input_dimension));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->inputs[i],min(s->r2->lstms[0]->input_size,s->l1->input_dimension));
                }
            }
        }
        else if(s->input1_type == VECTOR){
            float* output = s->v1->output;
            int i;
            for(i = 0; i < s->r2->n_lstm; i++){
                if(s->rmodel_input_left[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->h[i],min(s->r2->lstms[i]->output_size,s->v1->output_size));
                }
            }
            for(i = 0; i < s->r2->lstms[0]->window; i++){
                if(s->rmodel_input_down[i] != -1){
                    copy_array(&output[s->rmodel_input_left[i]],s->inputs[i],min(s->r2->lstms[0]->input_size,s->v1->output_size));
                }
            }
        }
    }
    else if(s->input2_type == TRANSFORMER_ENCODER && s->concatenate_flag != CONCATENATE){
        int i,minimum;
        // this is the case when we have a rmodel in output and something else in input (can also be another rmodel)
        if(s->input1_type == MODEL){
            
            
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->m1->output_dimension);
                    copy_array(&s->m1->output_layer[s->input_encoder_indeces[i]],&s->encoder_input[i],minimum);
                }
            }
        }
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->temporal_m[s->input_encoder_indeces[i]]->output_dimension);
                    copy_array(s->temporal_m[s->input_encoder_indeces[i]]->output_layer,&s->encoder_input[i],minimum);
                }
            }
        }
        
        else if(s->input1_type == RMODEL){
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->r1->lstms[s->r1->n_lstm-1]->output_size);
                    copy_array(get_ith_output_cell(s->r1,s->input_encoder_indeces[i]),&s->encoder_input[i],minimum);
                }
            }
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            float* output = get_output_layer_from_encoder_transf(s->e1);
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->e1->m->output_dimension);
                    copy_array(output,&s->encoder_input[i],minimum);
                }
                
            }
        }
        
        else if(s->input1_type == TRANSFORMER_DECODER){
            float* output = get_output_layer_from_encoder_transf(s->d1->e);
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->d1->e->m->output_dimension);
                    copy_array(output,&s->encoder_input[i],minimum);
                }
            }
        }
        else if(s->input1_type == TRANSFORMER){
            float* output = get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e);
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->t1->td[s->t1->n_td-1]->e->m->output_dimension);
                    copy_array(output,&s->encoder_input[i],minimum);
                }
            }
        }
        else if(s->input1_type == L2_NORM_CONN){
            float* output = s->l1->output;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->l1->input_dimension);
                    copy_array(output,&s->encoder_input[i],minimum);
                }
            }
        }
        else if(s->input1_type == VECTOR){
            float* output = s->v1->output;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->v1->output_size);
                    copy_array(output,&s->encoder_input[i],minimum);
                }
            }
        }
    }
    else if(s->input2_type == TRANSFORMER_DECODER && s->concatenate_flag != CONCATENATE){
        int i,minimum;
        // this is the case when we have a rmodel in output and something else in input (can also be another rmodel)
        if(s->input1_type == MODEL){
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,s->m1->output_dimension);
                    copy_array(&s->m1->output_layer[s->input_decoder_indeces_down[i]],&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,s->m1->output_dimension);
                    copy_array(&s->m1->output_layer[s->input_decoder_indeces_left[i]],&s->decoder_input_left[i],minimum);
                }
            }
        }
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,s->temporal_m[s->input_decoder_indeces_down[i]]->output_dimension);
                    copy_array(s->temporal_m[s->input_decoder_indeces_down[i]]->output_layer,&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,s->temporal_m[s->input_decoder_indeces_left[i]]->output_dimension);
                    copy_array(s->temporal_m[s->input_decoder_indeces_left[i]]->output_layer,&s->decoder_input_left[i],minimum);
                }
            }
        }
        
        else if(s->input1_type == RMODEL){
            
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,s->r1->lstms[s->r1->n_lstm-1]->output_size);
                    copy_array(get_ith_output_cell(s->r1,s->input_decoder_indeces_down[i]),&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,s->r1->lstms[s->r1->n_lstm-1]->output_size);
                    copy_array(get_ith_output_cell(s->r1,s->input_decoder_indeces_left[i]),&s->decoder_input_left[i],minimum);
                }
            }
            
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            float* output = get_output_layer_from_encoder_transf(s->e1);
            int out_size = s->e1->m->output_dimension;
            
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_down[i]],&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_left[i]],&s->decoder_input_left[i],minimum);
                }
            }
        }
        
        else if(s->input1_type == TRANSFORMER_DECODER){
            float* output = get_output_layer_from_encoder_transf(s->d1->e);
            int out_size = s->d1->e->m->output_dimension;
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_down[i]],&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_left[i]],&s->decoder_input_left[i],minimum);
                }
            }
        }
        else if(s->input1_type == TRANSFORMER){
            float* output = get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e);
            int out_size = s->t1->td[s->t1->n_td-1]->e->m->output_dimension;
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_down[i]],&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_left[i]],&s->decoder_input_left[i],minimum);
                }
            }
        }
        else if(s->input1_type == L2_NORM_CONN){
            float* output = s->l1->output;
            int out_size = s->l1->input_dimension;
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_down[i]],&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_left[i]],&s->decoder_input_left[i],minimum);
                }
            }
        }
        else if(s->input1_type == VECTOR){
            float* output = s->v1->output;
            int out_size = s->v1->output_size;
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    minimum = min(s->decoder_down_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_down[i]],&s->decoder_input_down[i],minimum);
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    minimum = min(s->decoder_left_input-i,out_size);
                    copy_array(&output[s->input_decoder_indeces_left[i]],&s->decoder_input_left[i],minimum);
                }
            }
        }
    }
    else if(s->input2_type == TRANSFORMER && s->concatenate_flag != CONCATENATE){
        int i,minimum;
        // this is the case when we have a rmodel in output and something else in input (can also be another rmodel)
        if(s->input1_type == MODEL){
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->m1->output_dimension);
                    copy_array(&s->m1->output_layer[s->input_transf_encoder_indeces[i]],&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,s->m1->output_dimension);
                    copy_array(&s->m1->output_layer[s->input_transf_decoder_indeces[i]],&s->transformer_input_decoder[i],minimum);
                }
            }
        }
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->temporal_m[s->input_transf_encoder_indeces[i]]->output_dimension);
                    copy_array(s->temporal_m[s->input_transf_encoder_indeces[i]]->output_layer,&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,s->temporal_m[s->input_transf_decoder_indeces[i]]->output_dimension);
                    copy_array(s->temporal_m[s->input_transf_decoder_indeces[i]]->output_layer,&s->transformer_input_decoder[i],minimum);
                }
            }
        }
        
        else if(s->input1_type == RMODEL){
            
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,s->r1->lstms[s->r1->n_lstm-1]->output_size);
                    copy_array(get_ith_output_cell(s->r1,s->input_transf_encoder_indeces[i]),&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,s->r1->lstms[s->r1->n_lstm-1]->output_size);
                    copy_array(get_ith_output_cell(s->r1,s->input_transf_decoder_indeces[i]),&s->transformer_input_decoder[i],minimum);
                }
            }
            
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            float* output = get_output_layer_from_encoder_transf(s->e1);
            int out_size = s->e1->m->output_dimension;
            
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,out_size);
                    copy_array(&output[s->input_transf_encoder_indeces[i]],&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,out_size);
                    copy_array(&output[s->input_transf_decoder_indeces[i]],&s->transformer_input_decoder[i],minimum);
                }
            }
        }
        
        else if(s->input1_type == TRANSFORMER_DECODER){
            float* output = get_output_layer_from_encoder_transf(s->d1->e);
            int out_size = s->d1->e->m->output_dimension;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,out_size);
                    copy_array(&output[s->input_transf_encoder_indeces[i]],&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,out_size);
                    copy_array(&output[s->input_transf_decoder_indeces[i]],&s->transformer_input_decoder[i],minimum);
                }
            }
        }
        else if(s->input1_type == TRANSFORMER){
            float* output = get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e);
            int out_size = s->t1->td[s->t1->n_td-1]->e->m->output_dimension;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,out_size);
                    copy_array(&output[s->input_transf_encoder_indeces[i]],&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,out_size);
                    copy_array(&output[s->input_transf_decoder_indeces[i]],&s->transformer_input_decoder[i],minimum);
                }
            }
        }
        else if(s->input1_type == L2_NORM_CONN){
            float* output = s->l1->output;
            int out_size = s->l1->input_dimension;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,out_size);
                    copy_array(&output[s->input_transf_encoder_indeces[i]],&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,out_size);
                    copy_array(&output[s->input_transf_decoder_indeces[i] ],&s->transformer_input_decoder[i],minimum);
                }
            }
        }
        else if(s->input1_type == VECTOR){
            float* output = s->v1->output;
            int out_size = s->v1->output_size;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    minimum = min(s->transf_enc_input-i,out_size);
                    copy_array(&output[s->input_transf_encoder_indeces[i]],&s->transformer_input_encoder[i],minimum);
                }
            }
            for(i = 0; i < s->transf_dec_input; i++){
                if(s->input_transf_decoder_indeces[i] != -1){
                    minimum = min(s->transf_dec_input-i,out_size);
                    copy_array(&output[s->input_transf_decoder_indeces[i]],&s->transformer_input_decoder[i],minimum);
                }
            }
        }
    }
}

// the ith paramater is used in case we have a rmodel in input
void ff_struc_conn(struct_conn* s, int transformer_flag){
    int i;
    if(s->input2_type == MODEL && s->concatenate_flag != CONCATENATE){
        if(s->input1_type == MODEL){
            model_tensor_input_ff(s->m2,1,1,s->m1->output_dimension,&s->m1->output_layer[s->model_input_index]);
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            model_tensor_input_ff(s->m2,1,1,s->temporal_m[s->model_input_index]->output_dimension,s->temporal_m[s->model_input_index]->output_layer);
        }
        
        else if(s->input1_type == RMODEL){
            model_tensor_input_ff(s->m2,1,1,s->r1->lstms[s->r1->n_lstm-1]->output_size,get_ith_output_cell(s->r1,s->model_input_index));
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            model_tensor_input_ff(s->m2,1,1,s->input_size,&get_output_layer_from_encoder_transf(s->e1)[s->model_input_index]);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            model_tensor_input_ff(s->m2,1,1,s->input_size,&get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index]);
        }
        else if(s->input1_type == TRANSFORMER){
            model_tensor_input_ff(s->m2,1,1,s->input_size,&get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index]);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            model_tensor_input_ff(s->m2,1,1,s->input_size,&s->l1->output[s->model_input_index]);
        }
        
        else if(s->input1_type == VECTOR){
            model_tensor_input_ff(s->m2,1,1,s->input_size,&s->v1->output[s->model_input_index]);
        }
    }
    else if(s->input2_type == TEMPORAL_ENCODING_MODEL && s->concatenate_flag != CONCATENATE){
        float** inputs = (float**)malloc(sizeof(float*)*s->temporal_encoding_model_size);
        if(s->input1_type == MODEL){
            fprintf(stderr,"Error: you are using temporal model for encoding but previous structure is a model too, instead you should use temporal for transformer /encoder transformer / decoder transforemr / rnn\n");
            exit(1);
        }
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            fprintf(stderr,"Error: you have 2 temporal encoding model one after another: useless we are exiting!\n");
            exit(1);
        }
        
        else if(s->input1_type == RMODEL){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = get_ith_output_cell(s->r1,s->input_temporal_index[i]);
            }
            model_tensor_input_ff_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = &(get_output_layer_from_encoder_transf(s->e1)[s->input_temporal_index[i]]);
            }
            model_tensor_input_ff_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = &(get_output_layer_from_encoder_transf(s->d1->e)[s->input_temporal_index[i]]);
            }
            model_tensor_input_ff_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size);
        }
        else if(s->input1_type == TRANSFORMER){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = &(get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->input_temporal_index[i]]);
            }
            model_tensor_input_ff_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            fprintf(stderr,"Error: you are using a temporal model on a previous normalization, useless!\n");
            exit(1);
        }
        
        else if(s->input1_type == VECTOR){
            fprintf(stderr,"Error: you are using temporal model on a previous vector structure, useless!\n");
            exit(1);
        }
        free(inputs);
    }
    
    else if(s->input2_type == RMODEL && s->concatenate_flag != CONCATENATE){
        ff_rmodel(s->h,s->c,s->inputs,s->r2);
    }
    
    else if(s->input2_type == TRANSFORMER_ENCODER && s->concatenate_flag != CONCATENATE){
            encoder_transformer_ff(s->encoder_input,s->e2,s->transf_enc_input);
    }
    else if(s->input2_type == TRANSFORMER_DECODER && s->concatenate_flag != CONCATENATE){
            decoder_transformer_ff(s->decoder_input_down,s->decoder_input_left,s->d2,s->decoder_down_input,s->decoder_left_input);
    }
    else if(s->input2_type == TRANSFORMER && s->concatenate_flag != CONCATENATE){
            transf_ff(s->t2,s->transformer_input_encoder,s->transf_enc_input,s->transformer_input_decoder,s->transf_dec_input,transformer_flag);
    }
    
    else if(s->input2_type == L2_NORM_CONN && s->concatenate_flag != CONCATENATE){
        if(s->input1_type == MODEL){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,&s->m1->output_layer[s->model_input_index],s->l2->output);
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,s->temporal_m[s->model_input_index]->output_layer,s->l2->output);
        }
        
        else if(s->input1_type == RMODEL){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,get_ith_output_cell(s->r1,s->model_input_index),s->l2->output);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,&get_output_layer_from_encoder_transf(s->e1)[s->model_input_index],s->l2->output);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,&get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index],s->l2->output);
        }
        else if(s->input1_type == TRANSFORMER){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,&get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index],s->l2->output);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,&s->l1->output[s->model_input_index],s->l2->output);
        }
        
        else if(s->input1_type == VECTOR){
            feed_forward_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->norm,&s->v1->output[s->model_input_index],s->l2->output);
        }
    }
    else if(s->input2_type == VECTOR && s->concatenate_flag != CONCATENATE){
        if(s->input1_type == MODEL){
            ff_vector(&s->m1->output_layer[s->model_input_index],NULL,s->v2);
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            ff_vector(s->temporal_m[s->model_input_index]->output_layer,NULL,s->v2);
        }
        
        else if(s->input1_type == RMODEL){
            ff_vector(get_ith_output_cell(s->r1,s->model_input_index),NULL,s->v2);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            ff_vector(&get_output_layer_from_encoder_transf(s->e1)[s->model_input_index],NULL,s->v2);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            ff_vector(&get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index],NULL,s->v2);
        }
        else if(s->input1_type == TRANSFORMER){
            ff_vector(&get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index],NULL,s->v2);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            ff_vector(&s->l1->output[s->model_input_index],NULL,s->v2);
        }
        
        else if(s->input1_type == VECTOR){
            ff_vector(&s->v1->output[s->model_input_index],NULL,s->v2);
        }
    }
    
    else if(s->output_type == VECTOR && s->concatenate_flag == CONCATENATE){
        float* output1, *output2;
        if(s->input1_type == MODEL){
            output1 = &s->m1->output_layer[s->model_input_index];
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            output1 = s->temporal_m[s->model_input_index]->output_layer;
        }
        
        else if(s->input1_type == RMODEL){
            output1 = get_ith_output_cell(s->r1,s->model_input_index);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            output1 = &get_output_layer_from_encoder_transf(s->e1)[s->model_input_index];
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            output1 = &get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index];
        }
        else if(s->input1_type == TRANSFORMER){
            output1 = &get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index];
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            output1 = &s->l2->output[s->model_input_index];
        }
        
        else if(s->input1_type == VECTOR){
            output1 = &s->v2->output[s->model_input_index];
        }
        
        if(s->input2_type == MODEL){
            output2 = &s->m2->output_layer[s->vector_index];
        }
        
        else if(s->input2_type == TEMPORAL_ENCODING_MODEL){
            output2 = s->temporal_m2[s->vector_index]->output_layer;
        }
        
        else if(s->input2_type == RMODEL){
            output2 = get_ith_output_cell(s->r2,s->vector_index);
        }
        
        else if(s->input2_type == TRANSFORMER_ENCODER){
            output2 = &get_output_layer_from_encoder_transf(s->e2)[s->vector_index];
        }
        else if(s->input2_type == TRANSFORMER_DECODER){
            output2 = &get_output_layer_from_encoder_transf(s->d2->e)[s->vector_index];
        }
        else if(s->input2_type == TRANSFORMER){
            output2 = &get_output_layer_from_encoder_transf(s->t2->td[s->t2->n_td-1]->e)[s->vector_index];
        }
        
        else if(s->input2_type == L2_NORM_CONN){
            output2 = &s->l2->output[s->vector_index];
        }
        
        else if(s->input2_type == VECTOR){
            output2 = &s->v2->output[s->vector_index];
        }
        
        ff_vector(output1,output2,s->v3);
    }
}

// e is the error of the current model, es is the super struct where must be added the current returning error of the current model
void bp_struc_conn(struct_conn* s, int transformer_flag, error_super_struct* e, error_super_struct* es){
    
    int i,j;
    
    float* err;
    float** temp_err;
    
    if(s->input2_type == MODEL && s->concatenate_flag != CONCATENATE){
        
        
        error_handler* h = (error_handler*)malloc(sizeof(error_handler));
        h->ret_error = NULL;
        h->size = s->input_size;
        h->reference_index = s->model_input_index;
        h->free_flag_error = 0;
        es->n_error_handlers++;
        es->e = realloc(es->e,sizeof(error_handler*)*es->n_error_handlers);
        es->e[es->n_error_handlers-1] = h;
        
        err = (float*)calloc(s->m2->output_dimension,sizeof(float));
        for(j = 0; j < e->n_error_handlers; j++){
            sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->m2->output_dimension - e->e[j]->reference_index,e->e[j]->size));
        }
        if(s->input1_type == MODEL){
            h->ret_error = model_tensor_input_bp(s->m2,1,1,s->m1->output_dimension,&s->m1->output_layer[s->model_input_index], err,s->m2->output_dimension);
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            h->ret_error = model_tensor_input_bp(s->m2,1,1,s->temporal_m[s->model_input_index]->output_dimension,s->temporal_m[s->model_input_index]->output_layer,err,s->m2->output_dimension);
        }
        
        else if(s->input1_type == RMODEL){
            model_tensor_input_bp(s->m2,1,1,s->r1->lstms[s->r1->n_lstm-1]->output_size,get_ith_output_cell(s->r1,s->model_input_index),err,s->m2->output_dimension);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            model_tensor_input_bp(s->m2,1,1,s->input_size,&get_output_layer_from_encoder_transf(s->e1)[s->model_input_index],err,s->m2->output_dimension);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            model_tensor_input_bp(s->m2,1,1,s->input_size,&get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index],err,s->m2->output_dimension);
        }
        else if(s->input1_type == TRANSFORMER){
            model_tensor_input_bp(s->m2,1,1,s->input_size,&get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index],err,s->m2->output_dimension);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            model_tensor_input_bp(s->m2,1,1,s->input_size,&s->l1->output[s->model_input_index],err,s->m2->output_dimension);
        }
        
        else if(s->input1_type == VECTOR){
            model_tensor_input_bp(s->m2,1,1,s->input_size,&s->v1->output[s->model_input_index],err,s->m2->output_dimension);
        }
        free(err);
    }
    
    else if(s->input2_type == TEMPORAL_ENCODING_MODEL && s->concatenate_flag != CONCATENATE){
        
        
        
        
        float** inputs = (float**)malloc(sizeof(float*)*s->temporal_encoding_model_size);
        float** ret_temporal_error = (float**)malloc(sizeof(float*)*s->temporal_encoding_model_size);
        temp_err = (float**)malloc(s->temporal_encoding_model_size*sizeof(float*));
        
        for(j = 0; j < s->temporal_encoding_model_size; j++){
            temp_err[j] = (float*)calloc(s->temporal_m[j]->output_dimension,sizeof(float));
        }
        for(j = 0; j < e->n_error_handlers; j++){
            sum1D(temp_err[e->e[j]->reference_index],e->e[j]->ret_error,temp_err[e->e[j]->reference_index],min(s->temporal_m[e->e[j]->reference_index]->output_dimension,e->e[j]->size));
        }
        
        
        
        if(s->input1_type == MODEL){
            fprintf(stderr,"Error: you are using temporal model for encoding but previous structure is a model too, instead you should use temporal for transformer /encoder transformer / decoder transforemr / rnn\n");
            exit(1);
        }
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            fprintf(stderr,"Error: in consecutive mode 2 different temporal models, no sense here!\n");
            exit(1);
        }
        
        else if(s->input1_type == RMODEL){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = get_ith_output_cell(s->r1,s->input_temporal_index[i]);
            }
            model_tensor_input_bp_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size,temp_err,s->m2->output_dimension,ret_temporal_error);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = &(get_output_layer_from_encoder_transf(s->e1)[s->input_temporal_index[i]]);
            }
            model_tensor_input_bp_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size,temp_err,s->m2->output_dimension,ret_temporal_error);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = &(get_output_layer_from_encoder_transf(s->d1->e)[s->input_temporal_index[i]]);
            }
            model_tensor_input_bp_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size,temp_err,s->m2->output_dimension,ret_temporal_error);
        }
        else if(s->input1_type == TRANSFORMER){
            for(i = 0; i < s->temporal_encoding_model_size; i++){
                inputs[i] = &(get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->input_temporal_index[i]]);
            }
            model_tensor_input_bp_multicore_opt(s->temporal_m,s->m2,1,1,s->input_size,inputs,s->temporal_encoding_model_size,s->temporal_encoding_model_size,temp_err,s->m2->output_dimension,ret_temporal_error);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            fprintf(stderr,"Error: you are using a temporal model on a previous normalization, useless!\n");
            exit(1);
        }
        
        else if(s->input1_type == VECTOR){
            fprintf(stderr,"Error: you are using temporal model on a previous vector structure, useless!\n");
            exit(1);
        }
        
        
        error_handler** h = (error_handler**)malloc(sizeof(error_handler*)*s->temporal_encoding_model_size);
        for(j = 0; j < s->temporal_encoding_model_size; j++){
            h[j] = (error_handler*)malloc(sizeof(error_handler));
            h[j]->ret_error = NULL;
            h[j]->free_flag_error = 0;
            h[j]->size = s->input_size;
            h[j]->reference_index = s->input_temporal_index[j];
        }
        
        es->n_error_handlers+=s->temporal_encoding_model_size;
        es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
        for(j = es->n_error_handlers-s->temporal_encoding_model_size; j < es->n_error_handlers; j++){
            h[j-(es->n_error_handlers-s->temporal_encoding_model_size)]->ret_error = ret_temporal_error[j-(es->n_error_handlers-s->temporal_encoding_model_size)];
            es->e[j] = h[j-(es->n_error_handlers-s->temporal_encoding_model_size)];    
        }
        
        free(h);
        free_matrix((void**)temp_err,s->temporal_encoding_model_size);
        free(ret_temporal_error);
        free(inputs);
    }
    
    else if(s->input2_type == RMODEL && s->concatenate_flag != CONCATENATE){
        
        int n = 0;
        
        temp_err = (float**)malloc(sizeof(float*)*s->r2->lstms[s->r2->n_lstm-1]->window);
        float** inp_err = (float**)malloc(sizeof(float*)*s->r2->lstms[0]->window);
        for(j = 0; j < s->r2->lstms[s->r2->n_lstm-1]->window; j++){
            temp_err[j] = (float*)calloc(s->r2->lstms[s->r2->n_lstm-1]->output_size,sizeof(float));
        }
        for(j = 0; j < e->n_error_handlers; j++){
            sum1D(temp_err[e->e[j]->reference_index],e->e[j]->ret_error,temp_err[e->e[j]->reference_index],s->r2->lstms[s->r2->n_lstm-1]->output_size);
        }
        float*** ret = bp_rmodel(s->h,s->c,s->inputs,temp_err,s->r2,inp_err);
        
        
        for(j = 0; j < s->r2->n_lstm; j++){
            if(s->rmodel_input_left[j] != -1){
                n++;
            }
        }
        for(j = 0; j < s->r2->lstms[s->r2->n_lstm-1]->window; j++){
            if(s->rmodel_input_down[j] != -1){
                n++;
            }
        }
        
        error_handler** h = (error_handler**)malloc(sizeof(error_handler*)*n);
        for(j = 0; j < n; j++){
            h[j] = (error_handler*)malloc(sizeof(error_handler));
            h[j]->ret_error = NULL;
            h[j]->free_flag_error = 1;
        }
        
        int m = n;
        
        for(n = 0, j = 0; j < s->r2->n_lstm; j++){
            if(s->rmodel_input_left[j] != -1){
                h[n]->ret_error = lstm_dh(0,s->r2->lstms[j]->output_size,ret[j],s->r2->lstms[j]);
                h[n]->reference_index = s->rmodel_input_left[j];
                h[n]->size = s->r2->lstms[j]->output_size;
                n++;
            }
        }
        for(j = 0; j < s->r2->lstms[s->r2->n_lstm-1]->window; j++){
            if(s->rmodel_input_down[j] != -1){
                h[n]->ret_error = inp_err[j];
                h[n]->reference_index = s->rmodel_input_down[j];
                h[n]->size = s->r2->lstms[0]->input_size;
                n++;
            }
        }
        es->n_error_handlers+=m;
        es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
        for(j = es->n_error_handlers-m; j < es->n_error_handlers; j++){
            es->e[j] = h[j-(es->n_error_handlers-m)];    
        }
        free(h);
        free_matrix((void**)temp_err,s->r2->lstms[s->r2->n_lstm-1]->window);
        free(inp_err);
        free_tensor(ret,s->r2->n_lstm,4);
        
    }
    
    else if(s->input2_type == TRANSFORMER_ENCODER && s->concatenate_flag != CONCATENATE){
            
            int n = 0;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1) n++;
            }
            
            
            
            error_handler** h = (error_handler**)malloc(sizeof(error_handler*)*n);
            
            

            err = (float*)calloc(s->e2->m->output_dimension,sizeof(float));
            for(j = 0; j < e->n_error_handlers; j++){
                sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->e2->m->output_dimension-e->e[j]->reference_index,e->e[j]->size));
            }
            float* ret = encoder_transformer_bp(s->encoder_input,s->e2,s->transf_enc_input,err);
            for(i = 0, n = 0; i < s->transf_enc_input; i++){
                if(s->input_encoder_indeces[i] != -1){
                    h[n] = (error_handler*)malloc(sizeof(error_handler));
                    h[n]->free_flag_error = 0;
                    h[n]->size = s->transf_enc_input - i;
                    h[n]->ret_error = &ret[i];                    
                    n++;
                }
            }
            es->n_error_handlers+=n;
            es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
            for(j = es->n_error_handlers-n; j < es->n_error_handlers; j++){
                es->e[j] = h[j-(es->n_error_handlers-n)];    
            }
            free(h);
            free(err);
    }
    
    else if(s->input2_type == TRANSFORMER_DECODER && s->concatenate_flag != CONCATENATE){
            int n = 0, flag = 0;
            for(i = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1) n++;
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1) n++;
            }
            
            float* input2_error = (float*)calloc(s->decoder_left_input,sizeof(float));
            
            
            error_handler** h = (error_handler**)malloc(sizeof(error_handler*)*n);
            
            

            err = (float*)calloc(s->d2->e->m->output_dimension,sizeof(float));
            for(j = 0; j < e->n_error_handlers; j++){
                sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->d2->e->m->output_dimension-e->e[j]->reference_index,e->e[j]->size));
            }
            float* ret = decoder_transformer_bp(s->decoder_input_down,s->decoder_input_left,s->d2,s->decoder_down_input,s->decoder_left_input,err, input2_error);
            for(i = 0, n = 0; i < s->decoder_down_input; i++){
                if(s->input_decoder_indeces_down[i] != -1){
                    h[n] = (error_handler*)malloc(sizeof(error_handler));
                    h[n]->free_flag_error = 0;
                    h[n]->size = s->decoder_down_input - i;
                    h[n]->ret_error = &ret[i];                    
                    n++;
                }
            }
            for(i = 0; i < s->decoder_left_input; i++){
                if(s->input_decoder_indeces_left[i] != -1){
                    h[n] = (error_handler*)malloc(sizeof(error_handler));
                    if(!flag)
                    h[n]->free_flag_error = 1;
                    else
                    h[n]->free_flag_error = 0;
                    h[n]->size = s->decoder_left_input - i;
                    h[n]->ret_error = &input2_error[i];                    
                    n++;
                    flag = 1;
                }
            }
            es->n_error_handlers+=n;
            es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
            for(j = es->n_error_handlers-n; j < es->n_error_handlers; j++){
                es->e[j] = h[j-(es->n_error_handlers-n)];    
            }
            free(h);
            free(err);
    }
    else if(s->input2_type == TRANSFORMER && s->concatenate_flag != CONCATENATE){
            int n = 0;
            for(i = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1) n++;
            }
            
            
            
            error_handler** h = (error_handler**)malloc(sizeof(error_handler*)*n);
            
            

            err = (float*)calloc(s->e2->m->output_dimension,sizeof(float));
            for(j = 0; j < e->n_error_handlers; j++){
                sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->t2->td[s->t2->n_td-1]->e->m->output_dimension-e->e[j]->reference_index,e->e[j]->size));
            }
            float* ret = transf_bp(s->t2,s->transformer_input_encoder,s->transf_enc_input,s->transformer_input_decoder,s->transf_dec_input,err,transformer_flag);
            for(i = 0, n = 0; i < s->transf_enc_input; i++){
                if(s->input_transf_encoder_indeces[i] != -1){
                    h[n] = (error_handler*)malloc(sizeof(error_handler));
                    h[n]->free_flag_error = 0;
                    h[n]->size = s->transf_enc_input - i;
                    h[n]->ret_error = &ret[i];                    
                    n++;
                }
            }
            es->n_error_handlers+=n;
            es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
            for(j = es->n_error_handlers-n; j < es->n_error_handlers; j++){
                es->e[j] = h[j-(es->n_error_handlers-n)];    
            }
            free(h);
            free(err);
    }
    
    else if(s->input2_type == L2_NORM_CONN && s->concatenate_flag != CONCATENATE){
        
        err = (float*)calloc(s->l2->input_dimension,sizeof(float));
        for(j = 0; j < e->n_error_handlers; j++){
            sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->l2->input_dimension-e->e[j]->reference_index,e->e[j]->size));
        }
        
        if(s->input1_type == MODEL){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,&s->m1->output_layer[s->model_input_index],err,s->l2->output_error);
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,s->temporal_m[s->model_input_index]->output_layer,err,s->l2->output_error);
        }
        
        else if(s->input1_type == RMODEL){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,get_ith_output_cell(s->r1,s->model_input_index),err,s->l2->output_error);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,&get_output_layer_from_encoder_transf(s->e1)[s->model_input_index],err,s->l2->output_error);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,&get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index],err,s->l2->output_error);
        }
        else if(s->input1_type == TRANSFORMER){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,&get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index],err,s->l2->output_error);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,&s->l1->output[s->model_input_index],err,s->l2->output_error);
        }
        
        else if(s->input1_type == VECTOR){
            back_propagation_scaled_l2_norm(s->l2->input_dimension,s->l2->learned_g,&s->l2->d_learned_g,s->l2->norm,&s->v1->output[s->model_input_index],err,s->l2->output_error);
        }
        error_handler* h = (error_handler*)malloc(sizeof(error_handler));
        h->free_flag_error = 0;
        h->size = s->l2->input_dimension;
        h->ret_error = s->l2->output_error;
        h->reference_index = s->model_input_index;
        es->n_error_handlers++;
        es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
        es->e[es->n_error_handlers-1] = h;
        free(err);
        
    }
    else if(s->input2_type == VECTOR && s->concatenate_flag != CONCATENATE){
        
        float* ret;
        err = (float*)calloc(s->v2->output_size,sizeof(float));
        for(j = 0; j < e->n_error_handlers; j++){
            sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->v2->output_size-e->e[j]->reference_index,e->e[j]->size));
        }
        
        if(s->input1_type == MODEL){
            ret = bp_vector(&s->m1->output_layer[s->model_input_index],NULL,s->v2,err);
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            ret = bp_vector(s->temporal_m[s->model_input_index]->output_layer,NULL,s->v2,err);
        }
        
        else if(s->input1_type == RMODEL){
            ret = bp_vector(get_ith_output_cell(s->r1,s->model_input_index),NULL,s->v2,err);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            
            ret = bp_vector(&get_output_layer_from_encoder_transf(s->e1)[s->model_input_index],NULL,s->v2,err);
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            ret = bp_vector(&get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index],NULL,s->v2,err);
        }
        else if(s->input1_type == TRANSFORMER){
            ret = bp_vector(&get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index],NULL,s->v2,err);
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            ret = bp_vector(&s->l1->output[s->model_input_index],NULL,s->v2,err);
        }
        
        else if(s->input1_type == VECTOR){
            ret = bp_vector(&s->v1->output[s->model_input_index],NULL,s->v2,err);
        }
        
        error_handler* h = (error_handler*)malloc(sizeof(error_handler));
        h->free_flag_error = 0;
        h->size = s->l2->input_dimension;
        h->ret_error = ret;
        h->reference_index = s->model_input_index;
        es->n_error_handlers++;
        es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
        es->e[es->n_error_handlers-1] = h;
        free(err);
        
    }
    
    else if(s->output_type == VECTOR && s->concatenate_flag == CONCATENATE){
        float* output1, *output2;
        
        float* ret;
        err = (float*)calloc(s->v3->output_size,sizeof(float));
        for(j = 0; j < e->n_error_handlers; j++){
            sum1D(&err[e->e[j]->reference_index],e->e[j]->ret_error,&err[e->e[j]->reference_index],min(s->v3->output_size-e->e[j]->reference_index,e->e[j]->size));
        }
        
        if(s->input1_type == MODEL){
            output1 = &s->m1->output_layer[s->model_input_index];
        }
        
        else if(s->input1_type == TEMPORAL_ENCODING_MODEL){
            output1 = s->temporal_m[s->model_input_index]->output_layer;
        }
        
        else if(s->input1_type == RMODEL){
            output1 = get_ith_output_cell(s->r1,s->model_input_index);
        }
        
        else if(s->input1_type == TRANSFORMER_ENCODER){
            output1 = &get_output_layer_from_encoder_transf(s->e1)[s->model_input_index];
        }
        else if(s->input1_type == TRANSFORMER_DECODER){
            output1 = &get_output_layer_from_encoder_transf(s->d1->e)[s->model_input_index];
        }
        else if(s->input1_type == TRANSFORMER){
            output1 = &get_output_layer_from_encoder_transf(s->t1->td[s->t1->n_td-1]->e)[s->model_input_index];
        }
        
        else if(s->input1_type == L2_NORM_CONN){
            output1 = &s->l2->output[s->model_input_index];
        }
        
        else if(s->input1_type == VECTOR){
            output1 = &s->v2->output[s->model_input_index];
        }
        
        if(s->input2_type == MODEL){
            output2 = &s->m2->output_layer[s->vector_index];
        }
        
        else if(s->input2_type == TEMPORAL_ENCODING_MODEL){
            output2 = s->temporal_m2[s->vector_index]->output_layer;
        }
        
        else if(s->input2_type == RMODEL){
            output2 = get_ith_output_cell(s->r2,s->vector_index);
        }
        
        else if(s->input2_type == TRANSFORMER_ENCODER){
            output2 = &get_output_layer_from_encoder_transf(s->e2)[s->vector_index];
        }
        else if(s->input2_type == TRANSFORMER_DECODER){
            output2 = &get_output_layer_from_encoder_transf(s->d2->e)[s->vector_index];
        }
        else if(s->input2_type == TRANSFORMER){
            output2 = &get_output_layer_from_encoder_transf(s->t2->td[s->t2->n_td-1]->e)[s->vector_index];
        }
        
        else if(s->input2_type == L2_NORM_CONN){
            output2 = &s->l2->output[s->vector_index];
        }
        
        else if(s->input2_type == VECTOR){
            output2 = &s->v2->output[s->vector_index];
        }
        
        ret = bp_vector(output1,output2,s->v3,err);
        
        error_handler** h = (error_handler**)malloc(sizeof(error_handler*)*2);
        h[0]->free_flag_error = 0;
        h[1]->free_flag_error = 0;
        h[0]->size = s->v3->index;
        h[1]->size = s->v3->output_size-s->v3->index;;
        h[0]->ret_error = ret;
        h[1]->ret_error = &ret[s->v3->index];
        h[0]->reference_index = s->model_input_index;
        h[1]->reference_index = s->vector_index;
        es->n_error_handlers+=2;
        es->e = realloc(es->e,sizeof(error_handler)*es->n_error_handlers);
        es->e[es->n_error_handlers-2] = h[0];
        es->e[es->n_error_handlers-1] = h[1];
        free(h);
        free(err);
        
    }
}

