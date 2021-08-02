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



struct_conn_handler* init_mother_of_all_structs(int n_inputs, int n_models,int n_rmodels,int n_encoders,int n_decoders,int n_transformers,int n_l2s,int n_vectors,int n_total_structures,int n_struct_conn,int n_targets, model** m, rmodel** r, transformer_encoder** e,transformer_decoder** d,transformer** t,scaled_l2_norm** l2,vector_struct** v,struct_conn** s, int** models, int** rmodels,int** encoders,int** decoders,int** transformers,int** l2s,int** vectors,float** targets,int* targets_index,int* targets_error_flag,float** targets_weights,float* targets_threshold1,float* targets_threshold2,float* targets_gamma, int* targets_size){
    struct_conn_handler* sch = (struct_conn_handler*)malloc(sizeof(struct_conn_handler));
    sch->n_inputs = n_inputs;
    sch->n_models = n_models;
    sch->n_rmodels = n_rmodels;
    sch->n_encoders = n_encoders;
    sch->n_decoders = n_decoders;
    sch->n_transformers = n_transformers;
    sch->n_l2s = n_l2s;
    sch->n_vectors = n_vectors;
    sch->n_total_structures = n_total_structures;
    sch->n_struct_conn = n_struct_conn;
    sch->n_targets = n_targets;
    sch->m = m;
    sch->r = r;
    sch->e = e;
    sch->d = d;
    sch->t = t;
    sch->l2 = l2;
    sch->v = v;
    sch->s = s;
    sch->models = models;
    sch->rmodels = rmodels;
    sch->encoders = encoders;
    sch->decoders = decoders;
    sch->transformers = transformers;
    sch->l2s = l2s;
    sch->vectors = vectors;
    sch->targets = targets;
    sch->targets_index = targets_index;
    sch->targets_size = targets_size;
    sch->targets_error_flag = targets_error_flag;
    sch->targets_weights = targets_weights;
    sch->targets_threshold1 = targets_threshold1;
    sch->targets_threshold2 = targets_threshold2;
    sch->targets_gamma = targets_gamma;
    return sch;
}

void free_struct_conn_handler(struct_conn_handler* s){
    int i;
    for(i = 0; i < s->n_models; i++){
        free_model(s->m[i]);
    }
    for(i = 0; i < s->n_rmodels; i++){
        free_rmodel(s->r[i]);
    }
    for(i = 0; i < s->n_encoders; i++){
        free_transformer_encoder_layer(s->e[i]);
    }
    for(i = 0; i < s->n_decoders; i++){
        free_transformer_decoder_layer(s->d[i]);
    }
    for(i = 0; i < s->n_transformers; i++){
        free_transf(s->t[i]);
    }
    for(i = 0; i < s->n_l2s; i++){
        free_scaled_l2_normalization_layer(s->l2[i]);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        free_struct_conn(s->s[i]);
    }
    
    for(i = 0; i < s->n_vectors; i++){
        free_vector(s->v[i]);
    }
    
    for(i = 0; i < s->n_targets; i++){
        free(s->targets_weights[i]);
    }
    free(s->targets);
    free(s->targets_index);
    free(s->targets_size);
    free(s->targets_weights);
    free(s->targets_error_flag);
    free(s->targets_threshold1);
    free(s->targets_threshold2);
    free(s->targets_gamma);
    free(s->m);
    free(s->r);
    free(s->e);
    free(s->d);
    free(s->t);
    free(s->l2);
    free(s->v);
    free(s->s);
    free(s);
    return;
}

void free_struct_conn_handler_without_learning_parameters(struct_conn_handler* s){
    int i;
    for(i = 0; i < s->n_models; i++){
        free_model_without_learning_parameters(s->m[i]);
    }
    for(i = 0; i < s->n_rmodels; i++){
        free_rmodel_without_learning_parameters(s->r[i]);
    }
    for(i = 0; i < s->n_encoders; i++){
        free_transformer_encoder_layer_without_learning_parameters(s->e[i]);
    }
    for(i = 0; i < s->n_decoders; i++){
        free_transformer_decoder_layer_without_learning_parameters(s->d[i]);
    }
    for(i = 0; i < s->n_transformers; i++){
        free_transf_without_learning_parameters(s->t[i]);
    }
    for(i = 0; i < s->n_l2s; i++){
        free_scaled_l2_normalization_layer(s->l2[i]);
    }
    
    for(i = 0; i < s->n_vectors; i++){
        free_vector(s->v[i]);
    }
    
    for(i = 0; i < s->n_struct_conn; i++){
        free_struct_conn(s->s[i]);
    }
    
    free(s->m);
    free(s->r);
    free(s->e);
    free(s->d);
    free(s->t);
    free(s->l2);
    free(s->v);
    free(s->s);
    free(s);
    return;
}

struct_conn_handler* copy_struct_conn_handler(struct_conn_handler* s){
    int i,j;
    int m1,m2,r1,r2,e1,e2,d1,d2,t1,t2,v1,v2,l1,l22,v3, counter;
    model** m = NULL;
    rmodel** r = NULL;
    transformer_encoder** e = NULL;
    transformer_decoder** d = NULL;
    transformer** t = NULL;
    scaled_l2_norm** l2 = NULL;
    vector_struct** v = NULL;
    struct_conn** sc = NULL;
    int** models = NULL;
    int** rmodels = NULL;
    int** encoders = NULL;
    int** decoders = NULL;
    int** transformers = NULL;
    int** l2s = NULL;
    int** vectors = NULL;
    float** targets = NULL;
    int* targets_index = NULL;
    int* targets_size = NULL;
    int* targets_error_flag = NULL;
    float** targets_weights = NULL;
    float* targets_threshold1 = NULL;
    float* targets_threshold2 = NULL;
    float* targets_gamma = NULL;
    
    if(s->m != NULL){
        m = (model**)malloc(sizeof(model*)*s->n_models);
        models = (int**)malloc(sizeof(int*)*s->n_models);
        for(i = 0; i < s->n_models; i++){
            m[i] = copy_model(s->m[i]);
            models[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->models[i],models[i],s->n_struct_conn);
        }
    }
    if(s->r != NULL){
        r = (rmodel**)malloc(sizeof(rmodel*)*s->n_rmodels);
        rmodels = (int**)malloc(sizeof(int*)*s->n_rmodels);
        for(i = 0; i < s->n_rmodels; i++){
            r[i] = copy_rmodel(s->r[i]);
            rmodels[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->rmodels[i],rmodels[i],s->n_struct_conn);
        }
    }
    if(s->e != NULL){
        e = (transformer_encoder**)malloc(sizeof(transformer_encoder*)*s->n_encoders);
        encoders = (int**)malloc(sizeof(int*)*s->n_encoders);
        for(i = 0; i < s->n_encoders; i++){
            e[i] = copy_transformer_encoder(s->e[i]);
            encoders[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->encoders[i],encoders[i],s->n_struct_conn);
        }
    }
    if(s->d != NULL){
        d = (transformer_decoder**)malloc(sizeof(transformer_decoder*)*s->n_decoders);
        decoders = (int**)malloc(sizeof(int*)*s->n_decoders);
        for(i = 0; i < s->n_decoders; i++){
            d[i] = copy_transformer_decoder(s->d[i]);
            decoders[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->decoders[i],decoders[i],s->n_struct_conn);
        }
    }
    if(s->t != NULL){
        t = (transformer**)malloc(sizeof(transformer*)*s->n_transformers);
        transformers = (int**)malloc(sizeof(int*)*s->n_transformers);
        for(i = 0; i < s->n_transformers; i++){
            t[i] = copy_transf(s->t[i]);
            transformers[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->transformers[i],transformers[i],s->n_struct_conn);
        }
    }
    if(s->l2 != NULL){
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*s->n_l2s);
        l2s = (int**)malloc(sizeof(int*)*s->n_l2s);
        for(i = 0; i < s->n_l2s; i++){
            l2[i] = copy_scaled_l2_norm(s->l2[i]);
            l2s[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->l2s[i],l2s[i],s->n_struct_conn);
        }
    }
    if(s->v != NULL){
        v = (vector_struct**)malloc(sizeof(vector_struct*)*s->n_vectors);
        vectors = (int**)malloc(sizeof(int*)*s->n_vectors);
        for(i = 0; i < s->n_vectors; i++){
            v[i] = copy_vector(s->v[i]);
            vectors[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->vectors[i],vectors[i],s->n_struct_conn);
        }
    }
    
    if(s->targets != NULL){
        targets = (float**)malloc(sizeof(float*)*s->n_targets);
        targets_weights = (float**)malloc(sizeof(float*)*s->n_targets);
        targets_index = (int*)malloc(sizeof(int)*s->n_targets);
        targets_size = (int*)malloc(sizeof(int)*s->n_targets);
        targets_error_flag = (int*)malloc(sizeof(int)*s->n_targets);
        targets_threshold1 = (float*)malloc(sizeof(float)*s->n_targets);
        targets_threshold2 = (float*)malloc(sizeof(float)*s->n_targets);
        targets_gamma = (float*)malloc(sizeof(float)*s->n_targets);
        copy_int_array(s->targets_index,targets_index,s->n_targets);
        copy_int_array(s->targets_size,targets_size,s->n_targets);
        copy_int_array(s->targets_error_flag,targets_error_flag,s->n_targets);
        copy_array(s->targets_threshold1,targets_threshold1,s->n_targets);
        copy_array(s->targets_threshold2,targets_threshold2,s->n_targets);
        copy_array(s->targets_gamma,targets_gamma,s->n_targets);
        for(i = 0; i < s->n_targets; i++){
            targets_weights[i] = (float*)calloc(s->targets_size[i],sizeof(float));
            copy_array(s->targets_weights[i],targets_weights[i],s->targets_size[i]);
        }
    }
    
    sc = (struct_conn**)malloc(sizeof(struct_conn*)*s->n_struct_conn);
    
    for(j = 0; j < s->n_struct_conn; j++){
        counter = 0;
        m1 = -1;m2 = -1;r1 = -1;r2 = -1;e1 = -1;e2 = -1;d1 = -1;d2 = -1;t1 = -1;t2 = -1;v1 = -1;v2 = -1;l1 = -1;l22 = -1;v3 = -1;
        for(i = 0; i < s->n_models; i++){
            if(models[i][j] == 1){
                if(counter == 1)
                    m2 = i;
                else
                    m1 = i;
                counter++;
            }
            else if(models[i][j] == 2){
                m2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_rmodels; i++){
            if(rmodels[i][j] == 1){
                if(counter == 1)
                    r2 = i;
                else
                    r1 = i;
                counter++;
            }
            else if(rmodels[i][j] == 2){
                r2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_encoders; i++){
            if(encoders[i][j] == 1){
                if(counter == 1)
                    e2 = i;
                else
                    e1 = i;
                counter++;
            }
            else if(encoders[i][j] == 2){
                e2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_decoders; i++){
            if(decoders[i][j] == 1){
                if(counter == 1)
                    d2 = i;
                else
                    d1 = i;
                counter++;
            }
            else if(decoders[i][j] == 2){
                d2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_transformers; i++){
            if(transformers[i][j] == 1){
                if(counter == 1)
                    t2 = i;
                else
                    t1 = i;
                counter++;
            }
            else if(transformers[i][j] == 2){
                t2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_l2s; i++){
            if(l2s[i][j] == 1){
                if(counter == 1)
                    l22 = i;
                else
                    l1 = i;
                counter++;
            }
            else if(l2s[i][j] == 2){
                l22 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_vectors; i++){
            if(vectors[i][j] == 1){
                if(counter == 1)
                    v2 = i;
                else
                    v1 = i;
                counter++;
            }
            else if(l2s[i][j] == 2){
                if(counter == 1)
                    v2 = i;
                else
                    v3 = i;
                counter++;
            }
        }
        
        model* temp_m1 = NULL;
        model* temp_m2 = NULL;
        rmodel* temp_r1 = NULL;
        rmodel* temp_r2 = NULL;
        transformer_encoder* temp_e1 = NULL;
        transformer_encoder* temp_e2 = NULL;
        transformer_decoder* temp_d1 = NULL;
        transformer_decoder* temp_d2 = NULL;
        transformer* temp_t1 = NULL;
        transformer* temp_t2 = NULL;
        scaled_l2_norm* temp_l1 = NULL;
        scaled_l2_norm* temp_l2 = NULL;
        vector_struct* temp_v1 = NULL;
        vector_struct* temp_v2 = NULL;
        vector_struct* temp_v3 = NULL;
        
        if(m1 != -1)
            temp_m1 = m[m1];
        if(m2 != -1)
            temp_m2 = m[m2];
        if(r1 != -1)
            temp_r1 = r[r1];
        if(r2 != -1)
            temp_r2 = r[r2];
        if(e1 != -1)
            temp_e1 = e[e1];
        if(e2 != -1)
            temp_e2 = e[e2];
        if(d1 != -1)
            temp_d1 = d[d1];
        if(d2 != -1)
            temp_d2 = d[d2];
        if(t1 != -1)
            temp_t1 = t[t1];
        if(t2 != -1)
            temp_t2 = t[t2];
        if(l1 != -1)
            temp_l1 = l2[l1];
        if(l22 != -1)
            temp_l2 = l2[l22];
        if(v1 != -1)
            temp_v1 = v[v1];
        if(v2 != -1)
            temp_v2 = v[v2];
        if(v3 != -1)
            temp_v3 = v[v3];
                
        sc[j] = structure_connection(s->s[j]->id, temp_m1, temp_m2, temp_r1, temp_r2, temp_e1, temp_e2, temp_d1, temp_d2, temp_t1, temp_t2, temp_l1, temp_l2, temp_v1, temp_v2, temp_v3, s->s[j]->input1_type, s->s[j]->input2_type, s->s[j]->output_type, get_new_copy_int_array(s->s[j]->input_temporal_index,s->s[j]->temporal_encoding_model_size),get_new_copy_int_array(s->s[j]->input_encoder_indeces,s->s[j]->transf_enc_input),get_new_copy_int_array(s->s[j]->input_decoder_indeces_left,s->s[j]->decoder_left_input), get_new_copy_int_array(s->s[j]->input_decoder_indeces_down,s->s[j]->decoder_down_input), get_new_copy_int_array(s->s[j]->input_transf_encoder_indeces,s->s[j]->transf_enc_input), get_new_copy_int_array(s->s[j]->input_transf_decoder_indeces,s->s[j]->transf_dec_input), get_new_copy_int_array(s->s[j]->rmodel_input_left,s->s[j]->r2->n_lstm), get_new_copy_int_array(s->s[j]->rmodel_input_down,s->s[j]->r2->lstms[0]->window), s->s[j]->decoder_left_input,s->s[j]->decoder_down_input, s->s[j]->transf_dec_input, s->s[j]->transf_enc_input, s->s[j]->concatenate_flag,s->s[j]->input_size, s->s[j]->model_input_index, s->s[j]->temporal_encoding_model_size,s->s[j]->vector_index);
    }
    
    for(i = 0; i < s->n_models; i++){
        int output_index = -1;
        for(j = 0; j < s->n_struct_conn; j++){
            if(models[i][j] == 2){
                output_index = j;
                break;
            }
        }
        if(output_index != -1){
            for(j = 0; j < s->n_struct_conn; j++){
                if(models[i][j] == 1){
                    if(sc[j]->input1_type == TEMPORAL_ENCODING_MODEL && sc[j]->m1 == m[i]){
                        sc[j]->temporal_m = sc[output_index]->temporal_m;
                    }
                    else if(sc[j]->input2_type == TEMPORAL_ENCODING_MODEL && sc[j]->m2 == m[i]){
                        sc[j]->temporal_m2 = sc[output_index]->temporal_m;
                    }
                }
                
            }
        }
    }
    
    return init_mother_of_all_structs(s->n_inputs,s->n_models,s->n_rmodels,s->n_encoders,s->n_decoders,s->n_transformers,s->n_l2s,s->n_vectors,s->n_total_structures,s->n_struct_conn,s->n_targets,m,r,e,d,t,l2,v,sc,models,rmodels,encoders,decoders,transformers,l2s,vectors,targets,targets_index,targets_error_flag,targets_weights,targets_threshold1,targets_threshold2,targets_gamma,targets_size);
}

struct_conn_handler* copy_struct_conn_handler_without_learning_parameters(struct_conn_handler* s){
    int i,j;
    int m1,m2,r1,r2,e1,e2,d1,d2,t1,t2,v1,v2,l1,l22,v3, counter;
    model** m = NULL;
    rmodel** r = NULL;
    transformer_encoder** e = NULL;
    transformer_decoder** d = NULL;
    transformer** t = NULL;
    scaled_l2_norm** l2 = NULL;
    vector_struct** v = NULL;
    struct_conn** sc = NULL;
    int** models = NULL;
    int** rmodels = NULL;
    int** encoders = NULL;
    int** decoders = NULL;
    int** transformers = NULL;
    int** l2s = NULL;
    int** vectors = NULL;
    float** targets = NULL;
    int* targets_index = NULL;
    int* targets_size = NULL;
    int* targets_error_flag = NULL;
    float** targets_weights = NULL;
    float* targets_threshold1 = NULL;
    float* targets_threshold2 = NULL;
    float* targets_gamma = NULL;
    
    if(s->m != NULL){
        m = (model**)malloc(sizeof(model*)*s->n_models);
        models = (int**)malloc(sizeof(int*)*s->n_models);
        for(i = 0; i < s->n_models; i++){
            m[i] = copy_model_without_learning_parameters(s->m[i]);
            models[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->models[i],models[i],s->n_struct_conn);
        }
    }
    if(s->r != NULL){
        r = (rmodel**)malloc(sizeof(rmodel*)*s->n_rmodels);
        rmodels = (int**)malloc(sizeof(int*)*s->n_rmodels);
        for(i = 0; i < s->n_rmodels; i++){
            r[i] = copy_rmodel_without_learning_parameters(s->r[i]);
            rmodels[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->rmodels[i],rmodels[i],s->n_struct_conn);
        }
    }
    if(s->e != NULL){
        e = (transformer_encoder**)malloc(sizeof(transformer_encoder*)*s->n_encoders);
        encoders = (int**)malloc(sizeof(int*)*s->n_encoders);
        for(i = 0; i < s->n_encoders; i++){
            e[i] = copy_transformer_encoder_without_learning_parameters(s->e[i]);
            encoders[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->encoders[i],encoders[i],s->n_struct_conn);
        }
    }
    if(s->d != NULL){
        d = (transformer_decoder**)malloc(sizeof(transformer_decoder*)*s->n_decoders);
        decoders = (int**)malloc(sizeof(int*)*s->n_decoders);
        for(i = 0; i < s->n_decoders; i++){
            d[i] = copy_transformer_decoder_without_learning_parameters(s->d[i]);
            decoders[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->decoders[i],decoders[i],s->n_struct_conn);
        }
    }
    if(s->t != NULL){
        t = (transformer**)malloc(sizeof(transformer*)*s->n_transformers);
        transformers = (int**)malloc(sizeof(int*)*s->n_transformers);
        for(i = 0; i < s->n_transformers; i++){
            t[i] = copy_transf_without_learning_parameters(s->t[i]);
            transformers[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->transformers[i],transformers[i],s->n_struct_conn);
        }
    }
    if(s->l2 != NULL){
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*s->n_l2s);
        l2s = (int**)malloc(sizeof(int*)*s->n_l2s);
        for(i = 0; i < s->n_l2s; i++){
            l2[i] = copy_scaled_l2_norm(s->l2[i]);
            l2s[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->l2s[i],l2s[i],s->n_struct_conn);
        }
    }
    if(s->v != NULL){
        v = (vector_struct**)malloc(sizeof(vector_struct*)*s->n_vectors);
        vectors = (int**)malloc(sizeof(int*)*s->n_vectors);
        for(i = 0; i < s->n_vectors; i++){
            v[i] = copy_vector(s->v[i]);
            vectors[i] = (int*)calloc(s->n_struct_conn,sizeof(int));
            copy_int_array(s->vectors[i],vectors[i],s->n_struct_conn);
        }
    }
    
    if(s->targets != NULL){
        targets = (float**)malloc(sizeof(float*)*s->n_targets);
        targets_weights = (float**)malloc(sizeof(float*)*s->n_targets);
        targets_index = (int*)malloc(sizeof(int)*s->n_targets);
        targets_size = (int*)malloc(sizeof(int)*s->n_targets);
        targets_error_flag = (int*)malloc(sizeof(int)*s->n_targets);
        targets_threshold1 = (float*)malloc(sizeof(float)*s->n_targets);
        targets_threshold2 = (float*)malloc(sizeof(float)*s->n_targets);
        targets_gamma = (float*)malloc(sizeof(float)*s->n_targets);
        copy_int_array(s->targets_index,targets_index,s->n_targets);
        copy_int_array(s->targets_size,targets_size,s->n_targets);
        copy_int_array(s->targets_error_flag,targets_error_flag,s->n_targets);
        copy_array(s->targets_threshold1,targets_threshold1,s->n_targets);
        copy_array(s->targets_threshold2,targets_threshold2,s->n_targets);
        copy_array(s->targets_gamma,targets_gamma,s->n_targets);
        for(i = 0; i < s->n_targets; i++){
            targets_weights[i] = (float*)calloc(s->targets_size[i],sizeof(float));
            copy_array(s->targets_weights[i],targets_weights[i],s->targets_size[i]);
        }
    }
    
    sc = (struct_conn**)malloc(sizeof(struct_conn*)*s->n_struct_conn);
    
    for(j = 0; j < s->n_struct_conn; j++){
        counter = 0;
        m1 = -1;m2 = -1;r1 = -1;r2 = -1;e1 = -1;e2 = -1;d1 = -1;d2 = -1;t1 = -1;t2 = -1;v1 = -1;v2 = -1;l1 = -1;l22 = -1;v3 = -1;
        for(i = 0; i < s->n_models; i++){
            if(models[i][j] == 1){
                if(counter == 1)
                    m2 = i;
                else
                    m1 = i;
                counter++;
            }
            else if(models[i][j] == 2){
                m2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_rmodels; i++){
            if(rmodels[i][j] == 1){
                if(counter == 1)
                    r2 = i;
                else
                    r1 = i;
                counter++;
            }
            else if(rmodels[i][j] == 2){
                r2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_encoders; i++){
            if(encoders[i][j] == 1){
                if(counter == 1)
                    e2 = i;
                else
                    e1 = i;
                counter++;
            }
            else if(encoders[i][j] == 2){
                e2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_decoders; i++){
            if(decoders[i][j] == 1){
                if(counter == 1)
                    d2 = i;
                else
                    d1 = i;
                counter++;
            }
            else if(decoders[i][j] == 2){
                d2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_transformers; i++){
            if(transformers[i][j] == 1){
                if(counter == 1)
                    t2 = i;
                else
                    t1 = i;
                counter++;
            }
            else if(transformers[i][j] == 2){
                t2 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_l2s; i++){
            if(l2s[i][j] == 1){
                if(counter == 1)
                    l22 = i;
                else
                    l1 = i;
                counter++;
            }
            else if(l2s[i][j] == 2){
                l22 = i;
                counter++;
            }
        }
        for(i = 0; i < s->n_vectors; i++){
            if(vectors[i][j] == 1){
                if(counter == 1)
                    v2 = i;
                else
                    v1 = i;
                counter++;
            }
            else if(l2s[i][j] == 2){
                if(counter == 1)
                    v2 = i;
                else
                    v3 = i;
                counter++;
            }
        }
        
        model* temp_m1 = NULL;
        model* temp_m2 = NULL;
        rmodel* temp_r1 = NULL;
        rmodel* temp_r2 = NULL;
        transformer_encoder* temp_e1 = NULL;
        transformer_encoder* temp_e2 = NULL;
        transformer_decoder* temp_d1 = NULL;
        transformer_decoder* temp_d2 = NULL;
        transformer* temp_t1 = NULL;
        transformer* temp_t2 = NULL;
        scaled_l2_norm* temp_l1 = NULL;
        scaled_l2_norm* temp_l2 = NULL;
        vector_struct* temp_v1 = NULL;
        vector_struct* temp_v2 = NULL;
        vector_struct* temp_v3 = NULL;
        
        if(m1 != -1){
            if(s->s[j]->input1_type == TEMPORAL_ENCODING_MODEL)
                temp_m1 = s->s[j]->m1;
            else
                temp_m1 = m[m1];
        }
        if(m2 != -1){
            if(s->s[j]->input2_type == TEMPORAL_ENCODING_MODEL)
                temp_m2 = s->s[j]->m2;
            else
                temp_m2 = m[m2];
        }
        if(r1 != -1)
            temp_r1 = r[r1];
        if(r2 != -1)
            temp_r2 = r[r2];
        if(e1 != -1)
            temp_e1 = e[e1];
        if(e2 != -1)
            temp_e2 = e[e2];
        if(d1 != -1)
            temp_d1 = d[d1];
        if(d2 != -1)
            temp_d2 = d[d2];
        if(t1 != -1)
            temp_t1 = t[t1];
        if(t2 != -1)
            temp_t2 = t[t2];
        if(l1 != -1)
            temp_l1 = l2[l1];
        if(l22 != -1)
            temp_l2 = l2[l22];
        if(v1 != -1)
            temp_v1 = v[v1];
        if(v2 != -1)
            temp_v2 = v[v2];
        if(v3 != -1)
            temp_v3 = v[v3];
                
        sc[j] = structure_connection(s->s[j]->id, temp_m1, temp_m2, temp_r1, temp_r2, temp_e1, temp_e2, temp_d1, temp_d2, temp_t1, temp_t2, temp_l1, temp_l2, temp_v1, temp_v2, temp_v3, s->s[j]->input1_type, s->s[j]->input2_type, s->s[j]->output_type, get_new_copy_int_array(s->s[j]->input_temporal_index,s->s[j]->temporal_encoding_model_size),get_new_copy_int_array(s->s[j]->input_encoder_indeces,s->s[j]->transf_enc_input),get_new_copy_int_array(s->s[j]->input_decoder_indeces_left,s->s[j]->decoder_left_input), get_new_copy_int_array(s->s[j]->input_decoder_indeces_down,s->s[j]->decoder_down_input), get_new_copy_int_array(s->s[j]->input_transf_encoder_indeces,s->s[j]->transf_enc_input), get_new_copy_int_array(s->s[j]->input_transf_decoder_indeces,s->s[j]->transf_dec_input), get_new_copy_int_array(s->s[j]->rmodel_input_left,s->s[j]->r2->n_lstm), get_new_copy_int_array(s->s[j]->rmodel_input_down,s->s[j]->r2->lstms[0]->window), s->s[j]->decoder_left_input,s->s[j]->decoder_down_input, s->s[j]->transf_dec_input, s->s[j]->transf_enc_input, s->s[j]->concatenate_flag,s->s[j]->input_size, s->s[j]->model_input_index, s->s[j]->temporal_encoding_model_size,s->s[j]->vector_index);
    }
    
    for(i = 0; i < s->n_models; i++){
        int output_index = -1;
        for(j = 0; j < s->n_struct_conn; j++){
            if(models[i][j] == 2){
                output_index = j;
                break;
            }
        }
        if(output_index != -1){
            for(j = 0; j < s->n_struct_conn; j++){
                if(models[i][j] == 1){
                    if(sc[j]->input1_type == TEMPORAL_ENCODING_MODEL && sc[j]->m1 == m[i]){
                        sc[j]->temporal_m = sc[output_index]->temporal_m;
                    }
                    else if(sc[j]->input2_type == TEMPORAL_ENCODING_MODEL && sc[j]->m2 == m[i]){
                        sc[j]->temporal_m2 = sc[output_index]->temporal_m;
                    }
                }
                
            }
        }
    }
    
    return init_mother_of_all_structs(s->n_inputs, s->n_models,s->n_rmodels,s->n_encoders,s->n_decoders,s->n_transformers,s->n_l2s,s->n_vectors,s->n_total_structures,s->n_struct_conn,s->n_targets,m,r,e,d,t,l2,v,sc,models,rmodels,encoders,decoders,transformers,l2s,vectors,targets,targets_index,targets_error_flag,targets_weights,targets_threshold1,targets_threshold2,targets_gamma,targets_size);
}


void paste_struct_conn_handler(struct_conn_handler* s, struct_conn_handler* copy){
    int i,j;
    for(i = 0; i < s->n_models; i++){
        paste_model(s->m[i],copy->m[i]);
        if(s->models != NULL)
            copy_int_array(s->models[i],copy->models[i],s->n_struct_conn);
        
    }
    for(i = 0; i < s->n_rmodels; i++){
        paste_rmodel(s->r[i],copy->r[i]);
        if(s->rmodels != NULL)
            copy_int_array(s->rmodels[i],copy->rmodels[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_encoders; i++){
        paste_transformer_encoder(s->e[i],copy->e[i]);
        if(s->encoders != NULL)
            copy_int_array(s->encoders[i],copy->encoders[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_decoders; i++){
        paste_transformer_decoder(s->d[i],copy->d[i]);
        if(s->decoders != NULL)
            copy_int_array(s->decoders[i],copy->decoders[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_transformers; i++){
        paste_transformer(s->t[i],copy->t[i]);
        if(s->transformers != NULL)
            copy_int_array(s->transformers[i],copy->transformers[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_vectors; i++){
        paste_vector(s->v[i],copy->v[i]);
        if(s->vectors != NULL)
            copy_int_array(s->vectors[i],copy->vectors[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_l2s; i++){
        paste_scaled_l2_norm(s->l2[i],copy->l2[i]);
        if(s->l2s != NULL)
            copy_int_array(s->l2s[i],copy->l2s[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        paste_struct_conn(s->s[i],copy->s[i]);
    }
    
    copy_int_array(s->targets_index,copy->targets_index,s->n_targets);
    copy_int_array(s->targets_size,copy->targets_size,s->n_targets);
    copy_int_array(s->targets_error_flag,copy->targets_error_flag,s->n_targets);
    copy_array(s->targets_threshold1,copy->targets_threshold1,s->n_targets);
    copy_array(s->targets_threshold2,copy->targets_threshold2,s->n_targets);
    copy_array(s->targets_gamma,copy->targets_gamma,s->n_targets);
    
    if(s->targets_weights != NULL){
        for(i = 0; i < s->n_targets; i++){
            copy_array(s->targets_weights[i],copy->targets_weights[i],s->targets_index[i]);
        }
    }
    
    return;
}
void paste_struct_conn_handler_without_learning_parameters(struct_conn_handler* s, struct_conn_handler* copy){
    int i,j;
    for(i = 0; i < s->n_models; i++){
        paste_model_without_learning_parameters(s->m[i],copy->m[i]);
        if(s->models != NULL)
            copy_int_array(s->models[i],copy->models[i],s->n_struct_conn);
        
    }
    for(i = 0; i < s->n_rmodels; i++){
        paste_rmodel_without_learning_parameters(s->r[i],copy->r[i]);
        if(s->rmodels != NULL)
            copy_int_array(s->rmodels[i],copy->rmodels[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_encoders; i++){
        paste_transformer_encoder_without_learning_parameters(s->e[i],copy->e[i]);
        if(s->encoders != NULL)
            copy_int_array(s->encoders[i],copy->encoders[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_decoders; i++){
        paste_transformer_decoder_without_learning_parameters(s->d[i],copy->d[i]);
        if(s->decoders != NULL)
            copy_int_array(s->decoders[i],copy->decoders[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_transformers; i++){
        paste_transformer_without_learning_parameters(s->t[i],copy->t[i]);
        if(s->transformers != NULL)
            copy_int_array(s->transformers[i],copy->transformers[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_vectors; i++){
        paste_vector(s->v[i],copy->v[i]);
        if(s->vectors != NULL)
            copy_int_array(s->vectors[i],copy->vectors[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_l2s; i++){
        paste_scaled_l2_norm(s->l2[i],copy->l2[i]);
        if(s->l2s != NULL)
            copy_int_array(s->l2s[i],copy->l2s[i],s->n_struct_conn);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        paste_struct_conn(s->s[i],copy->s[i]);
    }
    
    copy_int_array(s->targets_index,copy->targets_index,s->n_targets);
    copy_int_array(s->targets_size,copy->targets_size,s->n_targets);
    copy_int_array(s->targets_error_flag,copy->targets_error_flag,s->n_targets);
    copy_array(s->targets_threshold1,copy->targets_threshold1,s->n_targets);
    copy_array(s->targets_threshold2,copy->targets_threshold2,s->n_targets);
    copy_array(s->targets_gamma,copy->targets_gamma,s->n_targets);
    
    if(s->targets_weights != NULL){
        for(i = 0; i < s->n_targets; i++){
            copy_array(s->targets_weights[i],copy->targets_weights[i],s->targets_index[i]);
        }
    }
    
    return;
}


void slow_paste_struct_conn_handler(struct_conn_handler* s, struct_conn_handler* copy, float tau){
    int i,j;
    for(i = 0; i < s->n_models; i++){
        slow_paste_model(s->m[i],copy->m[i], tau);
        
    }
    for(i = 0; i < s->n_rmodels; i++){
        slow_paste_rmodel(s->r[i],copy->r[i], tau);
        
    }
    for(i = 0; i < s->n_encoders; i++){
        slow_paste_transformer_encoder(s->e[i],copy->e[i], tau);
        
    }
    for(i = 0; i < s->n_decoders; i++){
        slow_paste_transformer_decoder(s->d[i],copy->d[i], tau);
        
    }
    for(i = 0; i < s->n_transformers; i++){
        slow_paste_transformer(s->t[i],copy->t[i], tau);
        
    }
    for(i = 0; i < s->n_l2s; i++){
        slow_paste_scaled_l2_norm(s->l2[i],copy->l2[i], tau);
    }
    
    return;
}

void reset_struct_conn_handler(struct_conn_handler* s){
    int i,j;
    for(i = 0; i < s->n_models; i++){
        reset_model(s->m[i]);
    }
    for(i = 0; i < s->n_rmodels; i++){
        reset_rmodel(s->r[i]);
    }
    for(i = 0; i < s->n_encoders; i++){
        reset_transformer_encoder(s->e[i]);
    }
    for(i = 0; i < s->n_decoders; i++){
        reset_transformer_decoder(s->d[i]);
    }
    for(i = 0; i < s->n_transformers; i++){
        reset_transf(s->t[i]);
    }
    for(i = 0; i < s->n_vectors; i++){
        reset_vector(s->v[i]);
    }
    for(i = 0; i < s->n_l2s; i++){
        reset_scaled_l2_norm(s->l2[i]);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        reset_struct_conn(s->s[i]);
    }
    return;
}
void reset_struct_conn_handler_without_learning_parameters(struct_conn_handler* s){
    int i,j;
    for(i = 0; i < s->n_models; i++){
        reset_model_without_learning_parameters(s->m[i]);
    }
    for(i = 0; i < s->n_rmodels; i++){
        reset_rmodel_without_learning_parameters(s->r[i]);
    }
    for(i = 0; i < s->n_encoders; i++){
        reset_transformer_encoder_without_learning_parameters(s->e[i]);
    }
    for(i = 0; i < s->n_decoders; i++){
        reset_transformer_decoder_without_learning_parameters(s->d[i]);
    }
    for(i = 0; i < s->n_transformers; i++){
        reset_transf_without_learning_parameters(s->t[i]);
    }
    for(i = 0; i < s->n_vectors; i++){
        reset_vector(s->v[i]);
    }
    for(i = 0; i < s->n_l2s; i++){
        reset_scaled_l2_norm(s->l2[i]);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        reset_struct_conn(s->s[i]);
    }
    return;
}

uint64_t size_of_struct_conn_handler(struct_conn_handler* s){
	uint64_t sum = 0;
	int i,j;
    for(i = 0; i < s->n_models; i++){
        sum+=size_of_model(s->m[i]);
    }
    for(i = 0; i < s->n_rmodels; i++){
        sum+=size_of_rmodel(s->r[i]);
    }
    for(i = 0; i < s->n_encoders; i++){
		sum+=size_of_transformer_encoder(s->e[i]);
    }
    for(i = 0; i < s->n_decoders; i++){
        sum+=size_of_transformer_decoder(s->d[i]);
    }
    for(i = 0; i < s->n_transformers; i++){
		sum+=size_of_transformer(s->t[i]);
    }
    for(i = 0; i < s->n_vectors; i++){
		sum+=size_of_vector(s->v[i]);
    }
    for(i = 0; i < s->n_l2s; i++){
        reset_scaled_l2_norm(s->l2[i]);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        reset_struct_conn(s->s[i]);
    }
    return sum;
	
}

uint64_t size_of_struct_conn_handler_without_learning_parameters(struct_conn_handler* s){
	uint64_t sum = 0;
	int i,j;
    for(i = 0; i < s->n_models; i++){
        sum+=size_of_model_without_learning_parameters(s->m[i]);
    }
    for(i = 0; i < s->n_rmodels; i++){
        sum+=size_of_rmodel_without_learning_parameters(s->r[i]);
    }
    for(i = 0; i < s->n_encoders; i++){
		sum+=size_of_transformer_encoder_without_learning_parameters(s->e[i]);
    }
    for(i = 0; i < s->n_decoders; i++){
        sum+=size_of_transformer_decoder_without_learning_parameters(s->d[i]);
    }
    for(i = 0; i < s->n_transformers; i++){
		sum+=size_of_transformer_without_learning_parameters(s->t[i]);
    }
    for(i = 0; i < s->n_vectors; i++){
		sum+=size_of_vector(s->v[i]);
    }
    for(i = 0; i < s->n_l2s; i++){
        reset_scaled_l2_norm(s->l2[i]);
    }
    for(i = 0; i < s->n_struct_conn; i++){
        reset_struct_conn(s->s[i]);
    }
    return sum;
	
}

