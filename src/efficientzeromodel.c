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

efficientzeromodel* init_efficientzero_model(model* rapresentation_h, model* dynamics_g, model* prediction_f, model* prediction_f_policy,
                                             model* prediction_f_value, model* reward_prediction_model, rmodel* reward_prediction_rmodel,
                                             model* reward_prediction_temporal_model, model* p1, model* p2, int threads, int lstm_window){
     efficientzeromodel* m = (efficientzeromodel*)malloc(sizeof(efficientzeromodel));
     if(threads < 1 || lstm_window < 2){
         fprintf(stderr,"Error: batch size must be >= 1 and lstm_window must be >= 2\n");
         exit(1);
     }
     
     if(rapresentation_h == NULL || dynamics_g == NULL || prediction_f == NULL || prediction_f_policy == NULL || prediction_f_value == NULL ||
        reward_prediction_model == NULL || reward_prediction_rmodel == NULL || reward_prediction_temporal_model == NULL){
         fprintf(stderr,"Error: one of your models is null!\n");
         exit(1);
    }
    
    if(p1 != NULL && p2 == NULL || p1 == NULL && p2 != NULL){
        fprintf(stderr,"Error, if your p1 is not null and p2 is null and viceversa is an error!\n"),
        exit(1);
    }
    int batch_size = threads;
    m->batch_size = batch_size;
    m->lstm_window = lstm_window;
    m->rapresentation_h = rapresentation_h;
    m->dynamics_g = dynamics_g;
    m->prediction_f = prediction_f;
    m->prediction_f_policy = prediction_f_policy;
    m->prediction_f_value = prediction_f_value;
    m->reward_prediction_model = reward_prediction_model;
    m->reward_prediction_rmodel = reward_prediction_rmodel;
    m->reward_prediction_temporal_model = reward_prediction_temporal_model;
    m->p1 = p1;
    m->p2 = p2;
    
    m->batch_reward_prediction_model = (model**)malloc(sizeof(model*)*lstm_window*batch_size);
    m->batch_reward_prediction_temporal_model = (model**)malloc(sizeof(model*)*lstm_window*batch_size);
    int i;
    
    for(i = 0; i < batch_size*lstm_window; i++){
        m->batch_reward_prediction_model[i] = copy_model_without_learning_parameters(m->reward_prediction_model);
        m->batch_reward_prediction_temporal_model[i] = copy_model_without_learning_parameters(m->reward_prediction_temporal_model);
    }

    if(threads > 1){
        
        
        m->batch_rapresentation_h = (model**)malloc(sizeof(model*)*batch_size);
        m->batch_dynamics_g = (model**)malloc(sizeof(model*)*batch_size);
        m->batch_prediction_f = (model**)malloc(sizeof(model*)*batch_size);
        m->batch_prediction_f_policy = (model**)malloc(sizeof(model*)*batch_size);
        m->batch_prediction_f_value = (model**)malloc(sizeof(model*)*batch_size);
        m->batch_reward_prediction_rmodel = (rmodel**)malloc(sizeof(rmodel*)*batch_size);
        if(p1 != NULL){
            m->batch_p1 = (model**)malloc(sizeof(model*)*batch_size);
        }
        
        else{
            m->batch_p1 = NULL;
        }
        if(p2 != NULL){
            m->batch_p2 = (model**)malloc(sizeof(model*)*batch_size);
        }
        
        else{
            m->batch_p2 = NULL;
        }
        int i;
        for(i = 0; i < batch_size; i++){
            m->batch_rapresentation_h[i] = copy_model_without_learning_parameters(m->rapresentation_h);
            m->batch_dynamics_g[i] = copy_model_without_learning_parameters(m->dynamics_g);
            m->batch_prediction_f[i] = copy_model_without_learning_parameters(m->prediction_f);
            m->batch_prediction_f_policy[i] = copy_model_without_learning_parameters(m->prediction_f_policy);
            m->batch_prediction_f_value[i] = copy_model_without_learning_parameters(m->prediction_f_value);
            m->batch_reward_prediction_rmodel[i] = copy_rmodel_without_learning_parameters(m->reward_prediction_rmodel);
            if(p1 != NULL)
                m->batch_p1[i] = copy_model_without_learning_parameters(m->p1);
            if(p2 != NULL)
                m->batch_p2[i] = copy_model_without_learning_parameters(m->p2);
        }
    }
    
    else{
        m->batch_rapresentation_h = NULL;
        m->batch_dynamics_g = NULL;
        m->batch_prediction_f = NULL;
        m->batch_prediction_f_policy = NULL;
        m->batch_prediction_f_value = NULL;
        m->batch_reward_prediction_rmodel = NULL;
        m->batch_p1 = NULL;
        m->batch_p2 = NULL;
    }
    return m;
}


void make_efficientzeromodel_only_for_ff(efficientzeromodel* m){
    if(m == NULL)
        return;
    
    if(m->rapresentation_h != NULL)
        make_the_model_only_for_ff(m->rapresentation_h);
    
    if(m->dynamics_g != NULL)
        make_the_model_only_for_ff(m->dynamics_g);
    
    if(m->prediction_f != NULL)
        make_the_model_only_for_ff(m->prediction_f);
    
    if(m->prediction_f_policy != NULL)
        make_the_model_only_for_ff(m->prediction_f_policy);
    
    if(m->prediction_f_value != NULL)
        make_the_model_only_for_ff(m->prediction_f_value);
        
    if(m->reward_prediction_rmodel != NULL)
        make_the_rmodel_only_for_ff(m->reward_prediction_rmodel);
        
    if(m->reward_prediction_model != NULL)
        make_the_model_only_for_ff(m->reward_prediction_model);
        
    if(m->reward_prediction_temporal_model != NULL)
        make_the_model_only_for_ff(m->reward_prediction_temporal_model);
    
    if(m->p1 != NULL)
        make_the_model_only_for_ff(m->p1);
    
    if(m->p2 != NULL)
        make_the_model_only_for_ff(m->p2);
    
    int i;
    for(i = 0; i < m->batch_size*m->lstm_window; i++){
        make_the_model_only_for_ff(m->batch_reward_prediction_model[i]);
        make_the_model_only_for_ff(m->batch_reward_prediction_temporal_model[i]);
    }
    
    for(i = 0;m->batch_size > 1 &&  i < m->batch_size; i++){
        make_the_model_only_for_ff(m->batch_rapresentation_h[i]);
        make_the_model_only_for_ff(m->batch_dynamics_g[i]);
        make_the_model_only_for_ff(m->batch_prediction_f[i]);
        make_the_model_only_for_ff(m->batch_prediction_f_value[i]);
        make_the_model_only_for_ff(m->batch_prediction_f_policy[i]);
        make_the_rmodel_only_for_ff(m->batch_reward_prediction_rmodel[i]);
        if(m->p1 != NULL){
            make_the_model_only_for_ff(m->batch_p1[i]);
        }
        if(m->p2 != NULL){
            make_the_model_only_for_ff(m->batch_p2[i]);
        }
    }
        
}

void free_efficientzero_model(efficientzeromodel* m){
    if(m == NULL)
        return;
    int i;
    for(i = 0;m->batch_size > 1 &&  i < m->batch_size; i++){
        free_model_without_learning_parameters(m->batch_rapresentation_h[i]);
        free_model_without_learning_parameters(m->batch_dynamics_g[i]);
        free_model_without_learning_parameters(m->batch_prediction_f[i]);
        free_model_without_learning_parameters(m->batch_prediction_f_value[i]);
        free_model_without_learning_parameters(m->batch_prediction_f_policy[i]);
        free_rmodel_without_learning_parameters(m->batch_reward_prediction_rmodel[i]);
        if(m->p1 != NULL){
            free_model_without_learning_parameters(m->batch_p1[i]);
        }
        if(m->p2 != NULL){
            free_model_without_learning_parameters(m->batch_p2[i]);
        }
    }
    for(i = 0; i < m->batch_size*m->lstm_window; i++){
        free_model_without_learning_parameters(m->batch_reward_prediction_model[i]);
        free_model_without_learning_parameters(m->batch_reward_prediction_temporal_model[i]);
    }
    free(m->batch_rapresentation_h);
    free(m->batch_dynamics_g);
    free(m->batch_prediction_f);
    free(m->batch_prediction_f_policy);
    free(m->batch_prediction_f_value);
    free(m->batch_reward_prediction_model);
    free(m->batch_reward_prediction_rmodel);
    free(m->batch_reward_prediction_temporal_model);
    free(m->batch_p1);
    free(m->batch_p2);
    free_model(m->rapresentation_h);
    free_model(m->dynamics_g);
    free_model(m->prediction_f);
    free_model(m->prediction_f_policy);
    free_model(m->prediction_f_value);
    free_model(m->reward_prediction_model);
    free_rmodel(m->reward_prediction_rmodel);
    free_model(m->reward_prediction_temporal_model);
    free_model(m->p1);
    free_model(m->p2);
    free(m);
    return;
    
}

void efficientzero_ff_p1(efficientzeromodel* m, float* input){
    model_tensor_input_ff(m->p1,get_input_layer_size(m->p1),1,1,input); 
}

float* efficientzero_bp_p1(efficientzeromodel* m, float* input, float* error){
    return model_tensor_input_bp(m->p1,get_input_layer_size(m->p1),1,1,input, error, get_output_dimension_from_model(m->p1)); 
}

void efficientzero_reset_p1(efficientzeromodel* m){
    reset_model(m->p1);
}
void efficientzero_reset_only_for_ff_p1(efficientzeromodel* m){
    reset_model_only_for_ff(m->p1);
}

void efficientzero_ff_p2(efficientzeromodel* m, float* input){
    model_tensor_input_ff(m->p2,get_input_layer_size(m->p2),1,1,input); 
}

float* efficientzero_bp_p2(efficientzeromodel* m, float* input, float* error){
    return model_tensor_input_bp(m->p2,get_input_layer_size(m->p2),1,1,input, error, get_output_dimension_from_model(m->p2)); 
}

void efficientzero_reset_p2(efficientzeromodel* m){
    reset_model(m->p2);
}
void efficientzero_reset_only_for_ff_p2(efficientzeromodel* m){
    reset_model_only_for_ff(m->p2);
}

void efficientzero_ff_rapresentation_h(efficientzeromodel* m, float* input){
    model_tensor_input_ff(m->rapresentation_h,get_input_layer_size(m->rapresentation_h),1,1,input); 
}

float* efficientzero_bp_rapresentation_h(efficientzeromodel* m, float* input, float* error){
    return model_tensor_input_bp(m->rapresentation_h,get_input_layer_size(m->rapresentation_h),1,1,input, error, get_output_dimension_from_model(m->rapresentation_h)); 
}

void efficientzero_reset_rapresentation_h(efficientzeromodel* m){
    reset_model(m->rapresentation_h);
}

void efficientzero_reset_only_for_ff_rapresentation_h(efficientzeromodel* m){
    reset_model_only_for_ff(m->rapresentation_h);
}

void efficientzero_ff_dynamics_g(efficientzeromodel* m, float* input){
    model_tensor_input_ff(m->dynamics_g,get_input_layer_size(m->dynamics_g),1,1,input); 
}

float* efficientzero_bp_dynamics_g(efficientzeromodel* m, float* input, float* error){
    return model_tensor_input_bp(m->dynamics_g,get_input_layer_size(m->dynamics_g),1,1,input, error, get_output_dimension_from_model(m->dynamics_g)); 
}

void efficientzero_reset_dynamics_g(efficientzeromodel* m){
    reset_model(m->dynamics_g);
}

void efficientzero_reset_only_for_ff_dynamics_g(efficientzeromodel* m){
    reset_model_only_for_ff(m->dynamics_g);
}

void efficientzero_ff_prediction_f(efficientzeromodel* m, float* input){
    model_tensor_input_ff(m->prediction_f,get_input_layer_size(m->prediction_f),1,1,input); 
    model_tensor_input_ff(m->prediction_f_value,get_input_layer_size(m->prediction_f_value),1,1,m->prediction_f->output_layer); 
    model_tensor_input_ff(m->prediction_f_policy,get_input_layer_size(m->prediction_f_policy),1,1,m->prediction_f->output_layer);
}

float* efficientzero_bp_prediction_f(efficientzeromodel* m, float* input, float* error){
    float* temp1 = model_tensor_input_bp(m->prediction_f_policy,get_input_layer_size(m->prediction_f_policy),1,1,m->prediction_f->output_layer, error, get_output_dimension_from_model(m->prediction_f_policy)); 
    float* temp2 = model_tensor_input_bp(m->prediction_f_value,get_input_layer_size(m->prediction_f_value),1,1,m->prediction_f->output_layer, error, get_output_dimension_from_model(m->prediction_f_value));
    sum1D(temp1,temp2,temp1, get_output_dimension_from_model(m->prediction_f));
    return model_tensor_input_bp(m->prediction_f_value,get_input_layer_size(m->prediction_f_value),1,1,input, temp1, get_output_dimension_from_model(m->prediction_f_value)); 
}

void efficientzero_reset_prediction_f(efficientzeromodel* m){
    reset_model(m->prediction_f);
    reset_model(m->prediction_f_value);
    reset_model(m->prediction_f_policy);
}

void efficientzero_reset_only_for_ff_prediction_f(efficientzeromodel* m){
    reset_model_only_for_ff(m->prediction_f);
    reset_model_only_for_ff(m->prediction_f_value);
    reset_model_only_for_ff(m->prediction_f_policy);
}

void efficientzero_ff_reward_prediction_single_cell(efficientzeromodel* m, float* inputs, float** hidden_states, float** cell_states, float** new_hidden_states, float** new_cell_states, int fullfill_flag){
    int i;
    m->reward_prediction_rmodel->window = 1;
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        m->reward_prediction_rmodel->lstms[i]->window = 1;
    }
    model_tensor_input_ff(m->reward_prediction_model,get_input_layer_size(m->reward_prediction_model),1,1,inputs);
    float** input_rmodel = &m->reward_prediction_model->output_layer;
    ff_rmodel(hidden_states,cell_states,input_rmodel,m->reward_prediction_rmodel);
    model_tensor_input_ff(m->reward_prediction_temporal_model,get_input_layer_size(m->reward_prediction_temporal_model),1,1,m->reward_prediction_rmodel->lstms[m->reward_prediction_rmodel->layers-1]->out_up[0]);
    
    if(fullfill_flag){
        for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
            copy_array(m->reward_prediction_rmodel->lstms[i]->lstm_hidden[0],new_hidden_states[i], m->reward_prediction_rmodel->lstms[i]->output_size);
            copy_array(m->reward_prediction_rmodel->lstms[i]->lstm_cell[0],new_cell_states[i], m->reward_prediction_rmodel->lstms[i]->output_size);
        }
    }
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        m->reward_prediction_rmodel->lstms[i]->window = m->lstm_window;
    }
    m->reward_prediction_rmodel->window = m->lstm_window;
}

void set_efficientzero_model_training_edge_popup(efficientzeromodel* m, float k_percentage){
    if(m == NULL)
        return;
    int i;
    set_model_training_edge_popup(m->rapresentation_h,k_percentage);
    set_model_training_edge_popup(m->dynamics_g,k_percentage);
    set_model_training_edge_popup(m->prediction_f,k_percentage);
    set_model_training_edge_popup(m->prediction_f_policy,k_percentage);
    set_model_training_edge_popup(m->prediction_f_value,k_percentage);
    set_model_training_edge_popup(m->reward_prediction_model,k_percentage);
    set_model_training_edge_popup(m->reward_prediction_temporal_model,k_percentage);
    set_rmodel_training_edge_popup(m->reward_prediction_rmodel,k_percentage);
    for(i = 0; m->batch_size > 1 && i < m->batch_size; i++){
        set_model_training_edge_popup(m->batch_rapresentation_h[i],k_percentage);
        set_model_training_edge_popup(m->batch_prediction_f[i],k_percentage);
        set_model_training_edge_popup(m->batch_prediction_f_policy[i],k_percentage);
        set_model_training_edge_popup(m->batch_prediction_f_value[i],k_percentage);
        set_model_training_edge_popup(m->batch_dynamics_g[i],k_percentage);
        set_rmodel_training_edge_popup(m->batch_reward_prediction_rmodel[i],k_percentage);
    }
    
    for(i = 0; m->batch_size > 1 && i < m->batch_size*m->lstm_window; i++){
        set_model_training_edge_popup(m->batch_reward_prediction_model[i],k_percentage);
        set_model_training_edge_popup(m->batch_reward_prediction_temporal_model[i],k_percentage);
    }
    return;
}

void efficientzero_ff_reward_prediction(efficientzeromodel* m, float** inputs){
    model_tensor_input_ff_multicore_opt(m->batch_reward_prediction_model, m->reward_prediction_model,get_input_layer_size(m->reward_prediction_model),1,1,inputs, m->lstm_window, m->lstm_window);
    float** hidden_states = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
    float** cell_states = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
    float** input_rmodel = (float**)malloc(sizeof(float*)*m->lstm_window);
    int i;
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        hidden_states[i] = (float*)calloc(m->reward_prediction_rmodel->lstms[i]->output_size, sizeof(float));
        cell_states[i] = (float*)calloc(m->reward_prediction_rmodel->lstms[i]->output_size, sizeof(float));
    }
    for(i = 0; i < m->lstm_window; i++){
        input_rmodel[i] = m->batch_reward_prediction_model[i]->output_layer;
    }
    ff_rmodel(hidden_states,cell_states,input_rmodel,m->reward_prediction_rmodel);
    model_tensor_input_ff_multicore_opt(m->batch_reward_prediction_temporal_model, m->reward_prediction_temporal_model,get_input_layer_size(m->reward_prediction_temporal_model),1,1,m->reward_prediction_rmodel->lstms[m->reward_prediction_rmodel->layers-1]->out_up, m->lstm_window, m->lstm_window);
    free_matrix((void**)hidden_states,m->reward_prediction_rmodel->layers);
    free_matrix((void**)cell_states,m->reward_prediction_rmodel->layers);
    free(input_rmodel);
}

float** efficientzero_bp_reward_prediction(efficientzeromodel* m, float** inputs, float** errors){
    float** temp_errors = (float**)malloc(sizeof(float*)*m->lstm_window);
    model_tensor_input_bp_multicore_opt(m->batch_reward_prediction_temporal_model, m->reward_prediction_temporal_model,get_input_layer_size(m->reward_prediction_temporal_model),1,1,m->reward_prediction_rmodel->lstms[m->reward_prediction_rmodel->layers-1]->out_up, m->lstm_window, m->lstm_window, errors, get_output_dimension_from_model(m->reward_prediction_temporal_model), temp_errors);
    
    float** hidden_states = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
    float** cell_states = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
    float** input_rmodel = (float**)malloc(sizeof(float*)*m->lstm_window);
    int i;
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        hidden_states[i] = (float*)calloc(m->reward_prediction_rmodel->lstms[i]->output_size, sizeof(float));
        cell_states[i] = (float*)calloc(m->reward_prediction_rmodel->lstms[i]->output_size, sizeof(float));
    }
    for(i = 0; i < m->lstm_window; i++){
        input_rmodel[i] = m->batch_reward_prediction_model[i]->output_layer;
    }
    
    float** temp_errors2 = (float**)malloc(sizeof(float*)*m->lstm_window);
    
    float*** dfioc = bp_rmodel(hidden_states,cell_states,input_rmodel,temp_errors,m->reward_prediction_rmodel,temp_errors2);
    float** returning_value = (float**)malloc(sizeof(float*)*m->lstm_window);
    
    
    model_tensor_input_bp_multicore_opt(m->batch_reward_prediction_model, m->reward_prediction_model,get_input_layer_size(m->reward_prediction_model),1,1,inputs, m->lstm_window, m->lstm_window, temp_errors2, get_output_dimension_from_model(m->reward_prediction_model), returning_value);

    free_tensor(dfioc, m->reward_prediction_rmodel->layers,4);
    free(temp_errors);
    free_matrix((void**)hidden_states,m->reward_prediction_rmodel->layers);
    free_matrix((void**)temp_errors2,m->reward_prediction_rmodel->layers);
    free_matrix((void**)cell_states,m->reward_prediction_rmodel->layers);
    free(input_rmodel);
    return returning_value;
}

void efficientzero_reset_reward_prediction(efficientzeromodel* m){
    int i;
    for(i = 0; i < m->lstm_window; i++){
        reset_model_without_learning_parameters(m->batch_reward_prediction_model[i]);
        reset_model_without_learning_parameters(m->batch_reward_prediction_temporal_model[i]);
    }
    reset_rmodel(m->reward_prediction_rmodel);
}

void efficientzero_reset_only_for_ff_reward_prediction(efficientzeromodel* m){
    int i;
    for(i = 0; i < m->lstm_window; i++){
        reset_model_only_for_ff(m->batch_reward_prediction_model[i]);
        reset_model_only_for_ff(m->batch_reward_prediction_temporal_model[i]);
    }
    reset_rmodel_only_for_ff(m->reward_prediction_rmodel);
}

void efficientzero_reset_only_for_ff_reward_prediction_single_cell(efficientzeromodel* m){
    int i;
    m->reward_prediction_rmodel->window = 1;
    reset_model_only_for_ff(m->reward_prediction_model);
    reset_model_only_for_ff(m->reward_prediction_temporal_model);
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        m->reward_prediction_rmodel->lstms[i]->window = 1;
    }
    reset_rmodel_only_for_ff(m->reward_prediction_rmodel);
    for(i = 0; i < m->reward_prediction_rmodel->layers; i++){
        m->reward_prediction_rmodel->lstms[i]->window = m->lstm_window;
    }
}

void efficientzero_ff_p1_opt(efficientzeromodel* m, float** inputs, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    model_tensor_input_ff_multicore_opt(m->batch_p1, m->p1,get_input_layer_size(m->p1),1,1,inputs, threads, threads);
}

float** efficientzero_bp_p1_opt(efficientzeromodel* m, float** inputs, float** errors, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    float** ret_error = (float**)malloc(sizeof(float*)*m->batch_size);
    model_tensor_input_bp_multicore_opt(m->batch_p1, m->p1,get_input_layer_size(m->p1),1,1,inputs, threads, threads, errors, get_output_dimension_from_model(m->p1),ret_error);
    return ret_error;
}

void efficientzero_reset_p1_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_without_learning_parameters(m->batch_p1[i]);
    }
} 

void efficientzero_reset_only_for_ff_p1_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_only_for_ff(m->batch_p1[i]);
    }
} 

void efficientzero_ff_p2_opt(efficientzeromodel* m, float** inputs, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    model_tensor_input_ff_multicore_opt(m->batch_p2, m->p2,get_input_layer_size(m->p2),1,1,inputs, threads, threads);
}
float** efficientzero_bp_p2_opt(efficientzeromodel* m, float** inputs, float** errors, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    float** ret_error = (float**)malloc(sizeof(float*)*m->batch_size);
    model_tensor_input_bp_multicore_opt(m->batch_p2, m->p2,get_input_layer_size(m->p2),1,1,inputs, threads, threads, errors, get_output_dimension_from_model(m->p2),ret_error);
    return ret_error;
}

void efficientzero_reset_p2_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_without_learning_parameters(m->batch_p2[i]);
    }
} 

void efficientzero_reset_only_for_ff_p2_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_only_for_ff(m->batch_p2[i]);
    }
} 

void efficientzero_ff_rapresentation_h_opt(efficientzeromodel* m, float** inputs, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    model_tensor_input_ff_multicore_opt(m->batch_rapresentation_h, m->rapresentation_h,get_input_layer_size(m->rapresentation_h),1,1,inputs, threads, threads);
}

float** efficientzero_bp_rapresentation_h_opt(efficientzeromodel* m, float** inputs, float** errors, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    float** ret_error = (float**)malloc(sizeof(float*)*m->batch_size);
    model_tensor_input_bp_multicore_opt(m->batch_rapresentation_h, m->rapresentation_h,get_input_layer_size(m->rapresentation_h),1,1,inputs, threads, threads, errors, get_output_dimension_from_model(m->rapresentation_h),ret_error);
    return ret_error;
}

void efficientzero_reset_rapresentation_h_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_without_learning_parameters(m->batch_rapresentation_h[i]);
    }
} 

void efficientzero_reset_only_for_ff_rapresentation_h_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_only_for_ff(m->batch_rapresentation_h[i]);
    }
} 

void efficientzero_ff_dynamics_g_opt(efficientzeromodel* m, float** inputs, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    model_tensor_input_ff_multicore_opt(m->batch_dynamics_g, m->dynamics_g,get_input_layer_size(m->dynamics_g),1,1,inputs, threads, threads);
}

float** efficientzero_bp_dynamics_g_opt(efficientzeromodel* m, float** inputs, float** errors, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    float** ret_error = (float**)malloc(sizeof(float*)*m->batch_size);
    model_tensor_input_bp_multicore_opt(m->batch_dynamics_g, m->dynamics_g,get_input_layer_size(m->dynamics_g),1,1,inputs, threads, threads, errors, get_output_dimension_from_model(m->dynamics_g),ret_error);
    return ret_error;
}

void efficientzero_reset_dynamics_g_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_without_learning_parameters(m->batch_dynamics_g[i]);
    }
}

void efficientzero_reset_only_for_ff_dynamics_g_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_only_for_ff(m->batch_dynamics_g[i]);
    }
}

void efficientzero_ff_prediction_f_opt(efficientzeromodel* m, float** inputs, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    model_tensor_input_ff_multicore_opt(m->batch_prediction_f, m->prediction_f,get_input_layer_size(m->prediction_f),1,1,inputs, threads, threads);
    float** model_outputs = (float**)malloc(sizeof(float*)*threads);  
    for(i = 0; i < threads; i++){
        model_outputs[i] = m->batch_prediction_f[i]->output_layer;
    }
    model_tensor_input_ff_multicore_opt(m->batch_prediction_f_value, m->prediction_f_value,get_input_layer_size(m->prediction_f_value),1,1,model_outputs, threads, threads);
    model_tensor_input_ff_multicore_opt(m->batch_prediction_f_policy, m->prediction_f_policy,get_input_layer_size(m->prediction_f_policy),1,1,model_outputs, threads, threads);
    free(model_outputs);
}

float** efficientzero_bp_prediction_f_opt(efficientzeromodel* m, float** inputs, float** errors, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    float** ret_error = (float**)malloc(sizeof(float*)*m->batch_size);
    float** ret_error1 = (float**)malloc(sizeof(float*)*m->batch_size);
    float** ret_error2 = (float**)malloc(sizeof(float*)*m->batch_size);
    int i;
    float** model_outputs = (float**)malloc(sizeof(float*)*threads);  
    for(i = 0; i < threads; i++){
        model_outputs[i] = m->batch_prediction_f[i]->output_layer;
    }
    model_tensor_input_bp_multicore_opt(m->batch_prediction_f_policy, m->prediction_f_policy,get_input_layer_size(m->prediction_f_policy),1,1,model_outputs, threads, threads, errors, get_output_dimension_from_model(m->prediction_f_policy),ret_error1);
    model_tensor_input_bp_multicore_opt(m->batch_prediction_f_value, m->prediction_f_value,get_input_layer_size(m->prediction_f_value),1,1,model_outputs, threads, threads, errors, get_output_dimension_from_model(m->prediction_f_value),ret_error2);
    
    for(i = 0; i < threads; i++){
        sum1D(ret_error1[i],ret_error2[i],ret_error1[i],get_output_dimension_from_model(m->prediction_f));
    }
    
    model_tensor_input_bp_multicore_opt(m->batch_prediction_f, m->prediction_f,get_input_layer_size(m->prediction_f),1,1,inputs, threads, threads, ret_error1, get_output_dimension_from_model(m->prediction_f),ret_error);
    
    free(ret_error1);
    free(ret_error2);
    free(model_outputs);
    return ret_error;
}

void efficientzero_reset_prediction_f_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_without_learning_parameters(m->batch_prediction_f[i]);
        reset_model_without_learning_parameters(m->batch_prediction_f_value[i]);
        reset_model_without_learning_parameters(m->batch_prediction_f_policy[i]);
    }
}

void efficientzero_reset_only_for_ff_prediction_f_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < threads; i++){
        reset_model_only_for_ff(m->batch_prediction_f[i]);
        reset_model_only_for_ff(m->batch_prediction_f_value[i]);
        reset_model_only_for_ff(m->batch_prediction_f_policy[i]);
    }
}

void efficientzero_ff_reward_prediction_opt(efficientzeromodel* m, float** inputs, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    model_tensor_input_ff_multicore_opt(m->batch_reward_prediction_model, m->reward_prediction_model,get_input_layer_size(m->reward_prediction_model),1,1,inputs, m->lstm_window*threads, m->lstm_window*threads);
    float*** hidden_states = (float***)malloc(sizeof(float**)*threads);
    float*** cell_states = (float***)malloc(sizeof(float**)*threads);
    float*** input_rmodel = (float***)malloc(sizeof(float**)*threads);
    int i,j;
    for(i = 0; i < threads; i++){
        hidden_states[i] = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
        cell_states[i] = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
        input_rmodel[i] = (float**)malloc(sizeof(float*)*m->lstm_window);
        for(j = 0; j < m->reward_prediction_rmodel->layers; j++){
            hidden_states[i][j] = (float*)calloc(m->reward_prediction_rmodel->lstms[j]->output_size, sizeof(float));
            cell_states[i][j] = (float*)calloc(m->reward_prediction_rmodel->lstms[j]->output_size, sizeof(float));
        }
        for(j = 0; j < m->lstm_window; j++){
            input_rmodel[i][j] = m->batch_reward_prediction_model[i*threads+j]->output_layer;
        }
    }
    
    ff_rmodel_lstm_multicore_opt(hidden_states,cell_states,input_rmodel, m->batch_reward_prediction_rmodel, threads, threads, m->reward_prediction_rmodel);
    
    float** rmodel_output = (float**)malloc(sizeof(float*)*threads*m->lstm_window);
    for(i = 0; i < threads; i++){
        for(j = 0; j < m->lstm_window; j++){
            rmodel_output[i*m->lstm_window+j] = m->batch_reward_prediction_rmodel[i]->lstms[m->batch_reward_prediction_rmodel[i]->layers-1]->out_up[j];
        }
    }
    model_tensor_input_ff_multicore_opt(m->batch_reward_prediction_temporal_model, m->reward_prediction_temporal_model,get_input_layer_size(m->reward_prediction_temporal_model),1,1,rmodel_output, m->lstm_window*threads, m->lstm_window*threads);
    for(i = 0; i < threads; i++){
        free_matrix((void**)hidden_states[i],m->reward_prediction_rmodel->layers);
        free_matrix((void**)cell_states[i],m->reward_prediction_rmodel->layers);
        free(input_rmodel[i]);
    }
    free(hidden_states);
    free(cell_states);
    free(input_rmodel);
    free(rmodel_output);
}

float** efficientzero_bp_reward_prediction_opt(efficientzeromodel* m, float** inputs, float** errors, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    float*** hidden_states = (float***)malloc(sizeof(float**)*threads);
    float*** cell_states = (float***)malloc(sizeof(float**)*threads);
    float*** input_rmodel = (float***)malloc(sizeof(float**)*threads);
    float** ret_error1 = (float**)malloc(sizeof(float*)*threads*m->lstm_window);
    float** ret_error = (float**)malloc(sizeof(float*)*threads*m->lstm_window);
    float*** rmodel_output_error = (float***)malloc(sizeof(float**)*threads);
    float**** dfioc = (float****)malloc(sizeof(float***)*threads);
    float*** ret_error2 = (float***)malloc(sizeof(float**)*threads);
    float** ret_error3 = (float**)malloc(sizeof(float*)*threads*m->lstm_window);
    int i,j;
    for(i = 0; i < threads; i++){
        hidden_states[i] = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
        cell_states[i] = (float**)malloc(sizeof(float*)*m->reward_prediction_rmodel->layers);
        input_rmodel[i] = (float**)malloc(sizeof(float*)*m->lstm_window);
        for(j = 0; j < m->reward_prediction_rmodel->layers; j++){
            hidden_states[i][j] = (float*)calloc(m->reward_prediction_rmodel->lstms[j]->output_size, sizeof(float));
            cell_states[i][j] = (float*)calloc(m->reward_prediction_rmodel->lstms[j]->output_size, sizeof(float));
        }
        for(j = 0; j < m->lstm_window; j++){
            input_rmodel[i][j] = m->batch_reward_prediction_model[i*threads+j]->output_layer;
        }
    }
        
    float** rmodel_output = (float**)malloc(sizeof(float*)*threads*m->lstm_window);
    for(i = 0; i < threads; i++){
        for(j = 0; j < m->lstm_window; j++){
            rmodel_output[i*m->lstm_window+j] = m->batch_reward_prediction_rmodel[i]->lstms[m->batch_reward_prediction_rmodel[i]->layers-1]->out_up[j];
        }
    }
    
    model_tensor_input_bp_multicore_opt(m->batch_reward_prediction_temporal_model, m->reward_prediction_temporal_model,get_input_layer_size(m->reward_prediction_temporal_model),1,1,rmodel_output, m->lstm_window*threads, m->lstm_window*threads, errors, get_output_dimension_from_model(m->reward_prediction_temporal_model), ret_error1);
    
    for(i = 0; i < threads; i++){
        rmodel_output_error[i] = &ret_error1[i*m->lstm_window];
        ret_error2[i] = (float**)malloc(sizeof(float*)*m->lstm_window);
    }
    
    bp_rmodel_lstm_multicore_opt(hidden_states,cell_states,input_rmodel,m->batch_reward_prediction_rmodel, rmodel_output_error, threads, threads,dfioc,ret_error2,m->reward_prediction_rmodel);
    
    for(i = 0; i < threads; i++){
        for(j = 0; j < m->lstm_window; j++){
            ret_error3[i*m->lstm_window+j] = ret_error2[i][j];
        }
    }
    
    
    model_tensor_input_bp_multicore_opt(m->batch_reward_prediction_model, m->reward_prediction_model,get_input_layer_size(m->reward_prediction_model),1,1,inputs, m->lstm_window*threads, m->lstm_window*threads, ret_error3, get_output_dimension_from_model(m->reward_prediction_model), ret_error);

    for(i = 0; i < threads; i++){
        free_matrix((void**)hidden_states[i],m->reward_prediction_rmodel->layers);
        free_matrix((void**)cell_states[i],m->reward_prediction_rmodel->layers);
        free(input_rmodel[i]);
        free_tensor(dfioc[i],m->reward_prediction_rmodel->layers,4);
    }
    
    free(ret_error3);
    free_tensor(ret_error2,threads,m->lstm_window);
    free(dfioc);
    free(ret_error1);
    free(rmodel_output_error);
    free(hidden_states);
    free(cell_states);
    free(input_rmodel);
    free(rmodel_output);
    
    return ret_error;
}

void efficientzero_reset_reward_prediction_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < m->lstm_window*threads; i++){
        reset_model_without_learning_parameters(m->batch_reward_prediction_model[i]);
        reset_model_without_learning_parameters(m->batch_reward_prediction_temporal_model[i]);
    }
    for(i = 0; i < threads; i++){
        reset_rmodel_without_learning_parameters(m->batch_reward_prediction_rmodel[i]);
    }
}

void efficientzero_reset_only_for_ff_reward_prediction_without_learning_parameters(efficientzeromodel* m, int threads){
    if(threads > m->batch_size)
        threads = m->batch_size;
    int i;
    for(i = 0; i < m->lstm_window*threads; i++){
        reset_model_only_for_ff(m->batch_reward_prediction_model[i]);
        reset_model_only_for_ff(m->batch_reward_prediction_temporal_model[i]);
    }
    for(i = 0; i < threads; i++){
        reset_rmodel_only_for_ff(m->batch_reward_prediction_rmodel[i]);
    }
}

void save_efficientzero_model(efficientzeromodel* m, int n){
    if(m == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa_n(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    
    convert_data(&m->batch_size,sizeof(int),1);// new
    i = fwrite(&m->batch_size,sizeof(int),1,fw);// new
    convert_data(&m->batch_size,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    convert_data(&m->lstm_window,sizeof(int),1);// new
    i = fwrite(&m->lstm_window,sizeof(int),1,fw);// new
    convert_data(&m->lstm_window,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    int p1 = 0, p2 = 0;
    if(m->p1 != NULL)
        p1 = 1;
    if(m->p2 != NULL)
        p2 = 1;
    
    
    convert_data(&p1,sizeof(int),1);// new
    i = fwrite(&p1,sizeof(int),1,fw);// new
    convert_data(&p1,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    convert_data(&p2,sizeof(int),1);// new
    i = fwrite(&p2,sizeof(int),1,fw);// new
    convert_data(&p2,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    if(p1)
        save_model(m->p1,n);
    if(p2)
        save_model(m->p2,n);
        
    save_model(m->rapresentation_h,n);
    save_model(m->dynamics_g,n);
    save_model(m->prediction_f,n);
    save_model(m->prediction_f_policy,n);
    save_model(m->prediction_f_value,n);
    save_model(m->reward_prediction_model,n);
    save_model(m->reward_prediction_temporal_model,n);
    save_rmodel(m->reward_prediction_rmodel,n);
    
    free(s);
}

efficientzeromodel* load_efficientzeromodel(char* file, int batch_size){
    
    
    if(file == NULL)
        return NULL;
    int i;
    FILE* fr = fopen(file,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",file);
        exit(1);
    }
    
    int batch_s, lstm_window, p1, p2;
    
    i = fread(&batch_s,sizeof(int),1,fr);
    convert_data(&batch_s,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the efficientzeromodel\n");
        exit(1);
    }
    
    i = fread(&lstm_window,sizeof(int),1,fr);
    convert_data(&lstm_window,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the efficientzeromodel\n");
        exit(1);
    }
    
    i = fread(&p1,sizeof(int),1,fr);
    convert_data(&p1,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the efficientzeromodel\n");
        exit(1);
    }
    
    i = fread(&p2,sizeof(int),1,fr);
    convert_data(&p2,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the efficientzeromodel\n");
        exit(1);
    }
    
    batch_s = batch_size;
    
    model* model_p1 = NULL;
    model* model_p2 = NULL;
    
    if(p1)
        model_p1 = load_model_with_file_already_opened(fr);
    if(p2)
        model_p2 = load_model_with_file_already_opened(fr);
    
    model* rapresentation_h = load_model_with_file_already_opened(fr);
    model* dynamics_g = load_model_with_file_already_opened(fr);
    model* prediction_f = load_model_with_file_already_opened(fr);
    model* prediction_f_policy = load_model_with_file_already_opened(fr);
    model* prediction_f_value = load_model_with_file_already_opened(fr);
    model* reward_prediction_model = load_model_with_file_already_opened(fr);
    model* reward_prediction_temporal_model = load_model_with_file_already_opened(fr);
    rmodel* reward_prediction_rmodel = load_rmodel_with_file_already_opened(fr);

    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file);
        exit(1);
    }
    
    efficientzeromodel* m = init_efficientzero_model(rapresentation_h,dynamics_g,prediction_f,prediction_f_policy,prediction_f_value,reward_prediction_model,reward_prediction_rmodel,reward_prediction_temporal_model,model_p1,model_p2,batch_size,lstm_window);
    
    return m;
    
}




void save_efficientzero_model_given_directory(efficientzeromodel* m, int n, char* directory){
    if(m == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* ss = (char*)malloc(sizeof(char)*256);
    ss[0] = '\0';
    char* t = ".bin";
    s = itoa_n(n,s);
    s = strcat(s,t);
    ss = strcat(ss,directory);
    ss = strcat(ss,s);
    fw = fopen(ss,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    convert_data(&m->batch_size,sizeof(int),1);// new
    i = fwrite(&m->batch_size,sizeof(int),1,fw);// new
    convert_data(&m->batch_size,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    convert_data(&m->lstm_window,sizeof(int),1);// new
    i = fwrite(&m->lstm_window,sizeof(int),1,fw);// new
    convert_data(&m->lstm_window,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    int p1 = 0, p2 = 0;
    if(m->p1 != NULL)
        p1 = 1;
    if(m->p2 != NULL)
        p2 = 1;
    
    
    convert_data(&p1,sizeof(int),1);// new
    i = fwrite(&p1,sizeof(int),1,fw);// new
    convert_data(&p1,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    convert_data(&p2,sizeof(int),1);// new
    i = fwrite(&p2,sizeof(int),1,fw);// new
    convert_data(&p2,sizeof(int),1);// new
    if(i != 1){// new
        fprintf(stderr,"Error: an error occurred saving a efficientzero model\n");// new
        exit(1);// new
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    if(p1)
        save_model(m->p1,n);
    if(p2)
        save_model(m->p2,n);
        
    save_model(m->rapresentation_h,n);
    save_model(m->dynamics_g,n);
    save_model(m->prediction_f,n);
    save_model(m->prediction_f_policy,n);
    save_model(m->prediction_f_value,n);
    save_model(m->reward_prediction_model,n);
    save_model(m->reward_prediction_temporal_model,n);
    save_rmodel(m->reward_prediction_rmodel,n);
    
    free(s);
    free(ss);
}



void reset_efficientzeromodel(efficientzeromodel* m){
    if(m == NULL)
        return;
    
    efficientzero_reset_dynamics_g(m);
    efficientzero_reset_rapresentation_h(m);
    efficientzero_reset_p1(m);
    efficientzero_reset_p2(m);
    efficientzero_reset_prediction_f(m);
    efficientzero_reset_reward_prediction(m);

}


void reset_efficientzeromodel_only_for_ff(efficientzeromodel* m){
    if(m == NULL)
        return;
    
    efficientzero_reset_only_for_ff_dynamics_g;
    efficientzero_reset_only_for_ff_rapresentation_h(m);
    efficientzero_reset_only_for_ff_p1(m);
    efficientzero_reset_only_for_ff_p2(m);
    efficientzero_reset_only_for_ff_prediction_f(m);
    efficientzero_reset_only_for_ff_reward_prediction(m);

}

void reset_efficientzeromodel_without_learning_parameters(efficientzeromodel* m){
    if(m == NULL)
        return;
    
    efficientzero_reset_dynamics_g_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_rapresentation_h_without_learning_parameters(m, m->batch_size);
    if(m->p1 != NULL)
    efficientzero_reset_p1_without_learning_parameters(m, m->batch_size);
    if(m->p2 != NULL)
    efficientzero_reset_p2_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_prediction_f_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_reward_prediction_without_learning_parameters(m, m->batch_size);

}


void reset_efficientzeromodel_only_for_ff_without_learning_parameters(efficientzeromodel* m){
    if(m == NULL)
        return;
    
    efficientzero_reset_only_for_ff_dynamics_g_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_only_for_ff_rapresentation_h_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_only_for_ff_p1_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_only_for_ff_p2_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_only_for_ff_prediction_f_without_learning_parameters(m, m->batch_size);
    efficientzero_reset_only_for_ff_reward_prediction_without_learning_parameters(m, m->batch_size);

}


efficientzeromodel* copy_efficientzero_model(efficientzeromodel* m){
    if(m == NULL)
        return NULL;
        
    model* p1 = copy_model(m->p1);
    model* p2 = copy_model(m->p2);
    model* rapresentation_h = copy_model(m->rapresentation_h);
    model* dynamics_g = copy_model(m->dynamics_g);
    model* prediction_f = copy_model(m->prediction_f);
    model* prediction_f_policy = copy_model(m->prediction_f_policy);
    model* prediction_f_value = copy_model(m->prediction_f_value);
    model* reward_prediction_model = copy_model(m->reward_prediction_model);
    model* reward_prediction_temporal_model = copy_model(m->reward_prediction_temporal_model);
    rmodel* reward_prediction_rmodel = copy_rmodel(m->reward_prediction_rmodel);
    
    return init_efficientzero_model(rapresentation_h,dynamics_g,prediction_f,prediction_f_policy,prediction_f_value,reward_prediction_model,reward_prediction_rmodel,reward_prediction_temporal_model,p1,p2,m->batch_size,m->lstm_window);
}


void paste_efficientzero_model(efficientzeromodel* m, efficientzeromodel* copy){
    if(m == NULL || copy == NULL)
        return;
    paste_model(m->p1,copy->p1);
    paste_model(m->p2,copy->p2);
    paste_model(m->rapresentation_h,copy->rapresentation_h);
    paste_model(m->dynamics_g,copy->dynamics_g);
    paste_model(m->prediction_f,copy->prediction_f);
    paste_model(m->prediction_f_value,copy->prediction_f_value);
    paste_model(m->prediction_f_policy,copy->prediction_f_policy);
    paste_model(m->reward_prediction_model,copy->reward_prediction_model);
    paste_model(m->reward_prediction_temporal_model,copy->reward_prediction_temporal_model);
    paste_rmodel(m->reward_prediction_rmodel,copy->reward_prediction_rmodel);
    return;
}

void slow_paste_efficientzero_model(efficientzeromodel* m, efficientzeromodel* copy, float tau){
    if(m == NULL || copy == NULL)
        return;
    slow_paste_model(m->rapresentation_h,copy->rapresentation_h,tau);
    slow_paste_model(m->dynamics_g,copy->dynamics_g,tau);
    slow_paste_model(m->prediction_f,copy->prediction_f,tau);
    slow_paste_model(m->prediction_f_policy,copy->prediction_f_policy,tau);
    slow_paste_model(m->prediction_f_value,copy->prediction_f_value,tau);
    slow_paste_model(m->reward_prediction_model,copy->reward_prediction_model,tau);
    slow_paste_model(m->reward_prediction_temporal_model,copy->reward_prediction_temporal_model,tau);
    slow_paste_rmodel(m->reward_prediction_rmodel,copy->reward_prediction_rmodel,tau);
    if(m->p1 != NULL)
        slow_paste_model(m->p1,copy->p1,tau);
    if(m->p2 != NULL)
        slow_paste_model(m->p2,copy->p2,tau);
    return;
}

uint64_t size_of_efficientzero_model(efficientzeromodel* m){
    uint64_t sum = 0;
    if(m->p1 != NULL)
    sum+=size_of_model(m->p1);
    if(m->p2 != NULL)
    sum+=size_of_model(m->p2);
    sum+=size_of_model(m->rapresentation_h);
    sum+=size_of_model(m->dynamics_g);
    sum+=size_of_model(m->prediction_f);
    sum+=size_of_model(m->prediction_f_value);
    sum+=size_of_model(m->prediction_f_policy);
    sum+=size_of_model(m->reward_prediction_model);
    sum+=size_of_model(m->reward_prediction_temporal_model);
    sum+=size_of_rmodel(m->reward_prediction_rmodel);
    return sum;
}

uint64_t size_of_efficientzero_model_without_learning_parameters(efficientzeromodel* m){
    uint64_t sum = 0;
    if(m->p1 != NULL)
    sum+=size_of_model(m->p1);
    if(m->p2 != NULL)
    sum+=size_of_model_without_learning_parameters(m->p2);
    sum+=size_of_model_without_learning_parameters(m->rapresentation_h);
    sum+=size_of_model_without_learning_parameters(m->dynamics_g);
    sum+=size_of_model_without_learning_parameters(m->prediction_f);
    sum+=size_of_model_without_learning_parameters(m->prediction_f_value);
    sum+=size_of_model_without_learning_parameters(m->prediction_f_policy);
    sum+=size_of_model_without_learning_parameters(m->reward_prediction_model);
    sum+=size_of_model_without_learning_parameters(m->reward_prediction_temporal_model);
    sum+=size_of_rmodel_without_learning_parameters(m->reward_prediction_rmodel);
    return sum;
}

uint64_t count_weights_efficientzero_model(efficientzeromodel* m){
    if(m == NULL)
        return 0;
    uint64_t sum = 0;
    if(m->p1 != NULL)
    sum+=count_weights(m->p1);
    if(m->p2 != NULL)
    sum+=count_weights(m->p2);
    sum+=count_weights(m->rapresentation_h);
    sum+=count_weights(m->dynamics_g);
    sum+=count_weights(m->prediction_f);
    sum+=count_weights(m->prediction_f_policy);
    sum+=count_weights(m->prediction_f_value);
    sum+=count_weights(m->reward_prediction_model);
    sum+=count_weights(m->reward_prediction_temporal_model);
    sum+=count_weights_rmodel(m->reward_prediction_rmodel);
    return sum;
}

uint64_t get_array_size_params_efficientzero_model(efficientzeromodel* m){
    if(m == NULL)
        return 0;
    uint64_t sum = 0;
    if(m->p1 != NULL)
    sum+=get_array_size_params_model(m->p1);
    if(m->p2 != NULL)
    sum+=get_array_size_params_model(m->p2);
    sum+=get_array_size_params_model(m->rapresentation_h);
    sum+=get_array_size_params_model(m->dynamics_g);
    sum+=get_array_size_params_model(m->prediction_f);
    sum+=get_array_size_params_model(m->prediction_f_policy);
    sum+=get_array_size_params_model(m->prediction_f_value);
    sum+=get_array_size_params_model(m->reward_prediction_model);
    sum+=get_array_size_params_model(m->reward_prediction_temporal_model);
    sum+=get_array_size_params_rmodel(m->reward_prediction_rmodel);
    return sum;
}



uint64_t get_array_size_scores_efficientzero_model(efficientzeromodel* m){
    if(m == NULL)
        return 0;
    uint64_t sum = 0;
    if(m->p1 != NULL)
    sum+=get_array_size_scores_model(m->p1);
    if(m->p2 != NULL)
    sum+=get_array_size_scores_model(m->p2);
    sum+=get_array_size_scores_model(m->rapresentation_h);
    sum+=get_array_size_scores_model(m->dynamics_g);
    sum+=get_array_size_scores_model(m->prediction_f);
    sum+=get_array_size_scores_model(m->prediction_f_policy);
    sum+=get_array_size_scores_model(m->prediction_f_value);
    sum+=get_array_size_scores_model(m->reward_prediction_model);
    sum+=get_array_size_scores_model(m->reward_prediction_temporal_model);
    sum+=get_array_size_scores_rmodel(m->reward_prediction_rmodel);
    return sum;
}


uint64_t get_array_size_weights_efficientzero_model(efficientzeromodel* m){
    if(m == NULL)
        return 0;
    uint64_t sum = 0;
    if(m->p1 != NULL)
    sum+=get_array_size_weights_model(m->p1);
    if(m->p2 != NULL)
    sum+=get_array_size_weights_model(m->p2);
    sum+=get_array_size_weights_model(m->rapresentation_h);
    sum+=get_array_size_weights_model(m->dynamics_g);
    sum+=get_array_size_weights_model(m->prediction_f);
    sum+=get_array_size_weights_model(m->prediction_f_policy);
    sum+=get_array_size_weights_model(m->prediction_f_value);
    sum+=get_array_size_weights_model(m->reward_prediction_model);
    sum+=get_array_size_weights_model(m->reward_prediction_temporal_model);
    sum+=get_array_size_weights_rmodel(m->reward_prediction_rmodel);
    return sum;
}

void memcopy_vector_to_params_efficientzero_model(efficientzeromodel* m, float* vector){
    if(m == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    if(m->p1 != NULL){
        memcopy_vector_to_params_model(m->p1,vector);
        sum+=get_array_size_params_model(m->p1);
    }
    if(m->p2 != NULL){
        memcopy_vector_to_params_model(m->p2,vector+sum);
        sum+=get_array_size_params_model(m->p2);
    }
    memcopy_vector_to_params_model(m->rapresentation_h,vector+sum);
    sum+=get_array_size_params_model(m->rapresentation_h);
    memcopy_vector_to_params_model(m->dynamics_g,vector+sum);
    sum+=get_array_size_params_model(m->dynamics_g);
    memcopy_vector_to_params_model(m->prediction_f,vector+sum);
    sum+=get_array_size_params_model(m->prediction_f);
    memcopy_vector_to_params_model(m->prediction_f_policy,vector+sum);
    sum+=get_array_size_params_model(m->prediction_f_policy);
    memcopy_vector_to_params_model(m->prediction_f_value,vector+sum);
    sum+=get_array_size_params_model(m->prediction_f_value);
    memcopy_vector_to_params_model(m->reward_prediction_model,vector+sum);
    sum+=get_array_size_params_model(m->reward_prediction_model);
    memcopy_vector_to_params_model(m->reward_prediction_temporal_model,vector+sum);
    sum+=get_array_size_params_model(m->reward_prediction_temporal_model);
    memcopy_vector_to_params_rmodel(m->reward_prediction_rmodel,vector+sum);
    sum+=get_array_size_params_rmodel(m->reward_prediction_rmodel);
}


void memcopy_vector_to_scores_efficientzero_model(efficientzeromodel* m, float* vector){
    if(m == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    if(m->p1 != NULL){
        memcopy_vector_to_scores_model(m->p1,vector);
        sum+=get_array_size_scores_model(m->p1);
    }
    if(m->p2 != NULL){
        memcopy_vector_to_scores_model(m->p2,vector+sum);
        sum+=get_array_size_scores_model(m->p2);
    }
    
    memcopy_vector_to_scores_model(m->rapresentation_h,vector+sum);
    sum+=get_array_size_scores_model(m->rapresentation_h);
    memcopy_vector_to_scores_model(m->dynamics_g,vector+sum);
    sum+=get_array_size_scores_model(m->dynamics_g);
    memcopy_vector_to_scores_model(m->prediction_f,vector+sum);
    sum+=get_array_size_scores_model(m->prediction_f);
    memcopy_vector_to_scores_model(m->prediction_f_policy,vector+sum);
    sum+=get_array_size_scores_model(m->prediction_f_policy);
    memcopy_vector_to_scores_model(m->prediction_f_value,vector+sum);
    sum+=get_array_size_scores_model(m->prediction_f_value);
    memcopy_vector_to_scores_model(m->reward_prediction_model,vector+sum);
    sum+=get_array_size_scores_model(m->reward_prediction_model);
    memcopy_vector_to_scores_model(m->reward_prediction_temporal_model,vector+sum);
    sum+=get_array_size_scores_model(m->reward_prediction_temporal_model);
    memcopy_vector_to_scores_rmodel(m->reward_prediction_rmodel,vector+sum);
    sum+=get_array_size_scores_rmodel(m->reward_prediction_rmodel);
    
}

void memcopy_params_to_vector_efficientzero_model(efficientzeromodel* m, float* vector){
    if(m == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    if(m->p1 != NULL){
        memcopy_params_to_vector_model(m->p1,vector);
        sum+=get_array_size_params_model(m->p1);
    }
    if(m->p2 != NULL){
        memcopy_params_to_vector_model(m->p2,vector+sum);
        sum+=get_array_size_params_model(m->p2);
    }
    
    
    memcopy_params_to_vector_model(m->rapresentation_h,vector+sum);
    sum+=get_array_size_params_model(m->rapresentation_h);
    memcopy_params_to_vector_model(m->dynamics_g,vector+sum);
    sum+=get_array_size_params_model(m->dynamics_g);
    memcopy_params_to_vector_model(m->prediction_f,vector+sum);
    sum+=get_array_size_params_model(m->prediction_f);
    memcopy_params_to_vector_model(m->prediction_f_policy,vector+sum);
    sum+=get_array_size_params_model(m->prediction_f_policy);
    memcopy_params_to_vector_model(m->prediction_f_value,vector+sum);
    sum+=get_array_size_params_model(m->prediction_f_value);
    memcopy_params_to_vector_model(m->reward_prediction_model,vector+sum);
    sum+=get_array_size_params_model(m->reward_prediction_model);
    memcopy_params_to_vector_model(m->reward_prediction_temporal_model,vector+sum);
    sum+=get_array_size_params_model(m->reward_prediction_temporal_model);
    memcopy_params_to_vector_rmodel(m->reward_prediction_rmodel,vector+sum);
    sum+=get_array_size_params_rmodel(m->reward_prediction_rmodel);
    
}


void memcopy_weights_to_vector_efficientzero_model(efficientzeromodel* m, float* vector){
    if(m == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    
    if(m->p1 != NULL){
        memcopy_weights_to_vector_model(m->p1,vector);
        sum+=get_array_size_weights_model(m->p1);
    }
    if(m->p2 != NULL){
        memcopy_weights_to_vector_model(m->p2,vector+sum);
        sum+=get_array_size_weights_model(m->p2);
    }
    
    memcopy_weights_to_vector_model(m->rapresentation_h,vector+sum);
    sum+=get_array_size_weights_model(m->rapresentation_h);
    memcopy_weights_to_vector_model(m->dynamics_g,vector+sum);
    sum+=get_array_size_weights_model(m->dynamics_g);
    memcopy_weights_to_vector_model(m->prediction_f,vector+sum);
    sum+=get_array_size_weights_model(m->prediction_f);
    memcopy_weights_to_vector_model(m->prediction_f_policy,vector+sum);
    sum+=get_array_size_weights_model(m->prediction_f_policy);
    memcopy_weights_to_vector_model(m->prediction_f_value,vector+sum);
    sum+=get_array_size_weights_model(m->prediction_f_value);
    memcopy_weights_to_vector_model(m->reward_prediction_model,vector+sum);
    sum+=get_array_size_weights_model(m->reward_prediction_model);
    memcopy_weights_to_vector_model(m->reward_prediction_temporal_model,vector+sum);
    sum+=get_array_size_weights_model(m->reward_prediction_temporal_model);
    memcopy_weights_to_vector_rmodel(m->reward_prediction_rmodel,vector+sum);
    sum+=get_array_size_weights_rmodel(m->reward_prediction_rmodel);

    
}


void memcopy_vector_to_weights_efficientzero_model(efficientzeromodel* m, float* vector){
    if(m == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    if(m->p1 != NULL){
        memcopy_vector_to_weights_model(m->p1,vector);
        sum+=get_array_size_weights_model(m->p1);
    }
    if(m->p2 != NULL){
        memcopy_vector_to_weights_model(m->p2,vector+sum);
        sum+=get_array_size_weights_model(m->p2);
    }
    
    memcopy_vector_to_weights_model(m->rapresentation_h,vector+sum);
    sum+=get_array_size_weights_model(m->rapresentation_h);
    memcopy_vector_to_weights_model(m->dynamics_g,vector+sum);
    sum+=get_array_size_weights_model(m->dynamics_g);
    memcopy_vector_to_weights_model(m->prediction_f,vector+sum);
    sum+=get_array_size_weights_model(m->prediction_f);
    memcopy_vector_to_weights_model(m->prediction_f_policy,vector+sum);
    sum+=get_array_size_weights_model(m->prediction_f_policy);
    memcopy_vector_to_weights_model(m->prediction_f_value,vector+sum);
    sum+=get_array_size_weights_model(m->prediction_f_value);
    memcopy_vector_to_weights_model(m->reward_prediction_model,vector+sum);
    sum+=get_array_size_weights_model(m->reward_prediction_model);
    memcopy_vector_to_weights_model(m->reward_prediction_temporal_model,vector+sum);
    sum+=get_array_size_weights_model(m->reward_prediction_temporal_model);
    memcopy_vector_to_weights_rmodel(m->reward_prediction_rmodel,vector+sum);
    sum+=get_array_size_weights_rmodel(m->reward_prediction_rmodel);
}


void memcopy_scores_to_vector_efficientzero_model(efficientzeromodel* m, float* vector){
    if(m == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    
    if(m->p1 != NULL){
        memcopy_scores_to_vector_model(m->p1,vector);
        sum+=get_array_size_scores_model(m->p1);
    }
    if(m->p2 != NULL){
        memcopy_scores_to_vector_model(m->p2,vector+sum);
        sum+=get_array_size_scores_model(m->p2);
    }
    
    memcopy_scores_to_vector_model(m->rapresentation_h,vector+sum);
    sum+=get_array_size_scores_model(m->rapresentation_h);
    memcopy_scores_to_vector_model(m->dynamics_g,vector+sum);
    sum+=get_array_size_scores_model(m->dynamics_g);
    memcopy_scores_to_vector_model(m->prediction_f,vector+sum);
    sum+=get_array_size_scores_model(m->prediction_f);
    memcopy_scores_to_vector_model(m->prediction_f_policy,vector+sum);
    sum+=get_array_size_scores_model(m->prediction_f_policy);
    memcopy_scores_to_vector_model(m->prediction_f_value,vector+sum);
    sum+=get_array_size_scores_model(m->prediction_f_value);
    memcopy_scores_to_vector_model(m->reward_prediction_model,vector+sum);
    sum+=get_array_size_scores_model(m->reward_prediction_model);
    memcopy_scores_to_vector_model(m->reward_prediction_temporal_model,vector+sum);
    sum+=get_array_size_scores_model(m->reward_prediction_temporal_model);
    memcopy_scores_to_vector_rmodel(m->reward_prediction_rmodel,vector+sum);
    sum+=get_array_size_scores_rmodel(m->reward_prediction_rmodel);

    
}


void set_efficientzero_model_beta(efficientzeromodel* m, float b1, float b2){
    if(m == NULL)
        return;
    if(m->p1 != NULL)
        set_model_beta(m->p1,b1,b2);
    if(m->p2 != NULL)
        set_model_beta(m->p2,b1,b2);
    set_model_beta(m->rapresentation_h,b1,b2);
    set_model_beta(m->dynamics_g,b1,b2);
    set_model_beta(m->prediction_f,b1,b2);
    set_model_beta(m->prediction_f_policy,b1,b2);
    set_model_beta(m->prediction_f_value,b1,b2);
    set_model_beta(m->reward_prediction_model,b1,b2);
    set_model_beta(m->reward_prediction_temporal_model,b1,b2);
    set_rmodel_beta(m->reward_prediction_rmodel,b1,b2);
    return;
}

void set_efficientzero_model_beta_adamond(efficientzeromodel* m, float b1){
    if(m == NULL)
        return;
    if(m->p1 != NULL)
        set_model_beta_adamod(m->p1,b1);
    if(m->p2 != NULL)
        set_model_beta_adamod(m->p2,b1);
    set_model_beta_adamod(m->rapresentation_h,b1);
    set_model_beta_adamod(m->dynamics_g,b1);
    set_model_beta_adamod(m->prediction_f,b1);
    set_model_beta_adamod(m->prediction_f_policy,b1);
    set_model_beta_adamod(m->prediction_f_value,b1);
    set_model_beta_adamod(m->reward_prediction_model,b1);
    set_model_beta_adamod(m->reward_prediction_temporal_model,b1);
    set_rmodel_beta_adamod(m->reward_prediction_rmodel,b1);
    return;
}





