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

#include "dueling_categorical_dqn.h"

dueling_categorical_dqn* dueling_categorical_dqn_init(int input_size, int action_size, int n_atoms, float v_min, float v_max, model* shared_hidden_layers, model* v_hidden_layers, model* a_hidden_layers, model* v_linear_last_layer, model* a_linear_last_layer){
    if(shared_hidden_layers == NULL || v_hidden_layers == NULL || a_hidden_layers == NULL || v_linear_last_layer == NULL || a_linear_last_layer == NULL){
        fprintf(stderr,"Error: you cannot have null model passed as input!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->layers <= 0 || v_hidden_layers->layers <= 0 || a_hidden_layers->layers <= 0 || v_linear_last_layer->layers != 1 || a_linear_last_layer->layers != 1){
        fprintf(stderr,"Error: your number of layers for some of your model is not correct! Remember: for linear models you can only have 1 layer and no activation should be performed!\n");
        exit(1);
    }
    
    if(input_size <= 0){
        fprintf(stderr,"Error: invalid input size (must be > 0)\n");
        exit(1);
    }
    
    if(n_atoms <= 1){
        fprintf(stderr,"Error: atmos must be > 1\n");
        exit(1);
    }
    
    if(action_size <= 0){
        fprintf(stderr,"error: action size must be > 0\n");
        exit(1);
    }
    
    if(v_min > v_max){
        fprintf(stderr,"Error: v_min should be <= v_max\n");
        exit(1);
    }
    
    if(get_input_layer_size(shared_hidden_layers) != input_size){
        fprintf(stderr,"Error: invalid input size: it doesn't match the shared model!\n");
        exit(1);
    }
    
    if(action_size*n_atoms != a_linear_last_layer->output_dimension){
        fprintf(stderr,"Error: you action size* n_atoms doesn't match the output dimension of action_linear model!\n");
        exit(1);
    }
    
    if(v_linear_last_layer->output_dimension != n_atoms){
        fprintf(stderr,"Error: you v linear model should have an output dimension of 1!\n");
        exit(1);
    }
    
    if(v_linear_last_layer->n_cl != 0 || v_linear_last_layer->n_rl != 0 || v_linear_last_layer->n_fcl != 1){
        fprintf(stderr,"Error: your v linear model should have only a fully connected layer!\n");
        exit(1);
    }
    
    if(a_linear_last_layer->n_cl != 0 || a_linear_last_layer->n_rl != 0 || a_linear_last_layer->n_fcl != 1){
        fprintf(stderr,"Error: your a linear model should have only a fully connected layer!\n");
        exit(1);
    }
    
    if(v_linear_last_layer->fcls[0]->activation_flag || v_linear_last_layer->fcls[0]->normalization_flag){
        fprintf(stderr,"Error: you should not have activation neither normalization for your last fully connected layers!\n");
        exit(1);
    }
    if(a_linear_last_layer->fcls[0]->activation_flag || a_linear_last_layer->fcls[0]->normalization_flag){
        fprintf(stderr,"Error: you should not have activation neither normalization for your last fully connected layers!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->output_dimension != get_input_layer_size(v_hidden_layers)){
        fprintf(stderr,"Error: final dimension of shared hidden layers doesn't match the input dimension of the v_hidden_layers!\n"),
        exit(1);
    }
    if(shared_hidden_layers->output_dimension != get_input_layer_size(a_hidden_layers)){
        fprintf(stderr,"Error: final dimension of shared hidden layers doesn't match the input dimension of the a_hidden_layers!\n"),
        exit(1);
    }
    if(v_hidden_layers->output_dimension != get_input_layer_size(v_linear_last_layer)){
        fprintf(stderr,"Error: final dimension of v_hidden_layers doesn't match the input dimension of the v linear last layer!\n"),
        exit(1);
    }
    if(a_hidden_layers->output_dimension != get_input_layer_size(a_linear_last_layer)){
        fprintf(stderr,"Error: final dimension of a_hidden_layers doesn't match the input dimension of the a linear last layer!\n"),
        exit(1);
    }
    
    if(shared_hidden_layers->beta1_adam != v_hidden_layers->beta1_adam || shared_hidden_layers->beta1_adam != v_linear_last_layer->beta1_adam || shared_hidden_layers->beta1_adam != a_hidden_layers->beta1_adam || shared_hidden_layers->beta1_adam != a_linear_last_layer->beta1_adam){
        fprintf(stderr,"Error: all hyperparameters must be equals in the models!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->beta2_adam != v_hidden_layers->beta2_adam || shared_hidden_layers->beta2_adam != v_linear_last_layer->beta2_adam || shared_hidden_layers->beta2_adam != a_hidden_layers->beta2_adam || shared_hidden_layers->beta2_adam != a_linear_last_layer->beta2_adam){
        fprintf(stderr,"Error: all hyperparameters must be equals in the models!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->beta3_adamod != v_hidden_layers->beta3_adamod || shared_hidden_layers->beta3_adamod != v_linear_last_layer->beta3_adamod || shared_hidden_layers->beta3_adamod != a_hidden_layers->beta3_adamod || shared_hidden_layers->beta3_adamod != a_linear_last_layer->beta3_adamod){
        fprintf(stderr,"Error: all hyperparameters must be equals in the models!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = (dueling_categorical_dqn*)malloc(sizeof(dueling_categorical_dqn));
    dqn->shared_hidden_layers = shared_hidden_layers;
    dqn->v_hidden_layers = v_hidden_layers;
    dqn->a_hidden_layers = a_hidden_layers;
    dqn->v_linear_last_layer = v_linear_last_layer;
    dqn->a_linear_last_layer = a_linear_last_layer;
    dqn->n_atoms = n_atoms;
    dqn->input_size = input_size;
    dqn->v_min = v_min;
    dqn->v_max = v_max;
    dqn->z_delta = (v_max-v_min)/((float)(n_atoms-1));
    dqn->action_size = action_size;
    dqn->action_mean_layer = (float*)calloc(n_atoms,sizeof(float));
    dqn->v_linear_layer_error = (float*)calloc(n_atoms,sizeof(float));
    dqn->add_layer = (float*)calloc(action_size*n_atoms,sizeof(float));
    dqn->a_linear_layer_error = (float*)calloc(action_size*n_atoms,sizeof(float));
    dqn->error = (float*)calloc(action_size*n_atoms,sizeof(float));
    dqn->softmax_layer = (float*)calloc(action_size*n_atoms,sizeof(float));
    dqn->derivative_softmax_layer = (float*)calloc(action_size*n_atoms,sizeof(float));
    dqn->support = (float*)malloc(sizeof(float)*n_atoms);
    dqn->q_functions = (float*)calloc(action_size,sizeof(float));
    int i;
    for(i = 0; i < n_atoms-1; i++){
        dqn->support[i] = v_min + (float)(i*(dqn->z_delta));
    }
    dqn->support[i] = v_max;
    
    return dqn;
}

dueling_categorical_dqn* dueling_categorical_dqn_init_without_arrays(int input_size, int action_size, int n_atoms, float v_min, float v_max, model* shared_hidden_layers, model* v_hidden_layers, model* a_hidden_layers, model* v_linear_last_layer, model* a_linear_last_layer){
    if(shared_hidden_layers == NULL || v_hidden_layers == NULL || a_hidden_layers == NULL || v_linear_last_layer == NULL || a_linear_last_layer == NULL){
        fprintf(stderr,"Error: you cannot have null model passed as input!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->layers <= 0 || v_hidden_layers->layers <= 0 || a_hidden_layers->layers <= 0 || v_linear_last_layer->layers != 1 || a_linear_last_layer->layers != 1){
        fprintf(stderr,"Error: your number of layers for some of your model is not correct! Remember: for linear models you can only have 1 layer and no activation should be performed!\n");
        exit(1);
    }
    
    if(input_size <= 0){
        fprintf(stderr,"Error: invalid input size (must be > 0)\n");
        exit(1);
    }
    
    if(n_atoms <= 1){
        fprintf(stderr,"Error: atmos must be > 1\n");
        exit(1);
    }
    
    if(action_size <= 0){
        fprintf(stderr,"error: action size must be > 0\n");
        exit(1);
    }
    
    if(v_min > v_max){
        fprintf(stderr,"Error: v_min should be <= v_max\n");
        exit(1);
    }
    
    if(get_input_layer_size(shared_hidden_layers) != input_size){
        fprintf(stderr,"Error: invalid input size: it doesn't match the shared model!\n");
        exit(1);
    }
    
    if(action_size*n_atoms != a_linear_last_layer->output_dimension){
        fprintf(stderr,"Error: you action size* n_atoms doesn't match the output dimension of action_linear model!\n");
        exit(1);
    }
    
    if(v_linear_last_layer->output_dimension != n_atoms){
        fprintf(stderr,"Error: you v linear model should have an output dimension of 1!\n");
        exit(1);
    }
    
    if(v_linear_last_layer->n_cl != 0 || v_linear_last_layer->n_rl != 0 || v_linear_last_layer->n_fcl != 1){
        fprintf(stderr,"Error: your v linear model should have only a fully connected layer!\n");
        exit(1);
    }
    
    if(a_linear_last_layer->n_cl != 0 || a_linear_last_layer->n_rl != 0 || a_linear_last_layer->n_fcl != 1){
        fprintf(stderr,"Error: your a linear model should have only a fully connected layer!\n");
        exit(1);
    }
    
    if(v_linear_last_layer->fcls[0]->activation_flag || v_linear_last_layer->fcls[0]->normalization_flag){
        fprintf(stderr,"Error: you should not have activation neither normalization for your last fully connected layers!\n");
        exit(1);
    }
    if(a_linear_last_layer->fcls[0]->activation_flag || a_linear_last_layer->fcls[0]->normalization_flag){
        fprintf(stderr,"Error: you should not have activation neither normalization for your last fully connected layers!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->output_dimension != get_input_layer_size(v_hidden_layers)){
        fprintf(stderr,"Error: final dimension of shared hidden layers doesn't match the input dimension of the v_hidden_layers!\n"),
        exit(1);
    }
    if(shared_hidden_layers->output_dimension != get_input_layer_size(a_hidden_layers)){
        fprintf(stderr,"Error: final dimension of shared hidden layers doesn't match the input dimension of the a_hidden_layers!\n"),
        exit(1);
    }
    if(v_hidden_layers->output_dimension != get_input_layer_size(v_linear_last_layer)){
        fprintf(stderr,"Error: final dimension of v_hidden_layers doesn't match the input dimension of the v linear last layer!\n"),
        exit(1);
    }
    if(a_hidden_layers->output_dimension != get_input_layer_size(a_linear_last_layer)){
        fprintf(stderr,"Error: final dimension of a_hidden_layers doesn't match the input dimension of the a linear last layer!\n"),
        exit(1);
    }
    if(shared_hidden_layers->beta1_adam != v_hidden_layers->beta1_adam || shared_hidden_layers->beta1_adam != v_linear_last_layer->beta1_adam || shared_hidden_layers->beta1_adam != a_hidden_layers->beta1_adam || shared_hidden_layers->beta1_adam != a_linear_last_layer->beta1_adam){
        fprintf(stderr,"Error: all hyperparameters must be equals in the models!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->beta2_adam != v_hidden_layers->beta2_adam || shared_hidden_layers->beta2_adam != v_linear_last_layer->beta2_adam || shared_hidden_layers->beta2_adam != a_hidden_layers->beta2_adam || shared_hidden_layers->beta2_adam != a_linear_last_layer->beta2_adam){
        fprintf(stderr,"Error: all hyperparameters must be equals in the models!\n");
        exit(1);
    }
    
    if(shared_hidden_layers->beta3_adamod != v_hidden_layers->beta3_adamod || shared_hidden_layers->beta3_adamod != v_linear_last_layer->beta3_adamod || shared_hidden_layers->beta3_adamod != a_hidden_layers->beta3_adamod || shared_hidden_layers->beta3_adamod != a_linear_last_layer->beta3_adamod){
        fprintf(stderr,"Error: all hyperparameters must be equals in the models!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = (dueling_categorical_dqn*)malloc(sizeof(dueling_categorical_dqn));
    dqn->shared_hidden_layers = shared_hidden_layers;
    dqn->v_hidden_layers = v_hidden_layers;
    dqn->a_hidden_layers = a_hidden_layers;
    dqn->v_linear_last_layer = v_linear_last_layer;
    dqn->a_linear_last_layer = a_linear_last_layer;
    dqn->n_atoms = n_atoms;
    dqn->input_size = input_size;
    dqn->v_min = v_min;
    dqn->v_max = v_max;
    dqn->z_delta = (v_max-v_min)/((float)(n_atoms-1));
    dqn->action_size = action_size;
    return dqn;
}

void save_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int n){
    if(dqn == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&dqn->input_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    i = fwrite(&dqn->action_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    i = fwrite(&dqn->n_atoms,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    i = fwrite(&dqn->v_min,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    i = fwrite(&dqn->v_max,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    save_model(dqn->shared_hidden_layers,n);
    save_model(dqn->v_hidden_layers,n);
    save_model(dqn->v_linear_last_layer,n);
    save_model(dqn->a_hidden_layers,n);
    save_model(dqn->a_linear_last_layer,n);
    free(s);
}


dueling_categorical_dqn* load_dueling_categorical_dqn(char* file){
    if(file == NULL)
        return NULL;
    int i;
    FILE* fr = fopen(file,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",file);
        exit(1);
    }
    
    int input_size, action_size, n_atoms;
    float v_min,v_max;
    
    i = fread(&input_size,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&action_size,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_atoms,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&v_min,sizeof(float),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&v_max,sizeof(float),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    
    model* shared_hidden_layers = load_model_with_file_already_opened(fr);
    model* v_hidden_layers = load_model_with_file_already_opened(fr);
    model* v_linear_last_layer = load_model_with_file_already_opened(fr);
    model* a_hidden_layers = load_model_with_file_already_opened(fr);
    model* a_linear_last_layer = load_model_with_file_already_opened(fr);
    
    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file);
        exit(1);
    }
    
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init(input_size,action_size,n_atoms,v_min,v_max,shared_hidden_layers,v_hidden_layers,a_hidden_layers,v_linear_last_layer,a_linear_last_layer);
    
    return dqn;
    
}


void save_dueling_categorical_dqn_given_directory(dueling_categorical_dqn* dqn, int n, char* directory){
    if(dqn == NULL)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* ss = (char*)malloc(sizeof(char)*256);
    ss[0] = '\0';
    char* t = ".bin";
    s = itoa(n,s);
    s = strcat(s,t);
    ss = strcat(ss,directory);
    ss = strcat(ss,s);
    fw = fopen(ss,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    i = fwrite(&dqn->input_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    i = fwrite(&dqn->action_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    i = fwrite(&dqn->n_atoms,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    i = fwrite(&dqn->v_min,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    i = fwrite(&dqn->v_max,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    save_model_given_directory(dqn->shared_hidden_layers,n,directory);
    save_model_given_directory(dqn->v_hidden_layers,n,directory);
    save_model_given_directory(dqn->v_linear_last_layer,n,directory);
    save_model_given_directory(dqn->a_hidden_layers,n,directory);
    save_model_given_directory(dqn->a_linear_last_layer,n,directory);
    
    free(s);
    free(ss);
}

void free_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    free_model(dqn->shared_hidden_layers);
    free_model(dqn->v_hidden_layers);
    free_model(dqn->a_hidden_layers);
    free_model(dqn->v_linear_last_layer);
    free_model(dqn->a_linear_last_layer);
    free(dqn->action_mean_layer);
    free(dqn->add_layer);
    free(dqn->softmax_layer);
    free(dqn->derivative_softmax_layer);
    free(dqn->v_linear_layer_error);
    free(dqn->a_linear_layer_error);
    free(dqn->error);
    free(dqn->support);
    free(dqn->q_functions);
    free(dqn);
    return;
}

void free_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    free_model_without_learning_parameters(dqn->shared_hidden_layers);
    free_model_without_learning_parameters(dqn->v_hidden_layers);
    free_model_without_learning_parameters(dqn->a_hidden_layers);
    free_model_without_learning_parameters(dqn->v_linear_last_layer);
    free_model_without_learning_parameters(dqn->a_linear_last_layer);
    free(dqn->action_mean_layer);
    free(dqn->add_layer);
    free(dqn->softmax_layer);
    free(dqn->derivative_softmax_layer);
    free(dqn->v_linear_layer_error);
    free(dqn->a_linear_layer_error);
    free(dqn->error);
    free(dqn->support);
    free(dqn->q_functions);
    free(dqn);
    return;
}

void free_dueling_categorical_dqn_without_arrays(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    free_model_without_arrays(dqn->shared_hidden_layers);
    free_model_without_arrays(dqn->v_hidden_layers);
    free_model_without_arrays(dqn->a_hidden_layers);
    free_model_without_arrays(dqn->v_linear_last_layer);
    free_model_without_arrays(dqn->a_linear_last_layer);
    free(dqn);
    return;
}

void reset_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    reset_model(dqn->shared_hidden_layers);
    reset_model(dqn->v_hidden_layers);
    reset_model(dqn->a_hidden_layers);
    reset_model(dqn->v_linear_last_layer);
    reset_model(dqn->a_linear_last_layer);
    set_vector_with_value(0,dqn->action_mean_layer,dqn->n_atoms);
    set_vector_with_value(0,dqn->v_linear_layer_error,dqn->n_atoms);
    set_vector_with_value(0,dqn->q_functions,dqn->action_size);
    set_vector_with_value(0,dqn->add_layer,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->a_linear_layer_error,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->error,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->softmax_layer,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->derivative_softmax_layer,dqn->action_size*dqn->n_atoms);
}

void reset_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    reset_model_without_learning_parameters(dqn->shared_hidden_layers);
    reset_model_without_learning_parameters(dqn->v_hidden_layers);
    reset_model_without_learning_parameters(dqn->a_hidden_layers);
    reset_model_without_learning_parameters(dqn->v_linear_last_layer);
    reset_model_without_learning_parameters(dqn->a_linear_last_layer);
    set_vector_with_value(0,dqn->action_mean_layer,dqn->n_atoms);
    set_vector_with_value(0,dqn->v_linear_layer_error,dqn->n_atoms);
    set_vector_with_value(0,dqn->add_layer,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->a_linear_layer_error,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->error,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->softmax_layer,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->derivative_softmax_layer,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->q_functions,dqn->action_size);
}

dueling_categorical_dqn* copy_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return NULL;
    model* shared_hidden_layers = copy_model(dqn->shared_hidden_layers);
    model* v_hidden_layers = copy_model(dqn->v_hidden_layers);
    model* a_hidden_layers = copy_model(dqn->a_hidden_layers);
    model* v_linear_last_layer = copy_model(dqn->v_linear_last_layer);
    model* a_linear_last_layer = copy_model(dqn->a_linear_last_layer);
    return dueling_categorical_dqn_init(dqn->input_size,dqn->action_size,dqn->n_atoms,dqn->v_min,dqn->v_max,shared_hidden_layers,v_hidden_layers,a_hidden_layers,v_linear_last_layer,a_linear_last_layer);
}

dueling_categorical_dqn* copy_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return NULL;
    model* shared_hidden_layers = copy_model_without_learning_parameters(dqn->shared_hidden_layers);
    model* v_hidden_layers = copy_model_without_learning_parameters(dqn->v_hidden_layers);
    model* a_hidden_layers = copy_model_without_learning_parameters(dqn->a_hidden_layers);
    model* v_linear_last_layer = copy_model_without_learning_parameters(dqn->v_linear_last_layer);
    model* a_linear_last_layer = copy_model_without_learning_parameters(dqn->a_linear_last_layer);
    return dueling_categorical_dqn_init(dqn->input_size,dqn->action_size,dqn->n_atoms,dqn->v_min,dqn->v_max,shared_hidden_layers,v_hidden_layers,a_hidden_layers,v_linear_last_layer,a_linear_last_layer);
}


void paste_dueling_categorical_dqn(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy){
    if(dqn == NULL || copy == NULL)
        return;
    paste_model(dqn->shared_hidden_layers,copy->shared_hidden_layers);
    paste_model(dqn->v_hidden_layers,copy->v_hidden_layers);
    paste_model(dqn->a_hidden_layers,copy->a_hidden_layers);
    paste_model(dqn->v_linear_last_layer,copy->v_linear_last_layer);
    paste_model(dqn->a_linear_last_layer,copy->a_linear_last_layer);
    return;
}


void paste_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy){
    if(dqn == NULL || copy == NULL)
        return;
    paste_model_without_learning_parameters(dqn->shared_hidden_layers,copy->shared_hidden_layers);
    paste_model_without_learning_parameters(dqn->v_hidden_layers,copy->v_hidden_layers);
    paste_model_without_learning_parameters(dqn->a_hidden_layers,copy->a_hidden_layers);
    paste_model_without_learning_parameters(dqn->v_linear_last_layer,copy->v_linear_last_layer);
    paste_model_without_learning_parameters(dqn->a_linear_last_layer,copy->a_linear_last_layer);
    return;
}

void slow_paste_dueling_categorical_dqn(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy, float tau){
    if(dqn == NULL || copy == NULL)
        return;
    slow_paste_model(dqn->shared_hidden_layers,copy->shared_hidden_layers,tau);
    slow_paste_model(dqn->v_hidden_layers,copy->v_hidden_layers, tau);
    slow_paste_model(dqn->a_hidden_layers,copy->a_hidden_layers, tau);
    slow_paste_model(dqn->v_linear_last_layer,copy->v_linear_last_layer, tau);
    slow_paste_model(dqn->a_linear_last_layer,copy->a_linear_last_layer, tau);
    return;
}

uint64_t size_of_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    uint64_t sum = 0;
    sum+=size_of_model(dqn->shared_hidden_layers);
    sum+=size_of_model(dqn->v_hidden_layers);
    sum+=size_of_model(dqn->a_hidden_layers);
    sum+=size_of_model(dqn->v_linear_last_layer);
    sum+=size_of_model(dqn->a_linear_last_layer);
    sum+=dqn->action_size*dqn->n_atoms*5+3*dqn->n_atoms+dqn->action_size;
    return sum;
}

uint64_t size_of_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn){
    uint64_t sum = 0;
    sum+=size_of_model_without_learning_parameters(dqn->shared_hidden_layers);
    sum+=size_of_model_without_learning_parameters(dqn->v_hidden_layers);
    sum+=size_of_model_without_learning_parameters(dqn->a_hidden_layers);
    sum+=size_of_model_without_learning_parameters(dqn->v_linear_last_layer);
    sum+=size_of_model_without_learning_parameters(dqn->a_linear_last_layer);
    sum+=dqn->action_size*dqn->n_atoms*5+3*dqn->n_atoms+dqn->action_size;
    return sum;
}

uint64_t count_weights_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    uint64_t sum = 0;
    sum+=count_weights(dqn->shared_hidden_layers);
    sum+=count_weights(dqn->v_hidden_layers);
    sum+=count_weights(dqn->v_linear_last_layer);
    sum+=count_weights(dqn->a_hidden_layers);
    sum+=count_weights(dqn->a_linear_last_layer);
    return sum;
}

uint64_t get_array_size_params_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    uint64_t sum = 0;
    sum+=get_array_size_params_model(dqn->shared_hidden_layers);
    sum+=get_array_size_params_model(dqn->v_hidden_layers);
    sum+=get_array_size_params_model(dqn->v_linear_last_layer);
    sum+=get_array_size_params_model(dqn->a_hidden_layers);
    sum+=get_array_size_params_model(dqn->a_linear_last_layer);
    return sum;
}



uint64_t get_array_size_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    uint64_t sum = 0;
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    sum+=get_array_size_scores_model(dqn->a_linear_last_layer);
    return sum;
}


uint64_t get_array_size_weights_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    uint64_t sum = 0;
    sum+=get_array_size_weights_model(dqn->shared_hidden_layers);
    sum+=get_array_size_weights_model(dqn->v_hidden_layers);
    sum+=get_array_size_weights_model(dqn->v_linear_last_layer);
    sum+=get_array_size_weights_model(dqn->a_hidden_layers);
    sum+=get_array_size_weights_model(dqn->a_linear_last_layer);
    return sum;
}

void memcopy_vector_to_params_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_vector_to_params_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_params_model(dqn->shared_hidden_layers);
    memcopy_vector_to_params_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_params_model(dqn->v_hidden_layers);
    memcopy_vector_to_params_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_params_model(dqn->v_linear_last_layer);
    memcopy_vector_to_params_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_params_model(dqn->a_hidden_layers);
    memcopy_vector_to_params_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_params_model(dqn->a_linear_last_layer);
}


void memcopy_vector_to_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_vector_to_scores_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    memcopy_vector_to_scores_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    memcopy_vector_to_scores_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    memcopy_vector_to_scores_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    memcopy_vector_to_scores_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_linear_last_layer);
}


void memcopy_params_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_params_to_vector_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_params_model(dqn->shared_hidden_layers);
    memcopy_params_to_vector_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_params_model(dqn->v_hidden_layers);
    memcopy_params_to_vector_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_params_model(dqn->v_linear_last_layer);
    memcopy_params_to_vector_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_params_model(dqn->a_hidden_layers);
    memcopy_params_to_vector_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_params_model(dqn->a_linear_last_layer);
}


void memcopy_weights_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_weights_to_vector_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_weights_model(dqn->shared_hidden_layers);
    memcopy_weights_to_vector_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_weights_model(dqn->v_hidden_layers);
    memcopy_weights_to_vector_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_weights_model(dqn->v_linear_last_layer);
    memcopy_weights_to_vector_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_weights_model(dqn->a_hidden_layers);
    memcopy_weights_to_vector_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_weights_model(dqn->a_linear_last_layer);
}


void memcopy_vector_to_weights_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_vector_to_weights_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_weights_model(dqn->shared_hidden_layers);
    memcopy_vector_to_weights_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_weights_model(dqn->v_hidden_layers);
    memcopy_vector_to_weights_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_weights_model(dqn->v_linear_last_layer);
    memcopy_vector_to_weights_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_weights_model(dqn->a_hidden_layers);
    memcopy_vector_to_weights_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_weights_model(dqn->a_linear_last_layer);
}


void memcopy_scores_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_scores_to_vector_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    memcopy_scores_to_vector_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    memcopy_scores_to_vector_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    memcopy_scores_to_vector_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    memcopy_scores_to_vector_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_linear_last_layer);
}


void set_dueling_categorical_dqn_biases_to_zero(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    set_model_biases_to_zero(dqn->shared_hidden_layers);
    set_model_biases_to_zero(dqn->v_hidden_layers);
    set_model_biases_to_zero(dqn->v_linear_last_layer);
    set_model_biases_to_zero(dqn->a_hidden_layers);
    set_model_biases_to_zero(dqn->a_linear_last_layer);
}


void set_dueling_categorical_dqn_unused_weights_to_zero(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
        
    set_model_unused_weights_to_zero(dqn->shared_hidden_layers);
    set_model_unused_weights_to_zero(dqn->v_hidden_layers);
    set_model_unused_weights_to_zero(dqn->v_linear_last_layer);
    set_model_unused_weights_to_zero(dqn->a_hidden_layers);
    set_model_unused_weights_to_zero(dqn->a_linear_last_layer);
}


void sum_score_dueling_categorical_dqn(dueling_categorical_dqn* input1, dueling_categorical_dqn* input2, dueling_categorical_dqn* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    sum_score_model(input1->shared_hidden_layers,input2->shared_hidden_layers,output->shared_hidden_layers);
    sum_score_model(input1->v_hidden_layers,input2->v_hidden_layers,output->v_hidden_layers);
    sum_score_model(input1->v_linear_last_layer,input2->v_linear_last_layer,output->v_linear_last_layer);
    sum_score_model(input1->a_hidden_layers,input2->a_hidden_layers,output->a_hidden_layers);
    sum_score_model(input1->a_linear_last_layer,input2->a_linear_last_layer,output->a_linear_last_layer);
}


void compare_score_dueling_categorical_dqn(dueling_categorical_dqn* input1, dueling_categorical_dqn* input2, dueling_categorical_dqn* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    compare_score_model(input1->shared_hidden_layers,input2->shared_hidden_layers,output->shared_hidden_layers);
    compare_score_model(input1->v_hidden_layers,input2->v_hidden_layers,output->v_hidden_layers);
    compare_score_model(input1->v_linear_last_layer,input2->v_linear_last_layer,output->v_linear_last_layer);
    compare_score_model(input1->a_hidden_layers,input2->a_hidden_layers,output->a_hidden_layers);
    compare_score_model(input1->a_linear_last_layer,input2->a_linear_last_layer,output->a_linear_last_layer);
}


void compare_score_dueling_categorical_dqn_with_vector(dueling_categorical_dqn* input1, float* input2, dueling_categorical_dqn* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    uint64_t sum = 0;
    compare_score_model_with_vector(input1->shared_hidden_layers,input2+sum,output->shared_hidden_layers);
    sum+=get_array_size_scores_model(input1->shared_hidden_layers);
    compare_score_model_with_vector(input1->v_hidden_layers,input2+sum,output->v_hidden_layers);
    sum+=get_array_size_scores_model(input1->v_hidden_layers);
    compare_score_model_with_vector(input1->v_linear_last_layer,input2+sum,output->v_linear_last_layer);
    sum+=get_array_size_scores_model(input1->v_linear_last_layer);
    compare_score_model_with_vector(input1->a_hidden_layers,input2+sum,output->a_hidden_layers);
    sum+=get_array_size_scores_model(input1->a_hidden_layers);
    compare_score_model_with_vector(input1->a_linear_last_layer,input2+sum,output->a_linear_last_layer);
}


void dividing_score_dueling_categorical_dqn(dueling_categorical_dqn* input1, float value){
    if(input1 == NULL || value == 0)
        return;
    dividing_score_model(input1->shared_hidden_layers,value);
    dividing_score_model(input1->v_hidden_layers,value);
    dividing_score_model(input1->v_linear_last_layer,value);
    dividing_score_model(input1->a_hidden_layers,value);
    dividing_score_model(input1->a_linear_last_layer,value);
}


void reset_score_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    reset_score_model(dqn->shared_hidden_layers);
    reset_score_model(dqn->v_hidden_layers);
    reset_score_model(dqn->v_linear_last_layer);
    reset_score_model(dqn->a_hidden_layers);
    reset_score_model(dqn->a_linear_last_layer);
    
}

void reinitialize_weights_according_to_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float percentage, float goodness){
    if(dqn == NULL)
        return;
    reinitialize_weights_according_to_scores_model(dqn->shared_hidden_layers,percentage,goodness);
    reinitialize_weights_according_to_scores_model(dqn->v_hidden_layers,percentage,goodness);
    reinitialize_weights_according_to_scores_model(dqn->v_linear_last_layer,percentage,goodness);
    reinitialize_weights_according_to_scores_model(dqn->a_hidden_layers,percentage,goodness);
    reinitialize_weights_according_to_scores_model(dqn->a_linear_last_layer,percentage,goodness);
}

void reinitialize_w_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    reinitialize_w_model(dqn->shared_hidden_layers);
    reinitialize_w_model(dqn->v_hidden_layers);
    reinitialize_w_model(dqn->v_linear_last_layer);
    reinitialize_w_model(dqn->a_hidden_layers);
    reinitialize_w_model(dqn->a_linear_last_layer);
}


dueling_categorical_dqn* reset_edge_popup_d_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if (dqn == NULL)
        return NULL;
    reset_edge_popup_d_model(dqn->shared_hidden_layers);
    reset_edge_popup_d_model(dqn->v_hidden_layers);
    reset_edge_popup_d_model(dqn->v_linear_last_layer);
    reset_edge_popup_d_model(dqn->a_hidden_layers);
    reset_edge_popup_d_model(dqn->a_linear_last_layer);
    return dqn;
}

void set_low_score_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    set_low_score_model(dqn->shared_hidden_layers);
    set_low_score_model(dqn->v_hidden_layers);
    set_low_score_model(dqn->v_linear_last_layer);
    set_low_score_model(dqn->a_hidden_layers);
    set_low_score_model(dqn->a_linear_last_layer);
    
}


void make_the_dueling_categorical_dqn_only_for_ff(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    
    make_the_model_only_for_ff(dqn->shared_hidden_layers);
    make_the_model_only_for_ff(dqn->v_hidden_layers);
    make_the_model_only_for_ff(dqn->v_linear_last_layer);
    make_the_model_only_for_ff(dqn->a_hidden_layers);
    make_the_model_only_for_ff(dqn->a_linear_last_layer);
}

void compute_probability_distribution(float* input , int input_size, dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    if(input_size != dqn->input_size)
        return;
    model_tensor_input_ff(dqn->shared_hidden_layers,1,1,input_size,input);
    model_tensor_input_ff(dqn->v_hidden_layers,1,1,dqn->shared_hidden_layers->output_dimension,dqn->shared_hidden_layers->output_layer);
    model_tensor_input_ff(dqn->a_hidden_layers,1,1,dqn->shared_hidden_layers->output_dimension,dqn->shared_hidden_layers->output_layer);
    model_tensor_input_ff(dqn->v_linear_last_layer,1,1,dqn->v_hidden_layers->output_dimension,dqn->v_hidden_layers->output_layer);
    model_tensor_input_ff(dqn->a_linear_last_layer,1,1,dqn->a_hidden_layers->output_dimension,dqn->a_hidden_layers->output_layer);
    int i,j;
    for(i = 0; i < dqn->n_atoms; i++){
        for(j = 0; j < dqn->action_size; j++){
            dqn->action_mean_layer[i] -= dqn->a_linear_last_layer->output_layer[j*dqn->n_atoms+i];
        }
        dqn->action_mean_layer[i]*=(float)(1.0/((float)(dqn->action_size)));
    }
    
    for(i = 0; i < dqn->action_size; i++){
        copy_array(&dqn->a_linear_last_layer->output_layer[i*dqn->n_atoms],dqn->add_layer+i*dqn->n_atoms,dqn->n_atoms);
        sum1D(&dqn->add_layer[i*dqn->n_atoms],dqn->v_linear_last_layer->output_layer,&dqn->add_layer[i*dqn->n_atoms],dqn->n_atoms);
        sum1D(&dqn->add_layer[i*dqn->n_atoms],dqn->action_mean_layer,&dqn->add_layer[i*dqn->n_atoms],dqn->n_atoms);
        softmax(dqn->add_layer+i*dqn->n_atoms,dqn->softmax_layer+i*dqn->n_atoms,dqn->n_atoms);
    }
    return;
}

void compute_probability_distribution_opt(float* input , int input_size, dueling_categorical_dqn* dqn, dueling_categorical_dqn* dqn_wlp){
    if(dqn == NULL)
        return;
    if(input_size != dqn->input_size)
        return;
    
    model_tensor_input_ff_without_learning_parameters(dqn_wlp->shared_hidden_layers,dqn->shared_hidden_layers,1,1,input_size,input);
    model_tensor_input_ff_without_learning_parameters(dqn_wlp->v_hidden_layers,dqn->v_hidden_layers,1,1,dqn_wlp->shared_hidden_layers->output_dimension,dqn_wlp->shared_hidden_layers->output_layer);
    model_tensor_input_ff_without_learning_parameters(dqn_wlp->a_hidden_layers,dqn->a_hidden_layers,1,1,dqn_wlp->shared_hidden_layers->output_dimension,dqn_wlp->shared_hidden_layers->output_layer);
    model_tensor_input_ff_without_learning_parameters(dqn_wlp->v_linear_last_layer,dqn->v_linear_last_layer,1,1,dqn_wlp->v_hidden_layers->output_dimension,dqn_wlp->v_hidden_layers->output_layer);
    model_tensor_input_ff_without_learning_parameters(dqn_wlp->a_linear_last_layer,dqn->a_linear_last_layer,1,1,dqn_wlp->a_hidden_layers->output_dimension,dqn_wlp->a_hidden_layers->output_layer);
    int i,j;
    for(i = 0; i < dqn->n_atoms; i++){
        for(j = 0; j < dqn->action_size; j++){
            dqn_wlp->action_mean_layer[i] -= dqn_wlp->a_linear_last_layer->output_layer[j*dqn->n_atoms+i];
        }
        dqn_wlp->action_mean_layer[i]*=(float)(1.0/((float)(dqn_wlp->action_size)));
    }
    for(i = 0; i < dqn->action_size; i++){
        copy_array(&dqn_wlp->a_linear_last_layer->output_layer[i*dqn->n_atoms],dqn_wlp->add_layer+i*dqn_wlp->n_atoms,dqn_wlp->n_atoms);
        sum1D(&dqn_wlp->add_layer[i*dqn->n_atoms],dqn_wlp->v_linear_last_layer->output_layer,&dqn_wlp->add_layer[i*dqn->n_atoms],dqn_wlp->n_atoms);
        sum1D(&dqn_wlp->add_layer[i*dqn->n_atoms],dqn_wlp->action_mean_layer,&dqn_wlp->add_layer[i*dqn->n_atoms],dqn_wlp->n_atoms);
        softmax(&dqn_wlp->add_layer[i*dqn_wlp->n_atoms],&dqn_wlp->softmax_layer[i*dqn_wlp->n_atoms],dqn_wlp->n_atoms);
        
    }
    return;
}

float* bp_dueling_categorical_network(float* input, int input_size, float* error, dueling_categorical_dqn* dqn){
    int i,j,k;
    for(i = 0; i < dqn->action_size; i++){
        derivative_softmax(&dqn->derivative_softmax_layer[i*dqn->n_atoms],&dqn->softmax_layer[i*dqn->n_atoms],&error[i*dqn->n_atoms],dqn->n_atoms);
        for(j = 0; j < dqn->n_atoms; j++){
            dqn->v_linear_layer_error[j]+=dqn->derivative_softmax_layer[i*dqn->n_atoms+j];
        }
    }
    for(i = 0; i < dqn->action_size; i++){
        for(j = 0; j < dqn->action_size; j++){
            for(k = 0; k < dqn->n_atoms; k++){
                if(i == j)
                    dqn->a_linear_layer_error[i*dqn->n_atoms+k] = ((float)((float)(dqn->n_atoms-1)/((float)(dqn->n_atoms))))*dqn->derivative_softmax_layer[j*dqn->n_atoms+k];
                else
                    dqn->a_linear_layer_error[i*dqn->n_atoms+k] = ((float)((float)(-1)/((float)(dqn->n_atoms))))*dqn->derivative_softmax_layer[j*dqn->n_atoms+k];
            }
        }    
    }
    float* temp1 = model_tensor_input_bp(dqn->v_linear_last_layer,1,1,dqn->v_hidden_layers->output_dimension,dqn->v_hidden_layers->output_layer,dqn->v_linear_layer_error,dqn->v_linear_last_layer->output_dimension);
    float* temp2 = model_tensor_input_bp(dqn->a_linear_last_layer,1,1,dqn->a_hidden_layers->output_dimension,dqn->a_hidden_layers->output_layer,dqn->a_linear_layer_error,dqn->a_linear_last_layer->output_dimension);
    
    temp1 = model_tensor_input_bp(dqn->v_hidden_layers,1,1,dqn->shared_hidden_layers->output_dimension,dqn->shared_hidden_layers->output_layer,temp1,dqn->v_hidden_layers->output_dimension);
    temp2 = model_tensor_input_bp(dqn->a_hidden_layers,1,1,dqn->shared_hidden_layers->output_dimension,dqn->shared_hidden_layers->output_layer,temp2,dqn->a_hidden_layers->output_dimension);
    sum1D(temp1,temp2,temp1,dqn->shared_hidden_layers->output_dimension);
    return model_tensor_input_bp(dqn->shared_hidden_layers,1,1,input_size,input,temp1,dqn->shared_hidden_layers->output_dimension);
}

float* bp_dueling_categorical_network_opt(float* input, int input_size, float* error, dueling_categorical_dqn* dqn, dueling_categorical_dqn* dqn_wlp){
    int i,j,k;
    for(i = 0; i < dqn->action_size; i++){
        derivative_softmax(&dqn_wlp->derivative_softmax_layer[i*dqn_wlp->n_atoms],&dqn_wlp->softmax_layer[i*dqn_wlp->n_atoms],&error[i*dqn_wlp->n_atoms],dqn_wlp->n_atoms);
        for(j = 0; j < dqn_wlp->n_atoms; j++){
            dqn_wlp->v_linear_layer_error[j]+=dqn_wlp->derivative_softmax_layer[i*dqn_wlp->n_atoms+j];
        }
    }
    for(i = 0; i < dqn_wlp->action_size; i++){
        for(j = 0; j < dqn_wlp->action_size; j++){
            for(k = 0; k < dqn_wlp->n_atoms; k++){
                if(i == j)
                    dqn_wlp->a_linear_layer_error[i*dqn_wlp->n_atoms+k] = ((float)((float)(dqn_wlp->n_atoms-1)/((float)(dqn_wlp->n_atoms))))*dqn_wlp->derivative_softmax_layer[j*dqn_wlp->n_atoms+k];
                else
                    dqn_wlp->a_linear_layer_error[i*dqn_wlp->n_atoms+k] = ((float)((float)(-1)/((float)(dqn_wlp->n_atoms))))*dqn_wlp->derivative_softmax_layer[j*dqn_wlp->n_atoms+k];
            }
        }    
    }
    float* temp1 = model_tensor_input_bp_without_learning_parameters(dqn_wlp->v_linear_last_layer,dqn->v_linear_last_layer,1,1,dqn_wlp->v_hidden_layers->output_dimension,dqn_wlp->v_hidden_layers->output_layer,dqn_wlp->v_linear_layer_error,dqn_wlp->v_linear_last_layer->output_dimension);
    float* temp2 = model_tensor_input_bp_without_learning_parameters(dqn_wlp->a_linear_last_layer,dqn->a_linear_last_layer,1,1,dqn_wlp->a_hidden_layers->output_dimension,dqn_wlp->a_hidden_layers->output_layer,dqn_wlp->a_linear_layer_error,dqn_wlp->a_linear_last_layer->output_dimension);
    
    temp1 = model_tensor_input_bp_without_learning_parameters(dqn_wlp->v_hidden_layers,dqn->v_hidden_layers,1,1,dqn_wlp->shared_hidden_layers->output_dimension,dqn_wlp->shared_hidden_layers->output_layer,temp1,dqn_wlp->v_hidden_layers->output_dimension);
    temp2 = model_tensor_input_bp_without_learning_parameters(dqn_wlp->a_hidden_layers,dqn->a_hidden_layers,1,1,dqn_wlp->shared_hidden_layers->output_dimension,dqn_wlp->shared_hidden_layers->output_layer,temp2,dqn_wlp->a_hidden_layers->output_dimension);
    sum1D(temp1,temp2,temp1,dqn_wlp->shared_hidden_layers->output_dimension);
    return model_tensor_input_bp_without_learning_parameters(dqn_wlp->shared_hidden_layers,dqn->shared_hidden_layers,1,1,input_size,input,temp1,dqn_wlp->shared_hidden_layers->output_dimension);
}

float* compute_q_functions(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return NULL;
    int i,j;
    for(i = 0; i < dqn->action_size; i++){
        for(j = 0; j < dqn->n_atoms; j++){
            dqn->q_functions[i] += dqn->softmax_layer[i*dqn->n_atoms+j]*dqn->support[j];
        }
    }
    return dqn->q_functions;
}

float* get_loss_for_dueling_categorical_dqn(dueling_categorical_dqn* online_net, dueling_categorical_dqn* target_net, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1){
    compute_probability_distribution(state_t_1,state_sizes,online_net);
    compute_probability_distribution(state_t_1,state_sizes,target_net);
    compute_q_functions(target_net);
    int action_index = argmax(target_net->q_functions,target_net->action_size);
    int i;
    float* tzj = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* b = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* m = (float*)calloc(online_net->n_atoms,sizeof(float));
    for(i = 0; i < online_net->n_atoms; i++){
        if(nonterminal_s_t_1)
            tzj[i] = reward_t + lambda_value*online_net->support[i];
        else
            tzj[i] = reward_t;
    }
    clip_vector(tzj,online_net->v_min,online_net->v_max,online_net->n_atoms);
    for(i = 0; i < online_net->n_atoms; i++){
        b[i] = (tzj[i]-online_net->v_min)/online_net->z_delta;
        int l = (int)b[i];
        int u;
        if(b[i] != (int)(b[i]))
            u = l+1;
        else{
            if (l > 0)
                l--;
            u = l+1;
        }
        if(nonterminal_s_t_1){
            m[l] += target_net->softmax_layer[action_index*online_net->n_atoms+i]*((float)u-b[i]);
            m[u] += target_net->softmax_layer[action_index*online_net->n_atoms+i]*(b[i]-(float)l);
        }
        else{
            m[l] += ((float)u-b[i])/((float)(target_net->n_atoms));
            m[u] += (b[i]-(float)l)/((float)(target_net->n_atoms));
        }
    }
    for(i = 0; i < online_net->n_atoms; i++){
        if(m[i] == 0)
            online_net->error[action_t*online_net->n_atoms+i] = 0;
        else if(online_net->softmax_layer[action_t*online_net->n_atoms+i] == 0)
            online_net->error[action_t*online_net->n_atoms+i] = -99999;
        else
            online_net->error[action_t*online_net->n_atoms+i] = -m[i]/online_net->softmax_layer[action_t*online_net->n_atoms+i];
        
    }
    free(tzj);
    free(b);
    free(m);
    return online_net->error;
}

float* get_loss_for_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1){
    compute_probability_distribution_opt(state_t,state_sizes,online_net, online_net_wlp);
    compute_probability_distribution_opt(state_t_1,state_sizes,target_net,target_net_wlp);
    compute_q_functions(target_net_wlp);
    int action_index = argmax(target_net_wlp->q_functions,target_net_wlp->action_size);
    int i;
    float* tzj = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* b = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* m = (float*)calloc(online_net->n_atoms,sizeof(float));
    for(i = 0; i < online_net->n_atoms; i++){
        if(nonterminal_s_t_1){
            tzj[i] = reward_t + lambda_value*online_net_wlp->support[i];
        }
        else{
            tzj[i] = reward_t;
        }
    }
    clip_vector(tzj,online_net_wlp->v_min,online_net_wlp->v_max,online_net_wlp->n_atoms);
    for(i = 0; i < online_net->n_atoms; i++){
        b[i] = (tzj[i]-online_net->v_min)/online_net->z_delta;
        int l = (int)b[i];
        int u;
        if(b[i] != (int)(b[i]))
            u = l+1;
        else{
            if (l > 0)
                l--;
            u = l+1;
        }
        if(nonterminal_s_t_1){
            m[l] += target_net_wlp->softmax_layer[action_index*online_net->n_atoms+i]*((float)u-b[i]);
            m[u] += target_net_wlp->softmax_layer[action_index*online_net->n_atoms+i]*(b[i]-(float)l);
        }
        else{
            m[l] += ((float)u-b[i])/((float)(target_net_wlp->n_atoms));
            m[u] += (b[i]-(float)l)/((float)(target_net_wlp->n_atoms));
        }
    }
    for(i = 0; i < online_net->n_atoms; i++){
        if(m[i] == 0)
            online_net_wlp->error[action_t*online_net->n_atoms+i] = 0;
        else if(online_net_wlp->softmax_layer[action_t*online_net->n_atoms+i] == 0)
            online_net_wlp->error[action_t*online_net->n_atoms+i] = -99999;
        else
            online_net_wlp->error[action_t*online_net->n_atoms+i] = -m[i]/online_net_wlp->softmax_layer[action_t*online_net->n_atoms+i];
    }
    free(tzj);
    free(b);
    free(m);
    return online_net_wlp->error;
}

float* get_loss_for_dueling_categorical_dqn_opt_with_error(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1, float* new_error, float weight_error){
    compute_probability_distribution_opt(state_t,state_sizes,online_net, online_net_wlp);
    compute_probability_distribution_opt(state_t_1,state_sizes,target_net,target_net_wlp);
    compute_q_functions(target_net_wlp);
    int action_index = argmax(target_net_wlp->q_functions,target_net_wlp->action_size);
    int i;
    float* tzj = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* b = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* m = (float*)calloc(online_net->n_atoms,sizeof(float));
    for(i = 0; i < online_net->n_atoms; i++){
        if(nonterminal_s_t_1){
            tzj[i] = reward_t + lambda_value*online_net_wlp->support[i];
        }
        else{
            tzj[i] = reward_t;
        }
    }
    clip_vector(tzj,online_net_wlp->v_min,online_net_wlp->v_max,online_net_wlp->n_atoms);
    for(i = 0; i < online_net->n_atoms; i++){
        b[i] = (tzj[i]-online_net->v_min)/online_net->z_delta;
        int l = (int)b[i];
        int u;
        if(b[i] != (int)(b[i]))
            u = l+1;
        else{
            if (l > 0)
                l--;
            u = l+1;
        }
        if(nonterminal_s_t_1){
            m[l] += target_net_wlp->softmax_layer[action_index*online_net->n_atoms+i]*((float)u-b[i]);
            m[u] += target_net_wlp->softmax_layer[action_index*online_net->n_atoms+i]*(b[i]-(float)l);
        }
        else{
            m[l] += ((float)u-b[i])/((float)(target_net_wlp->n_atoms));
            m[u] += (b[i]-(float)l)/((float)(target_net_wlp->n_atoms));
        }
    }
    float temp_err = 0;
    for(i = 0; i < online_net->n_atoms; i++){
        if(m[i] == 0){
            online_net_wlp->error[action_t*online_net->n_atoms+i] = 0;
            temp_err += 0;
        }
        else if(online_net_wlp->softmax_layer[action_t*online_net->n_atoms+i] == 0){
            online_net_wlp->error[action_t*online_net->n_atoms+i] = -99999;
            temp_err += -99999;
        }
        else{
            online_net_wlp->error[action_t*online_net->n_atoms+i] = -weight_error*m[i]/online_net_wlp->softmax_layer[action_t*online_net->n_atoms+i];
            temp_err += log(m[i]/online_net_wlp->softmax_layer[action_t*online_net->n_atoms+i])*m[i]*weight_error;
        }    
    }
    if(!bool_is_real(temp_err))
        temp_err = 99999;
    if(temp_err < 0)
        new_error[0] = -temp_err;
    else
        new_error[0] = temp_err;    
    free(tzj);
    free(b);
    free(m);
    return online_net_wlp->error;
}

// returns the error
float compute_kl_dueling_categorical_dqn(dueling_categorical_dqn* online_net, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    // feed forward
    compute_probability_distribution(state_t,size,online_net);
    // q functions
    compute_q_functions(online_net);
    int i, j,index = 0;
    // some initializations
    float max_q1 = online_net->q_functions[0];
    float max_q2 = q_functions[0];
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* softmax_derivative_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* other_distr = (float*)calloc(online_net->action_size,sizeof(float));
    float* error = (float*)calloc(online_net->action_size,sizeof(float));
    float ret = 0;// the returning error
    
    // softmax of the q functions to get the policy
    softmax(online_net->q_functions,softmax_arr,online_net->action_size);
    
    // log(e^x/sum(e^x)) ~ x - max(x)
    for(i = 0; i < online_net->action_size; i++){
        // getting the maximum of the q functions of the network and saving its index
        if(max_q1 < online_net->q_functions[i]){
            max_q1 = online_net->q_functions[i];
            index = i;
        }
        // getting the maximum of the q functions we want to diverge from
        max_q2 = max_float(max_q2,q_functions[i]);
    }
    
    // kl = sum (e^x/sum(e^x))*(x-max(x)-y+max(y))
    // -kl = sum (e^x/sum(e^x))*(-x+max(x)+y-max(y))
    // derivative: d/dx((e^x/sum(e^x)))*(-x+max(x)+y-max(y)) + ((e^x/sum(e^x)))*d/dx(-x+max(x)+y-max(y))
    for(i = 0; i < online_net->action_size; i++){
        float temp = (q_functions[i]-max_q2-online_net->q_functions[i]+max_q1);// second part of the first derivative
        other_distr[i] = alpha*weight*temp;// we must rescale everything with alpha and weight
        // first and second part of the second derivative
        if(i != index){
            error[i] = -alpha*weight*softmax_arr[i];
            error[index]+=alpha*weight*softmax_arr[i];
        }
        ret+=temp*softmax_arr[i];// final error (no derivative)
    }
    // first derivative multiplied by other_distr that is the second part
    derivative_softmax(softmax_derivative_arr,softmax_arr,other_distr,online_net->action_size);
    // sum of the 2 derivatives
    sum1D(softmax_derivative_arr,error,error,online_net->action_size);
    // clipping the derivatives as the paper says (no supplementary material found, gg nimps and other publishers, they keep saying hyperparams are in the supplementary
    // material, but after looking for this F@#%$ supplementary material i could not find anything so i need to come with the clipping right value, and this also for the distance threshold
    // used rescale alpha that is the most important part)
    clip_vector(error,-clip,clip,online_net->action_size);
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net->error[i*online_net->n_atoms+j]+=error[i]*online_net->support[j];
        }
    }
    free(softmax_arr);
    free(softmax_derivative_arr);
    free(other_distr);
    free(error);
    return ret;
}

// returns the error
float compute_kl_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    if(clip<0)
        clip = -clip;
    // feed forward
    compute_probability_distribution_opt(state_t,size,online_net,online_net_wlp);
    // q functions
    compute_q_functions(online_net_wlp);
    int i, j,index = 0;
    // some initializations
    float max_q1 = online_net_wlp->q_functions[0];
    float max_q2 = q_functions[0];
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* softmax_derivative_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* other_distr = (float*)calloc(online_net->action_size,sizeof(float));
    float* error = (float*)calloc(online_net->action_size,sizeof(float));
    float ret = 0;// the returning error
    
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    
    // log(e^x/sum(e^x)) ~ x - max(x)
    for(i = 0; i < online_net->action_size; i++){
        // getting the maximum of the q functions of the network and saving its index
        if(max_q1 < online_net_wlp->q_functions[i]){
            max_q1 = online_net_wlp->q_functions[i];
            index = i;
        }
        // getting the maximum of the q functions we want to diverge from
        max_q2 = max_float(max_q2,q_functions[i]);
    }
    
    // kl = sum (e^x/sum(e^x))*(x-max(x)-y+max(y))
    // derivative: d/dx((e^x/sum(e^x)))*(x-max(x)-y+max(y)) + ((e^x/sum(e^x)))*d/dx(x-max(x)-y+max(y))
    // however, we must consider -kl divergence because kl divergence tells us how similar 2 distribution are
    // and if we perform gradient descent then we are trying to minimize this error and we will end up getting
    // a more similar distribution to the one we targeted. Instead we should use gradient ascent that can be performed
    // using the gradient descent and inverting the sign, so -kl divergence.
    for(i = 0; i < online_net->action_size; i++){
        float temp = (q_functions[i]-max_q2-online_net_wlp->q_functions[i]+max_q1);// second part of the first derivative, inverted sign
        other_distr[i] = alpha*weight*temp;// we must rescale everything with alpha and weight
        // first and second part of the second derivative
        if(i != index){
            error[i] = -alpha*weight*softmax_arr[i];// inverted sign
            error[index]+=alpha*weight*softmax_arr[i];// inverted sign
        }
        ret+=temp*softmax_arr[i];// final error (no derivative no inversion, we need it to then perform annhilation of alpha or not)
    }
    // first derivative multiplied by other_distr that is the second part
    derivative_softmax(softmax_derivative_arr,softmax_arr,other_distr,online_net->action_size);
    // sum of the 2 derivatives
    sum1D(softmax_derivative_arr,error,error,online_net->action_size);
    // clipping the derivatives as the paper says (no supplementary material found, gg nimps and other publishers, they keep saying hyperparams are in the supplementary
    // material, but after looking for this F@#%$ supplementary material i could not find anything so i need to come with the clipping right value, and this also for the distance threshold
    // used to rescale alpha that is the most important part)
    clip_vector(error,-clip,clip,online_net->action_size);
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer of the network
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=error[i]*online_net_wlp->support[j];
        }
    }
    free(softmax_arr);
    free(softmax_derivative_arr);
    free(other_distr);
    free(error);
    return ret;
}

void set_dueling_categorical_dqn_training_edge_popup(dueling_categorical_dqn* dqn, float k_percentage){
    if(dqn == NULL)
        return;
    set_model_training_edge_popup(dqn->shared_hidden_layers,k_percentage);
    set_model_training_edge_popup(dqn->v_hidden_layers,k_percentage);
    set_model_training_edge_popup(dqn->v_linear_last_layer,k_percentage);
    set_model_training_edge_popup(dqn->a_hidden_layers,k_percentage);
    set_model_training_edge_popup(dqn->a_linear_last_layer,k_percentage);
    return;
}

void set_dueling_categorical_dqn_training_gd(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    set_model_training_gd(dqn->shared_hidden_layers);
    set_model_training_gd(dqn->v_hidden_layers);
    set_model_training_gd(dqn->v_linear_last_layer);
    set_model_training_gd(dqn->a_hidden_layers);
    set_model_training_gd(dqn->a_linear_last_layer);
    return;
    
}

void set_dueling_categorical_dqn_beta(dueling_categorical_dqn* dqn, float b1, float b2){
    if(dqn == NULL)
        return;
    set_model_beta(dqn->shared_hidden_layers,b1,b2);
    set_model_beta(dqn->v_hidden_layers,b1,b2);
    set_model_beta(dqn->v_linear_last_layer,b1,b2);
    set_model_beta(dqn->a_hidden_layers,b1,b2);
    set_model_beta(dqn->a_linear_last_layer,b1,b2);
    return;
}

void set_dueling_categorical_dqn_beta_adamod(dueling_categorical_dqn*  dqn, float b){
    if(dqn == NULL)
        return;
    set_model_beta_adamod(dqn->shared_hidden_layers,b);
    set_model_beta_adamod(dqn->v_hidden_layers,b);
    set_model_beta_adamod(dqn->v_linear_last_layer,b);
    set_model_beta_adamod(dqn->a_hidden_layers,b);
    set_model_beta_adamod(dqn->a_linear_last_layer,b);
    return;
}

float get_beta1_from_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    return get_beta1_from_model(dqn->shared_hidden_layers);
}

float get_beta2_from_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    return get_beta2_from_model(dqn->shared_hidden_layers);
}

float get_beta3_from_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    return get_beta3_from_model(dqn->shared_hidden_layers);
}

void set_ith_layer_training_mode_dueling_categorical_dqn_shared(dueling_categorical_dqn* dqn, int ith, int training_flag){
    if(dqn == NULL)
        return;
    set_ith_layer_training_mode_model(dqn->shared_hidden_layers,ith,training_flag);
    return;
}

void set_ith_layer_training_mode_dueling_categorical_dqn_v_hid(dueling_categorical_dqn* dqn, int ith, int training_flag){
    if(dqn == NULL)
        return;
    set_ith_layer_training_mode_model(dqn->v_hidden_layers,ith,training_flag);
    return;
}

void set_ith_layer_training_mode_dueling_categorical_dqn_v_lin(dueling_categorical_dqn* dqn, int ith, int training_flag){
    if(dqn == NULL)
        return;
    set_ith_layer_training_mode_model(dqn->v_linear_last_layer,ith,training_flag);
    return;
}

void set_ith_layer_training_mode_dueling_categorical_dqn_a_hid(dueling_categorical_dqn* dqn, int ith, int training_flag){
    if(dqn == NULL)
        return;
    set_ith_layer_training_mode_model(dqn->a_hidden_layers,ith,training_flag);
    return;
}
void set_ith_layer_training_mode_dueling_categorical_dqn_a_lin(dueling_categorical_dqn* dqn, int ith, int training_flag){
    if(dqn == NULL)
        return;
    set_ith_layer_training_mode_model(dqn->a_linear_last_layer,ith,training_flag);
    return;
}

void set_k_percentage_of_ith_layer_dueling_categorical_dqn_shared(dueling_categorical_dqn* dqn, int ith, float k_percentage){
    if(dqn == NULL)
        return;
    set_k_percentage_of_ith_layer_model(dqn->shared_hidden_layers,ith,k_percentage);
}

void set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_hid(dueling_categorical_dqn* dqn, int ith, float k_percentage){
    if(dqn == NULL)
        return;
    set_k_percentage_of_ith_layer_model(dqn->v_hidden_layers,ith,k_percentage);
}

void set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_lin(dueling_categorical_dqn* dqn, int ith, float k_percentage){
    if(dqn == NULL)
        return;
    set_k_percentage_of_ith_layer_model(dqn->v_linear_last_layer,ith,k_percentage);
}

void set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_hid(dueling_categorical_dqn* dqn, int ith, float k_percentage){
    if(dqn == NULL)
        return;
    set_k_percentage_of_ith_layer_model(dqn->a_hidden_layers,ith,k_percentage);
}

void set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_lin(dueling_categorical_dqn* dqn, int ith, float k_percentage){
    if(dqn == NULL)
        return;
    set_k_percentage_of_ith_layer_model(dqn->a_linear_last_layer,ith,k_percentage);
}

int get_input_layer_size_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return 0;
    return get_input_layer_size(dqn->shared_hidden_layers);
}
