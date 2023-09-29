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
    dqn->is_qr = 0;
    dqn->is_iqn= 0;
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
    
    dqn->k = 0;
    dqn->n = 0;
    dqn->quantile_value = 0;
    dqn->quantile_array = NULL;
    dqn->single_ff_network = NULL;
    dqn->single_ff_networks = NULL;
    dqn->v_hidden_layers_q = NULL;
    dqn->v_linear_last_layer_q = NULL;
    dqn->a_hidden_layers_q = NULL;
    dqn->a_linear_last_layer_q = NULL;
    return dqn;
}

dueling_categorical_dqn* dueling_categorical_dqn_init_without_arrays(int input_size, int action_size, int n_atoms, float v_min, float v_max, model* shared_hidden_layers, model* v_hidden_layers, model* a_hidden_layers, model* v_linear_last_layer, model* a_linear_last_layer){
    if(shared_hidden_layers == NULL || v_hidden_layers == NULL || a_hidden_layers == NULL || v_linear_last_layer == NULL || a_linear_last_layer == NULL){
        return NULL;
    }
    
    if(shared_hidden_layers->layers <= 0 || v_hidden_layers->layers <= 0 || a_hidden_layers->layers <= 0 || v_linear_last_layer->layers != 1 || a_linear_last_layer->layers != 1){
        return NULL;
    }
    
    if(input_size <= 0){
        return NULL;
    }
    
    if(n_atoms <= 1){
        return NULL;
    }
    
    if(action_size <= 0){
        return NULL;
    }
    
    if(v_min > v_max){
        return NULL;
    }
    
    if(get_input_layer_size(shared_hidden_layers) != input_size){
        return NULL;
    }
    
    if(action_size*n_atoms != a_linear_last_layer->output_dimension){
        return NULL;
    }
    
    if(v_linear_last_layer->output_dimension != n_atoms){
        return NULL;
    }
    
    if(v_linear_last_layer->n_cl != 0 || v_linear_last_layer->n_rl != 0 || v_linear_last_layer->n_fcl != 1){
        return NULL;
    }
    
    if(a_linear_last_layer->n_cl != 0 || a_linear_last_layer->n_rl != 0 || a_linear_last_layer->n_fcl != 1){
        return NULL;
    }
    
    if(v_linear_last_layer->fcls[0]->activation_flag || v_linear_last_layer->fcls[0]->normalization_flag){
        return NULL;
    }
    if(a_linear_last_layer->fcls[0]->activation_flag || a_linear_last_layer->fcls[0]->normalization_flag){
        return NULL;
    }
    
    if(shared_hidden_layers->output_dimension != get_input_layer_size(v_hidden_layers)){
        return NULL;
    }
    if(shared_hidden_layers->output_dimension != get_input_layer_size(a_hidden_layers)){
        return NULL;
    }
    if(v_hidden_layers->output_dimension != get_input_layer_size(v_linear_last_layer)){
        return NULL;
    }
    if(a_hidden_layers->output_dimension != get_input_layer_size(a_linear_last_layer)){
        return NULL;
    }
    if(shared_hidden_layers->beta1_adam != v_hidden_layers->beta1_adam || shared_hidden_layers->beta1_adam != v_linear_last_layer->beta1_adam || shared_hidden_layers->beta1_adam != a_hidden_layers->beta1_adam || shared_hidden_layers->beta1_adam != a_linear_last_layer->beta1_adam){
        return NULL;
    }
    
    if(shared_hidden_layers->beta2_adam != v_hidden_layers->beta2_adam || shared_hidden_layers->beta2_adam != v_linear_last_layer->beta2_adam || shared_hidden_layers->beta2_adam != a_hidden_layers->beta2_adam || shared_hidden_layers->beta2_adam != a_linear_last_layer->beta2_adam){
        return NULL;
    }
    
    if(shared_hidden_layers->beta3_adamod != v_hidden_layers->beta3_adamod || shared_hidden_layers->beta3_adamod != v_linear_last_layer->beta3_adamod || shared_hidden_layers->beta3_adamod != a_hidden_layers->beta3_adamod || shared_hidden_layers->beta3_adamod != a_linear_last_layer->beta3_adamod){
        return NULL;
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
    dqn->is_qr = 0;
    dqn->z_delta = (v_max-v_min)/((float)(n_atoms-1));
    dqn->action_size = action_size;
    
    dqn->is_iqn = 0;
    dqn->k = 0;
    dqn->n = 0;
    dqn->quantile_value = 0;
    dqn->quantile_array = NULL;
    dqn->single_ff_network = NULL;
    dqn->single_ff_networks = NULL;
    dqn->v_hidden_layers_q = NULL;
    dqn->v_linear_last_layer_q = NULL;
    dqn->a_hidden_layers_q = NULL;
    dqn->a_linear_last_layer_q = NULL;
    
    
    return dqn;
}

void save_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int n){
    if(dqn == NULL)
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
    convert_data(&dqn->is_qr,sizeof(int),1);
    i = fwrite(&dqn->is_qr,sizeof(int),1,fw);
    convert_data(&dqn->is_qr,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    convert_data(&dqn->input_size,sizeof(int),1);
    i = fwrite(&dqn->input_size,sizeof(int),1,fw);
    convert_data(&dqn->input_size,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    convert_data(&dqn->action_size,sizeof(int),1);
    i = fwrite(&dqn->action_size,sizeof(int),1,fw);
    convert_data(&dqn->action_size,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->n_atoms,sizeof(int),1);
    i = fwrite(&dqn->n_atoms,sizeof(int),1,fw);
    convert_data(&dqn->n_atoms,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->v_min,sizeof(float),1);
    i = fwrite(&dqn->v_min,sizeof(float),1,fw);
    convert_data(&dqn->v_min,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->v_max,sizeof(float),1);
    i = fwrite(&dqn->v_max,sizeof(float),1,fw);
    convert_data(&dqn->v_max,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    // new data for implicit quantile network
    /*
    convert_data(&dqn->is_iqn,sizeof(int),1);
    i = fwrite(&dqn->is_iqn,sizeof(int),1,fw);
    convert_data(&dqn->is_iqn,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->k,sizeof(int),1);
    i = fwrite(&dqn->k,sizeof(int),1,fw);
    convert_data(&dqn->k,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->n,sizeof(int),1);
    i = fwrite(&dqn->n,sizeof(int),1,fw);
    convert_data(&dqn->n,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->quantile_value,sizeof(int),1);
    i = fwrite(&dqn->quantile_value,sizeof(int),1,fw);
    convert_data(&dqn->quantile_value,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    */
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
    
    // new data for implicit quantile network
    /*
    if(dqn->is_iqn)
        save_model(dqn->single_ff_network,n);
    */
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
    
    int input_size, action_size, n_atoms, is_qr, is_iqn, n, k, quantile_value;
    float v_min,v_max;
    
    i = fread(&is_qr,sizeof(int),1,fr);
    convert_data(&is_qr,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&input_size,sizeof(int),1,fr);
    convert_data(&input_size,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&action_size,sizeof(int),1,fr);
    convert_data(&action_size,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    
    i = fread(&n_atoms,sizeof(int),1,fr);
    convert_data(&n_atoms,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&v_min,sizeof(float),1,fr);
    convert_data(&v_min,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&v_max,sizeof(float),1,fr);
    convert_data(&v_max,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    // new data for implicit quantile network
    /*
    i = fread(&is_iqn,sizeof(float),1,fr);
    convert_data(&is_iqn,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&k,sizeof(float),1,fr);
    convert_data(&k,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&n,sizeof(float),1,fr);
    convert_data(&n,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    i = fread(&quantile_value,sizeof(float),1,fr);
    convert_data(&quantile_value,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the model\n");
        exit(1);
    }
    */
    
    
    model* shared_hidden_layers = load_model_with_file_already_opened(fr);
    model* v_hidden_layers = load_model_with_file_already_opened(fr);
    model* v_linear_last_layer = load_model_with_file_already_opened(fr);
    model* a_hidden_layers = load_model_with_file_already_opened(fr);
    model* a_linear_last_layer = load_model_with_file_already_opened(fr);
    model* single = NULL;
    
    // new data for implicit quantile network
    /*
    if(is_iqn)
        single = load_model_with_file_already_opened(fr);
    */
    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file);
        exit(1);
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init(input_size,action_size,n_atoms,v_min,v_max,shared_hidden_layers,v_hidden_layers,a_hidden_layers,v_linear_last_layer,a_linear_last_layer);
    dqn->is_qr = is_qr;
    // new data for implicit quantile network
    /*
    if(is_iqn){
        set_iqn(dqn,n,k,quantile_value,single);
    }
    * */
    return dqn;
    
}

// new data for implicit quantile network
/*
void set_iqn(dueling_categorical_dqn* dqn, int n, int k, int quantile_value, model* single){
    dqn->is_qr = 0;
    dqn->is_iqn = 1;
    dqn->n = n;
    dqn->k = k;
    dqn->quantile_value = quantile_value;
    dqn->quantile_array = (float*)calloc(dqn->quantile_value,sizeof(float));
    int max = n;
    if(k > max)
        max = k;
    if(single != NULL)
        dqn->single_ff_network = single;
    else{
        fcl** fcls = (fcl**)malloc(sizeof(fcl*));
        fcls[0] = fully_connected(quantile_value,dqn->shared_hidden_layers->output_dimension,0,0,5,0,0,0,GRADIENT_DESCENT,FULLY_FEED_FORWARD,0);
        dqn->single_ff_network = network(1,0,0,1,NULL,NULL,fcls);
    }
    dqn->single_ff_networks = (model**)malloc(sizeof(model*)*max);
    dqn->v_hidden_layers_q = (model**)malloc(sizeof(model*)*max);
    dqn->v_linear_last_layer_q = (model**)malloc(sizeof(model*)*max);
    dqn->a_hidden_layers_q = (model**)malloc(sizeof(model*)*max);
    dqn->a_linear_last_layer_q = (model**)malloc(sizeof(model*)*max);
    int i;
    for(i = 0; i < max; i++){
        dqn->single_ff_networks[i] = copy_model_without_learning_parameters(dqn->single_ff_network);
        dqn->v_hidden_layers_q[i] = copy_model_without_learning_parameters(dqn->v_hidden_layers);
        dqn->v_linear_last_layer_q[i] = copy_model_without_learning_parameters(dqn->v_linear_last_layer);
        dqn->a_hidden_layers_q[i] = copy_model_without_learning_parameters(dqn->a_hidden_layers);
        dqn->a_linear_last_layer_q[i] = copy_model_without_learning_parameters(dqn->a_linear_last_layer);
    }    
}
*/

void save_dueling_categorical_dqn_given_directory(dueling_categorical_dqn* dqn, int n, char* directory){
    if(dqn == NULL)
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
    
    convert_data(&dqn->is_qr,sizeof(int),1);
    i = fwrite(&dqn->is_qr,sizeof(int),1,fw);
    convert_data(&dqn->is_qr,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    
    convert_data(&dqn->input_size,sizeof(int),1);
    i = fwrite(&dqn->input_size,sizeof(int),1,fw);
    convert_data(&dqn->input_size,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->action_size,sizeof(int),1);
    i = fwrite(&dqn->action_size,sizeof(int),1,fw);
    convert_data(&dqn->action_size,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->n_atoms,sizeof(int),1);
    i = fwrite(&dqn->n_atoms,sizeof(int),1,fw);
    convert_data(&dqn->n_atoms,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->v_min,sizeof(float),1);
    i = fwrite(&dqn->v_min,sizeof(float),1,fw);
    convert_data(&dqn->v_min,sizeof(float),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the dqn\n");
        exit(1);
    }
    convert_data(&dqn->v_max,sizeof(float),1);
    i = fwrite(&dqn->v_max,sizeof(float),1,fw);
    convert_data(&dqn->v_max,sizeof(float),1);
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

void set_is_qr(dueling_categorical_dqn* dqn, int is_qr){
    if(dqn == NULL)
        return;
    dqn->is_qr = is_qr;
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

void free_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    free_scores_model(dqn->shared_hidden_layers);
    free_scores_model(dqn->v_hidden_layers);
    free_scores_model(dqn->a_hidden_layers);
    free_scores_model(dqn->v_linear_last_layer);
    free_scores_model(dqn->a_linear_last_layer);
}

void free_indices_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    free_indices_model(dqn->shared_hidden_layers);
    free_indices_model(dqn->v_hidden_layers);
    free_indices_model(dqn->a_hidden_layers);
    free_indices_model(dqn->v_linear_last_layer);
    free_indices_model(dqn->a_linear_last_layer);
}

void set_null_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    set_null_scores_model(dqn->shared_hidden_layers);
    set_null_scores_model(dqn->v_hidden_layers);
    set_null_scores_model(dqn->a_hidden_layers);
    set_null_scores_model(dqn->v_linear_last_layer);
    set_null_scores_model(dqn->a_linear_last_layer);
}

void set_null_indices_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    set_null_indices_model(dqn->shared_hidden_layers);
    set_null_indices_model(dqn->v_hidden_layers);
    set_null_indices_model(dqn->a_hidden_layers);
    set_null_indices_model(dqn->v_linear_last_layer);
    set_null_indices_model(dqn->a_linear_last_layer);
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
void reset_dueling_categorical_dqn_only_for_ff(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    reset_model_only_for_ff(dqn->shared_hidden_layers);
    reset_model_only_for_ff(dqn->v_hidden_layers);
    reset_model_only_for_ff(dqn->a_hidden_layers);
    reset_model_only_for_ff(dqn->v_linear_last_layer);
    reset_model_only_for_ff(dqn->a_linear_last_layer);
    set_vector_with_value(0,dqn->action_mean_layer,dqn->n_atoms);
    set_vector_with_value(0,dqn->v_linear_layer_error,dqn->n_atoms);
    set_vector_with_value(0,dqn->q_functions,dqn->action_size);
    set_vector_with_value(0,dqn->add_layer,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->a_linear_layer_error,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->error,dqn->action_size*dqn->n_atoms);
    set_vector_with_value(0,dqn->softmax_layer,dqn->action_size*dqn->n_atoms);
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
    dueling_categorical_dqn* d = dueling_categorical_dqn_init(dqn->input_size,dqn->action_size,dqn->n_atoms,dqn->v_min,dqn->v_max,shared_hidden_layers,v_hidden_layers,a_hidden_layers,v_linear_last_layer,a_linear_last_layer);
    d->is_qr = dqn->is_qr;
    return d;
}

dueling_categorical_dqn* copy_dueling_categorical_dqn_without_learning_parameters(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return NULL;
    model* shared_hidden_layers = copy_model_without_learning_parameters(dqn->shared_hidden_layers);
    model* v_hidden_layers = copy_model_without_learning_parameters(dqn->v_hidden_layers);
    model* a_hidden_layers = copy_model_without_learning_parameters(dqn->a_hidden_layers);
    model* v_linear_last_layer = copy_model_without_learning_parameters(dqn->v_linear_last_layer);
    model* a_linear_last_layer = copy_model_without_learning_parameters(dqn->a_linear_last_layer);
    dueling_categorical_dqn* d = dueling_categorical_dqn_init(dqn->input_size,dqn->action_size,dqn->n_atoms,dqn->v_min,dqn->v_max,shared_hidden_layers,v_hidden_layers,a_hidden_layers,v_linear_last_layer,a_linear_last_layer);
    d->is_qr = dqn->is_qr;
    return d;
}


void paste_dueling_categorical_dqn(dueling_categorical_dqn* dqn, dueling_categorical_dqn* copy){
    if(dqn == NULL || copy == NULL)
        return;
    paste_model(dqn->shared_hidden_layers,copy->shared_hidden_layers);
    paste_model(dqn->v_hidden_layers,copy->v_hidden_layers);
    paste_model(dqn->a_hidden_layers,copy->a_hidden_layers);
    paste_model(dqn->v_linear_last_layer,copy->v_linear_last_layer);
    paste_model(dqn->a_linear_last_layer,copy->a_linear_last_layer);
    copy->is_qr = dqn->is_qr;
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
    copy->is_qr = dqn->is_qr;
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
    copy->is_qr = dqn->is_qr;
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


uint64_t get_array_size_scores_index_dueling_categorical_dqn(dueling_categorical_dqn* dqn, uint64_t s){
    if(dqn == NULL)
        return 0;
    uint64_t sum = 0,i,j;

    for(i = 0; i < dqn->shared_hidden_layers->n_fcl; i++){
        sum+=get_array_size_scores_fcl(dqn->shared_hidden_layers->fcls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->shared_hidden_layers->n_cl; i++){
        sum+=get_array_size_scores_cl(dqn->shared_hidden_layers->cls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->shared_hidden_layers->n_rl; i++){
        for(j = 0; j < dqn->shared_hidden_layers->rls[i]->n_cl; j++){
            sum+=(uint64_t)get_array_size_scores_cl(dqn->shared_hidden_layers->rls[i]->cls[j]);
            if (sum > s)
                return sum;
        }
    }
    
    for(i = 0; i < dqn->v_hidden_layers->n_fcl; i++){
        sum+=get_array_size_scores_fcl(dqn->v_hidden_layers->fcls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->v_hidden_layers->n_cl; i++){
        sum+=get_array_size_scores_cl(dqn->v_hidden_layers->cls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->v_hidden_layers->n_rl; i++){
        for(j = 0; j < dqn->v_hidden_layers->rls[i]->n_cl; j++){
            sum+=(uint64_t)get_array_size_scores_cl(dqn->v_hidden_layers->rls[i]->cls[j]);
            if (sum > s)
                return sum;
        }
    }
    
    for(i = 0; i < dqn->v_linear_last_layer->n_fcl; i++){
        sum+=get_array_size_scores_fcl(dqn->v_linear_last_layer->fcls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->v_linear_last_layer->n_cl; i++){
        sum+=get_array_size_scores_cl(dqn->v_linear_last_layer->cls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->v_linear_last_layer->n_rl; i++){
        for(j = 0; j < dqn->v_linear_last_layer->rls[i]->n_cl; j++){
            sum+=(uint64_t)get_array_size_scores_cl(dqn->v_linear_last_layer->rls[i]->cls[j]);
            if (sum > s)
                return sum;
        }
    }
    
    for(i = 0; i < dqn->a_hidden_layers->n_fcl; i++){
        sum+=get_array_size_scores_fcl(dqn->a_hidden_layers->fcls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->a_hidden_layers->n_cl; i++){
        sum+=get_array_size_scores_cl(dqn->a_hidden_layers->cls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->a_hidden_layers->n_rl; i++){
        for(j = 0; j < dqn->a_hidden_layers->rls[i]->n_cl; j++){
            sum+=(uint64_t)get_array_size_scores_cl(dqn->a_hidden_layers->rls[i]->cls[j]);
            if (sum > s)
                return sum;
        }
    }
    
    for(i = 0; i < dqn->a_linear_last_layer->n_fcl; i++){
        sum+=get_array_size_scores_fcl(dqn->a_linear_last_layer->fcls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->a_linear_last_layer->n_cl; i++){
        sum+=get_array_size_scores_cl(dqn->a_linear_last_layer->cls[i]);
        if (sum > s)
            return sum;
    }
    
    for(i = 0; i < dqn->a_linear_last_layer->n_rl; i++){
        for(j = 0; j < dqn->a_linear_last_layer->rls[i]->n_cl; j++){
            sum+=(uint64_t)get_array_size_scores_cl(dqn->a_linear_last_layer->rls[i]->cls[j]);
            if (sum > s)
                return sum;
        }
    }
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

void memcopy_vector_to_indices_dueling_categorical_dqn2(dueling_categorical_dqn* dqn, int* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_vector_to_indices_model2(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    
    memcopy_vector_to_indices_model2(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    
    memcopy_vector_to_indices_model2(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    
    memcopy_vector_to_indices_model2(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    
    memcopy_vector_to_indices_model2(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_linear_last_layer);
}

void assign_vector_to_scores_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;

    assign_vector_to_scores_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    assign_vector_to_scores_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    assign_vector_to_scores_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    assign_vector_to_scores_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    assign_vector_to_scores_model(dqn->a_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_linear_last_layer);
}

void memcopy_vector_to_indices_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_vector_to_indices_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    memcopy_vector_to_indices_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    memcopy_vector_to_indices_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    memcopy_vector_to_indices_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    memcopy_vector_to_indices_model(dqn->a_linear_last_layer,vector+sum);
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

void memcopy_indices_to_vector_dueling_categorical_dqn(dueling_categorical_dqn* dqn, int* vector){
    if(dqn == NULL || vector == NULL)
        return;
    uint64_t sum = 0;
    memcopy_indices_to_vector_model(dqn->shared_hidden_layers,vector);
    sum+=get_array_size_scores_model(dqn->shared_hidden_layers);
    memcopy_indices_to_vector_model(dqn->v_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_hidden_layers);
    memcopy_indices_to_vector_model(dqn->v_linear_last_layer,vector+sum);
    sum+=get_array_size_scores_model(dqn->v_linear_last_layer);
    memcopy_indices_to_vector_model(dqn->a_hidden_layers,vector+sum);
    sum+=get_array_size_scores_model(dqn->a_hidden_layers);
    memcopy_indices_to_vector_model(dqn->a_linear_last_layer,vector+sum);
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

void reinitialize_weights_according_to_scores_dueling_categorical_dqn_only_percentage(dueling_categorical_dqn* dqn, float percentage){
    if(dqn == NULL)
        return;
    reinitialize_weights_according_to_scores_model_only_percentage(dqn->shared_hidden_layers,percentage);
    reinitialize_weights_according_to_scores_model_only_percentage(dqn->v_hidden_layers,percentage);
    reinitialize_weights_according_to_scores_model_only_percentage(dqn->v_linear_last_layer,percentage);
    reinitialize_weights_according_to_scores_model_only_percentage(dqn->a_hidden_layers,percentage);
    reinitialize_weights_according_to_scores_model_only_percentage(dqn->a_linear_last_layer,percentage);
}

void reinitialize_weights_according_to_scores_and_inner_info_dueling_categorical_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return;
    reinitialize_weights_according_to_scores_and_inner_info_model(dqn->shared_hidden_layers);
    reinitialize_weights_according_to_scores_and_inner_info_model(dqn->v_hidden_layers);
    reinitialize_weights_according_to_scores_and_inner_info_model(dqn->v_linear_last_layer);
    reinitialize_weights_according_to_scores_and_inner_info_model(dqn->a_hidden_layers);
    reinitialize_weights_according_to_scores_and_inner_info_model(dqn->a_linear_last_layer);
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

void compute_probability_distribution_qr_dqn(float* input , int input_size, dueling_categorical_dqn* dqn){
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
    }
    copy_array(dqn->add_layer, dqn->softmax_layer, dqn->action_size*dqn->n_atoms);
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

void compute_probability_distribution_opt_qr_dqn(float* input , int input_size, dueling_categorical_dqn* dqn, dueling_categorical_dqn* dqn_wlp){
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
    }
    
    copy_array(dqn_wlp->add_layer, dqn_wlp->softmax_layer, dqn->action_size*dqn->n_atoms);
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

float* bp_qr_dqn_opt(float* input, int input_size, float* error, dueling_categorical_dqn* dqn, dueling_categorical_dqn* dqn_wlp){
    int i,j,k;
    for(i = 0; i < dqn->action_size; i++){
        for(j = 0; j < dqn_wlp->n_atoms; j++){
            dqn_wlp->v_linear_layer_error[j]+=error[i*dqn_wlp->n_atoms+j];
        }
    }
    for(i = 0; i < dqn_wlp->action_size; i++){
        for(j = 0; j < dqn_wlp->action_size; j++){
            for(k = 0; k < dqn_wlp->n_atoms; k++){
                if(i == j)
                    dqn_wlp->a_linear_layer_error[i*dqn_wlp->n_atoms+k] += (1.0-1.0/dqn->n_atoms)*error[j*dqn->n_atoms+k];
                else
                    dqn_wlp->a_linear_layer_error[i*dqn_wlp->n_atoms+k] += (-1.0/dqn->n_atoms)*error[j*dqn->n_atoms+k];
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

float* compute_q_functions_qr_dqn(dueling_categorical_dqn* dqn){
    if(dqn == NULL)
        return NULL;
    int i,j;
    for(i = 0; i < dqn->action_size; i++){
        for(j = 0; j < dqn->n_atoms; j++){
            dqn->q_functions[i] += dqn->softmax_layer[i*dqn->n_atoms+j]/dqn->n_atoms;
        }
    }
    return dqn->q_functions;
}

float* get_loss_for_dueling_categorical_dqn(dueling_categorical_dqn* online_net, dueling_categorical_dqn* target_net, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1){
    compute_probability_distribution(state_t_1,state_sizes,online_net);
    compute_probability_distribution(state_t_1,state_sizes,target_net);
    compute_q_functions(target_net);
    compute_q_functions(online_net);
    int action_index = argmax(target_net->q_functions,online_net->action_size);
    copy_array(&online_net->softmax_layer[action_index*online_net->n_atoms], &target_net->softmax_layer[action_index*online_net->n_atoms],online_net->n_atoms);
    reset_dueling_categorical_dqn(online_net);// set a reset for only feed forward
    compute_probability_distribution(state_t,state_sizes,online_net);
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
float* get_loss_for_qr_dqn(dueling_categorical_dqn* online_net, dueling_categorical_dqn* target_net, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1){
    compute_probability_distribution_qr_dqn(state_t_1,state_sizes,online_net);
    compute_probability_distribution_qr_dqn(state_t_1,state_sizes,target_net);
    compute_q_functions_qr_dqn(target_net);
    compute_q_functions_qr_dqn(online_net);
    int action_index = argmax(target_net->q_functions,online_net->action_size);
    float qs = online_net->q_functions[action_index];
    reset_dueling_categorical_dqn(online_net);// set reset for only feed forward
    compute_probability_distribution(state_t,state_sizes,online_net);
    int i;
    float* y_hat = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* vals= (float*)calloc(online_net->n_atoms,sizeof(float));
    for(i = 0; i < online_net->n_atoms; i++){
        if(nonterminal_s_t_1)
            y_hat[i] = reward_t + lambda_value*qs;
        else
            y_hat[i] = reward_t;
            
        if(y_hat[i]- online_net->softmax_layer[action_t*online_net->n_atoms + i] < 0){
            vals[i] = float_abs(((float)(i+1))/((float)(online_net->n_atoms)) - 1);
        }
        else{
            vals[i] = ((float)(i+1))/((float)(online_net->n_atoms));
        }
        vals[i]/=((float)(online_net->n_atoms));
    }
    derivative_huber_loss_array(y_hat, &online_net->softmax_layer[action_t*online_net->n_atoms + i], &online_net->error[action_t*online_net->n_atoms], 1, online_net->n_atoms);
    dot1D(vals, &online_net->error[action_t*online_net->n_atoms], &online_net->error[action_t*online_net->n_atoms], online_net->n_atoms);
    free(y_hat);
    free(vals);
    return online_net->error;
}

float* get_loss_for_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1){
    compute_probability_distribution_opt(state_t_1,state_sizes,target_net,target_net_wlp);
    compute_q_functions(target_net_wlp);
    compute_probability_distribution_opt(state_t_1,state_sizes,online_net,online_net_wlp);
    compute_q_functions(online_net_wlp);
    int action_index = argmax(target_net_wlp->q_functions,online_net_wlp->action_size);
    reset_dueling_categorical_dqn_without_learning_parameters(online_net_wlp);// set reset for only feed forward
    compute_probability_distribution_opt(state_t,state_sizes,online_net, online_net_wlp);
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

float* get_loss_for_qr_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1){
    compute_probability_distribution_opt_qr_dqn(state_t_1,state_sizes,online_net,online_net_wlp);
    compute_probability_distribution_opt_qr_dqn(state_t_1,state_sizes,target_net,target_net_wlp);
    compute_q_functions_qr_dqn(target_net_wlp);
    compute_q_functions_qr_dqn(online_net_wlp);
    int action_index = argmax(target_net_wlp->q_functions,online_net->action_size);
    float qs = online_net_wlp->q_functions[action_index];
    reset_dueling_categorical_dqn_without_learning_parameters(online_net_wlp);// set reset for only feed forward
    compute_probability_distribution_opt_qr_dqn(state_t,state_sizes,online_net,online_net_wlp);
    int i;
    float* y_hat = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* vals= (float*)calloc(online_net->n_atoms,sizeof(float));
    for(i = 0; i < online_net->n_atoms; i++){
        if(nonterminal_s_t_1)
            y_hat[i] = reward_t + lambda_value*qs;
        else
            y_hat[i] = reward_t;
            
        if(y_hat[i] - online_net_wlp->softmax_layer[action_t*online_net->n_atoms + i] < 0){
            vals[i] = float_abs(((float)(i+1))/((float)(online_net->n_atoms)) - 1);
        }
        else{
            vals[i] = ((float)(i+1))/((float)(online_net->n_atoms));
        }
        vals[i]/=((float)(online_net->n_atoms));
    }
    derivative_huber_loss_array(y_hat, &online_net_wlp->softmax_layer[action_t*online_net->n_atoms], &online_net_wlp->error[action_t*online_net->n_atoms], 1, online_net->n_atoms);
    dot1D(vals, &online_net_wlp->error[action_t*online_net->n_atoms], &online_net_wlp->error[action_t*online_net->n_atoms], online_net->n_atoms);
    free(y_hat);
    free(vals);
    return online_net_wlp->error;
}
float* get_loss_for_qr_dqn_opt_with_error(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1, float* new_error, float weight_error){

    compute_probability_distribution_opt_qr_dqn(state_t_1,state_sizes,online_net,online_net_wlp);
    compute_probability_distribution_opt_qr_dqn(state_t_1,state_sizes,target_net,target_net_wlp);
    compute_q_functions_qr_dqn(target_net_wlp);
    compute_q_functions_qr_dqn(online_net_wlp);
    int action_index = argmax(target_net_wlp->q_functions,online_net->action_size);
    float qs = online_net_wlp->q_functions[action_index];
    reset_dueling_categorical_dqn_without_learning_parameters(online_net_wlp);// set reset for only feed forward
    compute_probability_distribution_opt_qr_dqn(state_t,state_sizes,online_net,online_net_wlp);
    int i;
    float* y_hat = (float*)calloc(online_net->n_atoms,sizeof(float));
    float* vals= (float*)calloc(online_net->n_atoms,sizeof(float));
    for(i = 0; i < online_net->n_atoms; i++){
        if(nonterminal_s_t_1)
            y_hat[i] = reward_t + lambda_value*qs;
        else
            y_hat[i] = reward_t;
            
        if(y_hat[i] - online_net_wlp->softmax_layer[action_t*online_net->n_atoms + i] < 0){
            vals[i] = float_abs((((float)(2*i+1))/((float)(2*online_net->n_atoms))) - 1);
        }
        else{
            vals[i] = ((float)(2*i+1))/((float)(2*online_net->n_atoms));
        }
        // expected value at the end
        vals[i]/=online_net->n_atoms;
    }
    derivative_huber_loss_array(y_hat, &online_net_wlp->softmax_layer[action_t*online_net->n_atoms], &online_net_wlp->error[action_t*online_net->n_atoms], 1, online_net->n_atoms);
    dot1D(vals, &online_net_wlp->error[action_t*online_net->n_atoms], &online_net_wlp->error[action_t*online_net->n_atoms], online_net->n_atoms);
    mul_value(&online_net_wlp->error[action_t*online_net->n_atoms], weight_error, &online_net_wlp->error[action_t*online_net->n_atoms], online_net->n_atoms);
    for(i = 0; i < online_net->n_atoms; i++){
        new_error[0] += huber_loss(y_hat[i], online_net_wlp->softmax_layer[action_t*online_net->n_atoms + i],1)*vals[i];
    }
    new_error[0] = float_abs(new_error[0]);
    free(y_hat);
    free(vals);
    return online_net_wlp->error;
}

float* get_loss_for_dueling_categorical_dqn_opt_with_error(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, dueling_categorical_dqn* target_net, dueling_categorical_dqn* target_net_wlp, float* state_t, int action_t, float reward_t, float* state_t_1, float lambda_value, int state_sizes, int nonterminal_s_t_1, float* new_error, float weight_error){
    compute_probability_distribution_opt(state_t_1,state_sizes,target_net,target_net_wlp);
    compute_q_functions(target_net_wlp);
    compute_probability_distribution_opt(state_t_1,state_sizes,online_net,online_net_wlp);
    compute_q_functions(online_net_wlp);
    int action_index = argmax(target_net_wlp->q_functions,online_net_wlp->action_size);
    copy_array(&online_net_wlp->softmax_layer[action_index*online_net->n_atoms], &target_net_wlp->softmax_layer[action_index*online_net->n_atoms],online_net->n_atoms);
    reset_dueling_categorical_dqn_without_learning_parameters(online_net_wlp);
    compute_probability_distribution_opt(state_t,state_sizes,online_net, online_net_wlp);
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
        max_q1 = max_float(max_q1,online_net->q_functions[i]);
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
        ret+=-temp*softmax_arr[i];// final error (no derivative)
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


float compute_l1_dueling_categorical_dqn(dueling_categorical_dqn* online_net, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    // feed forward
    compute_probability_distribution(state_t,size,online_net);
    // q functions
    compute_q_functions(online_net);
    int i,j;
    float ret = 0;
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* softmax_arr2 = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* error = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* derivative_softmax_error= (float*)calloc(online_net->action_size,sizeof(float));// new
    // softmax of the q functions to get the policy
    softmax(online_net->q_functions,softmax_arr,online_net->action_size);
    softmax(q_functions,softmax_arr2,online_net->action_size);
    // computing the error for the threshold
    for(i = 0; i < online_net->action_size; i++){
        float temp = softmax_arr[i]-softmax_arr2[i];
        float temp2 = float_abs(temp);
        if(temp2 == 0)
            error[i] = 0;
        else
            error[i] = -alpha*weight*temp/(2.0*temp2);
        ret+=temp2/2.0;
    }
    derivative_softmax(derivative_softmax_error,softmax_arr,error,online_net->action_size);
    if(clip < 1){
        clip_vector(derivative_softmax_error,-clip,clip,online_net->action_size);
    }
    
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net->error[i*online_net->n_atoms+j]+=derivative_softmax_error[i]*online_net->support[j];
        }
    }
    
    free(softmax_arr);
    free(softmax_arr2);
    free(error);
    free(derivative_softmax_error);

    return ret;
}

float compute_l1_qr_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    if(clip<0)
        clip = -clip;
    // feed forward
    compute_probability_distribution_opt_qr_dqn(state_t,size,online_net,online_net_wlp);
    // q functions
    compute_q_functions_qr_dqn(online_net_wlp);
    int i,j;
    float ret = 0;
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* softmax_arr2 = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* error = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* derivative_softmax_error= (float*)calloc(online_net->action_size,sizeof(float));// new
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    softmax(q_functions,softmax_arr2,online_net->action_size);
    // computing the error for the threshold
    for(i = 0; i < online_net->action_size; i++){
        float temp = softmax_arr[i]-softmax_arr2[i];
        float temp2 = float_abs(temp);
        if(temp2 == 0)
            error[i] = 0;
        else
            error[i] = -alpha*weight*temp/(2.0*temp2);
        ret+=temp2/2.0;
    }
    derivative_softmax(derivative_softmax_error,softmax_arr,error,online_net->action_size);
    if(clip < 1){
        clip_vector(derivative_softmax_error,-clip,clip,online_net->action_size);
    }
    
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=derivative_softmax_error[i]/online_net->n_atoms;
        }
    }
    
    free(softmax_arr);
    free(softmax_arr2);
    free(error);
    free(derivative_softmax_error);

    return ret;
}

float compute_l1_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    if(clip<0)
        clip = -clip;
    // feed forward
    compute_probability_distribution_opt(state_t,size,online_net,online_net_wlp);
    // q functions
    compute_q_functions(online_net_wlp);
    int i,j;
    float ret = 0;
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* softmax_arr2 = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* error = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* derivative_softmax_error= (float*)calloc(online_net->action_size,sizeof(float));// new
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    softmax(q_functions,softmax_arr2,online_net->action_size);
    // computing the error for the threshold
    for(i = 0; i < online_net->action_size; i++){
        float temp = softmax_arr[i]-softmax_arr2[i];
        float temp2 = float_abs(temp);
        if(temp2 == 0)
            error[i] = 0;
        else
            error[i] = -alpha*weight*temp/(2.0*temp2);
        ret+=temp2/2.0;
    }
    derivative_softmax(derivative_softmax_error,softmax_arr,error,online_net->action_size);
    if(clip < 1){
        clip_vector(derivative_softmax_error,-clip,clip,online_net->action_size);
    }
    
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=derivative_softmax_error[i]*online_net->support[j];
        }
    }
    
    free(softmax_arr);
    free(softmax_arr2);
    free(error);
    free(derivative_softmax_error);

    return ret;
}

float compute_l_infinite_dueling_categorical_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    if(clip<0)
        clip = -clip;
    // feed forward
    compute_probability_distribution_opt(state_t,size,online_net,online_net_wlp);
    // q functions
    compute_q_functions(online_net_wlp);
    int i,j;
    float ret = 0;
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* error = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* derivative_softmax_error= (float*)calloc(online_net->action_size,sizeof(float));// new
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    // computing the error for the threshold
    // clipping and multiplying the derivative * alpha and weight and change sign cause we are using gradient descent
    int index = 0;
    float maximum = float_abs(softmax_arr[0]-q_functions[0]);
    for(i = 1; i < online_net->action_size; i++){
        float temp = float_abs(softmax_arr[i]-q_functions[i]);
        if(temp > maximum){
            maximum = temp;
            index = i;
        }
    }
    
    error[index] = -(softmax_arr[index]-q_functions[index])/maximum;
    ret=maximum;
    derivative_softmax(derivative_softmax_error,softmax_arr,error,online_net->action_size);

    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=derivative_softmax_error[i]*online_net->support[j];
        }
    }
    
    free(softmax_arr);
    free(error);
    free(derivative_softmax_error);

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
    float* softmax_arr2 = (float*)calloc(online_net->action_size,sizeof(float));// new
    //float* softmax_derivative_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* other_distr = (float*)calloc(online_net->action_size,sizeof(float));
    float* log1 = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* log2 = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* error = (float*)calloc(online_net->action_size,sizeof(float));
    float ret = 0;// the returning error
    
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    softmax(q_functions,softmax_arr2,online_net->action_size);
    

    
    double summ1 = 0;// new
    double summ2 = 0;// new
    
    summ1 = sum_over_input(softmax_arr,online_net->action_size);// new
    summ2 = sum_over_input(softmax_arr2,online_net->action_size);// new
    // new
    for(i = 0; i < online_net->action_size; i++){
        other_distr[i] = softmax_arr2[i]/summ2;
        log2[i] = log(other_distr[i]);
        log1[i] = log(softmax_arr[i]/summ1);
        if(!bool_is_real(log2[i])){
            log2[i] = -999999;
        }
        if(!bool_is_real(log1[i])){
            log1[i] = -999999;
        }
        
        ret+= softmax_arr[i]*(log1[i]-log2[i]);
    }
    
    double sum = summ1*summ1;
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->action_size; j++){
            if(i == j){
                error[i]-=(alpha*weight*(summ1-softmax_arr[i])*softmax_arr[j]*(log1[i]-log2[i]+1))/sum;
            }
            
            else{
                error[i]+=(alpha*weight*softmax_arr[i]*softmax_arr[j]*(log1[j]-log2[j]+1))/sum;
            }
        }
    }
    /*
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
        ret+=-temp*softmax_arr[i];// final error (no derivative no inversion, we need it to then perform annhilation of alpha or not)
    }
    // first derivative multiplied by other_distr that is the second part
    derivative_softmax(softmax_derivative_arr,softmax_arr,other_distr,online_net->action_size);
    // sum of the 2 derivatives
    sum1D(softmax_derivative_arr,error,error,online_net->action_size);
    // clipping the derivatives as the paper says (no supplementary material found, gg nimps and other publishers, they keep saying hyperparams are in the supplementary
    // material, but after looking for this F@#%$ supplementary material i could not find anything so i need to come with the clipping right value, and this also for the distance threshold
    // used to rescale alpha that is the most important part)
    * */
    clip_vector(error,-clip,clip,online_net->action_size);
    /*for(i = 0; i < online_net->action_size; i++){
        printf("%f ",error[i]);
    }
    printf("\n");*/
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer of the network
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=error[i]*online_net_wlp->support[j];
        }
    }
    free(softmax_arr);
    free(softmax_arr2);
    //free(softmax_derivative_arr);
    free(other_distr);
    free(log1);
    free(log2);
    free(error);
    return ret;
}

/*
float compute_kl_qr_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    if(clip<0)
        clip = -clip;
    // feed forward
    compute_probability_distribution_opt_qr_dqn(state_t,size,online_net,online_net_wlp);
    // q functions
    compute_q_functions_qr_dqn(online_net_wlp);
    int i,j;
    float ret = 0;
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* softmax_arr2 = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* loss = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* error = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* derivative_softmax_error= (float*)calloc(online_net->action_size,sizeof(float));// new
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    // the softmax are already computed when stored
    //softmax(q_functions,softmax_arr2,online_net->action_size);
    // computing the error for the threshold
    kl_divergence(softmax_arr, q_functions, loss, online_net->action_size);
    
    
    // computing the derivative
    derivative_kl_divergence(softmax_array, q_functions, error, online_net->action_size);
    
    // clipping and multiplying the derivative * alpha and weight and change sign cause we are using gradient descent
    for(i = 0; i < online_net->action_size; i++){
        if(loss[i] > clip){
            loss[i] = clip;
            error[i] = 0;
        }
        if(loss[i] < -clip){
            loss[i] = -clip;
            error[i] = 0;
        }
        error[i]*=-alpha*weight;
    }
    
    ret+= alpha*sum_over_input(loss,online_net->action_size);

    
    derivative_softmax(derivative_softmax_error,softmax_arr,error,online_net->action_size);
    
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=derivative_softmax_error[i]/online_net->n_atoms;
        }
    }
    
    free(softmax_arr);
    free(softmax_arr2);
    free(error);
    free(derivative_softmax_error);
    free(loss);

    return ret;
}
* */

float compute_l_infinite_qr_dqn_opt(dueling_categorical_dqn* online_net,dueling_categorical_dqn* online_net_wlp, float* state_t, float* q_functions,  float weight, float alpha, float clip){
    
    // getting the input size of the network
    int size = get_input_layer_size_dueling_categorical_dqn(online_net);
    if(clip<0)
        clip = -clip;
    // feed forward
    compute_probability_distribution_opt_qr_dqn(state_t,size,online_net,online_net_wlp);
    // q functions
    compute_q_functions_qr_dqn(online_net_wlp);
    int i,j;
    float ret = 0;
    float* softmax_arr = (float*)calloc(online_net->action_size,sizeof(float));
    float* error = (float*)calloc(online_net->action_size,sizeof(float));// new
    float* derivative_softmax_error= (float*)calloc(online_net->action_size,sizeof(float));// new
    // softmax of the q functions to get the policy
    softmax(online_net_wlp->q_functions,softmax_arr,online_net->action_size);
    // the softmax are already computed when stored
    //softmax(q_functions,softmax_arr2,online_net->action_size);
    // computing the error for the threshold
    
    
    // clipping and multiplying the derivative * alpha and weight and change sign cause we are using gradient descent
    int index = 0;
    float maximum = float_abs(softmax_arr[0]-q_functions[0]);
    for(i = 1; i < online_net->action_size; i++){
        float temp = float_abs(softmax_arr[i]-q_functions[i]);
        if(temp > maximum){
            maximum = temp;
            index = i;
        }
    }
    
    error[index] = -(softmax_arr[index]-q_functions[index])/maximum;
    ret=maximum;
    
    derivative_softmax(derivative_softmax_error,softmax_arr,error,online_net->action_size);
    
    // we got the partial derivatives of the q functions, now we need to compute the partial derivatives respect to the softmax final layer
    for(i = 0; i < online_net->action_size; i++){
        for(j = 0; j < online_net->n_atoms; j++){
            online_net_wlp->error[i*online_net->n_atoms+j]+=derivative_softmax_error[i]/online_net->n_atoms;
        }
    }
    
    free(softmax_arr);
    free(error);
    free(derivative_softmax_error);

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

void inference_dqn(dueling_categorical_dqn* dqn){
    inference_model(dqn->shared_hidden_layers);
    inference_model(dqn->a_hidden_layers);
    inference_model(dqn->v_hidden_layers);
    inference_model(dqn->a_linear_last_layer);
    inference_model(dqn->v_linear_last_layer);
}

void train_dqn(dueling_categorical_dqn* dqn){
    train_model(dqn->shared_hidden_layers);
    train_model(dqn->a_hidden_layers);
    train_model(dqn->v_hidden_layers);
    train_model(dqn->a_linear_last_layer);
    train_model(dqn->v_linear_last_layer);
}

void dueling_dqn_eliminate_noisy_layers(dueling_categorical_dqn* dqn){
    model_eliminate_noisy_layers(dqn->shared_hidden_layers);
    model_eliminate_noisy_layers(dqn->a_hidden_layers);
    model_eliminate_noisy_layers(dqn->v_hidden_layers);
    model_eliminate_noisy_layers(dqn->a_linear_last_layer);
    model_eliminate_noisy_layers(dqn->v_linear_last_layer);
}

void assign_noise_arrays_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float** noise_biases1, float** noise1,float** noise_biases2, float** noise2,float** noise_biases3, float** noise3,float** noise_biases4, float** noise4,float** noise_biases5, float** noise5){
    assign_noise_arrays_model(dqn->shared_hidden_layers, noise_biases1, noise1);
    assign_noise_arrays_model(dqn->a_hidden_layers, noise_biases2, noise2);
    assign_noise_arrays_model(dqn->v_hidden_layers, noise_biases3, noise3);
    assign_noise_arrays_model(dqn->a_linear_last_layer, noise_biases4, noise4);
    assign_noise_arrays_model(dqn->v_linear_last_layer, noise_biases5, noise5);
}
