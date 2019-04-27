#include "llab.h"

/* This function builds a fully-connected layer according to the fcl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int input:= number of neurons of the previous layer
 *             @ int output:= number of neurons of the current layer
 *             @ int layer:= number of sequential layer [1,∞)
 *             @ int dropout_flag:= is set to 0 if you don't want to apply dropout
 *             @ int activation_flag:= is set to 0 if you don't want to apply the activation function else read in layers.h
 *             @ float dropout_threshold:= [0,1]
 * */
fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold){
    if(!input || !output || layer < 0){
        fprintf(stderr,"Error: input, output params must be > 0 and layer > -1\n");
        exit(1);
    }
    
    if(!activation_flag){
        fprintf(stderr,"Error: there must be some activation in the layer otherwise the neural_network is not able to learn everything\n");
        exit(1);
    }
    int i,j;
    
    fcl* f = (fcl*)malloc(sizeof(fcl));
    f->input = input;
    f->output = output;
    f->layer = layer;
    f->dropout_flag = dropout_flag;
    f->activation_flag = activation_flag;
    f->dropout_threshold = dropout_threshold;
    f->weights = (float*)malloc(sizeof(float)*output*input);
    f->d_weights = (float*)calloc(output*input,sizeof(float));
    f->d1_weights = (float*)calloc(output*input,sizeof(float));
    f->d2_weights = (float*)calloc(output*input,sizeof(float));
    f->biases = (float*)calloc(output,sizeof(float));
    f->d_biases = (float*)calloc(output,sizeof(float));
    f->d1_biases = (float*)calloc(output,sizeof(float));
    f->d2_biases = (float*)calloc(output,sizeof(float));
    f->pre_activation = (float*)calloc(output,sizeof(float));
    f->dropout_temp = (float*)calloc(output,sizeof(float));
    f->temp = (float*)calloc(output,sizeof(float));
    f->temp3 = (float*)calloc(output,sizeof(float));
    f->temp2 = (float*)calloc(input,sizeof(float));
    f->error2 = (float*)calloc(input,sizeof(float));
    f->post_activation = (float*)calloc(output,sizeof(float));

    
    f->dropout_mask = (float*)calloc(output,sizeof(float));

    
    for(i = 0; i < output; i++){
        for(j = 0; j < input; j++){
            f->weights[i*input+j] = random_general_gaussian(0, (float)input);
        }
        if(dropout_flag)
            f->dropout_mask[i] = 1;
    }
    
    return f;
}

/* Given a fcl* structure this function frees the space allocated by this structure*/
void free_fully_connected(fcl* f){
    if(f == NULL){
        return;
    }
    
    free(f->weights);
    free(f->d_weights);
    free(f->d1_weights);
    free(f->d2_weights);
    free(f->biases);
    free(f->d_biases);
    free(f->d1_biases);
    free(f->d2_biases);
    free(f->pre_activation);
    free(f->post_activation);
    free(f->dropout_mask);
    free(f->dropout_temp);
    free(f->temp);
    free(f->temp2);
    free(f->temp3);
    free(f->error2);
    free(f);    
}

/* This function builds a convolutional layer according to the cl structure defined in layers.h
 * 
 * Input:
 *             
 *             @ int channales:= number of channels of the previous layer
 *             @ int input_rows:= number of rows of each channel of the previous layer
 *             @ int input_cols:= number of columns of each channel of the previous layer
 *             @ int kernel_rows:= rows of the kernels of the current layers
 *             @ int kernel_cols:= columns of the kernels of the current layer
 *             @ int n_kernels:= number of kernels applied to the trevious layer to create n_kernels feature maps
 *             @ int stride1_rows:= stride used by the kernels on the rows
 *             @ int stride1_cols:= stride used by the kernels on the columns
 *             @ int padding1_rows:= the padding added after activation-normalization to the rows
 *             @ int padding1_cols:= the padding added after activation-normalization to the columns
 *             @ int stride2_rows:= the stride used by the pooling on the rows
 *             @ int stride2_cols:= the stride used by the pooling on the columns
 *             @ int padding2_rows:= the padding added after pooling on the rows
 *             @ int padding2_cols:= the padding added after pooling on the columns
 *             @ int pooling_rows:= the space of the pooling on the rows
 *             @ int pooling_cols:= the space of the pooling on the columns
 *             @ int normalization_flag:= is set to 1 if you wan't to apply normalization
 *             @ int activation_flag:= is set to 1 if you want to apply activation function
 *             @ int pooling_flag:= is set to 1 if you want to apply pooling
 * 
 * */
 
cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int layer, int convolutional_flag){
    if(!channels || !input_rows || !input_cols || !kernel_rows || !kernel_cols || !n_kernels || !stride1_rows || !stride1_cols || (pooling_flag && (!stride2_rows || !stride2_cols))){
        fprintf(stderr,"Error: channles, input_rows, input_cols, kernel_rows, kernel_cols, n_kernels, stride2_rows stride2_cols, stride2_rows, stride2_cols params must be > 0\n");
        exit(1);
    }
    
    if(padding1_rows!=padding1_cols || padding2_rows != padding2_cols || stride1_cols!= stride1_rows || stride2_cols!= stride2_rows){
        fprintf(stderr,"Error: stride1_rows must be equal to stride1_cols, padding1_rows must be equal to padding1_cols and stride2_rows must be equal to stride2_cols, padding2_rows must be equal to padding2_cols\n");
        exit(1);
    }
    
    if(padding1_rows && normalization_flag == BATCH_NORMALIZATION){
        fprintf(stderr,"Error: you cannot pad before the pooling if you have also a batch normalization layer as next computation(you can pad after the pooling: padding2_rows)\n");
        exit(1);
    }
    
    if(convolutional_flag == NO_CONVOLUTION && pooling_flag == NO_POOLING){
        fprintf(stderr,"Error: you don't apply convolution neither pooling\n");
        exit(1);
    }
    
    if(convolutional_flag == NO_CONVOLUTION && n_kernels != channels){
        fprintf(stderr,"Error: if you don't apply convolution, your n_kernels param should be equal to channels, 'cause n_kernels indicates the channel of the current_layer\n");
        exit(1);
    }
    
    
    
    int i,j;
    cl* c = (cl*)malloc(sizeof(cl));
    c->layer = layer;
    c->channels = channels;
    c->input_rows = input_rows;
    c->input_cols = input_cols;
    c->kernel_rows = kernel_rows;
    c->kernel_cols = kernel_cols;
    c->n_kernels = n_kernels;
    c->stride1_rows = stride1_rows;
    c->stride1_cols = stride1_cols;
    c->padding1_rows = padding1_rows;
    c->padding1_cols = padding1_cols;
    c->stride2_rows = stride2_rows;
    c->stride2_cols = stride2_cols;
    c->padding2_rows = padding2_rows;
    c->padding2_cols = padding2_cols;
    c->pooling_rows = pooling_rows;
    c->pooling_cols = pooling_cols;
    c->normalization_flag = normalization_flag;
    c->activation_flag = activation_flag;
    c->pooling_flag = pooling_flag;
    c->kernels = (float**)malloc(sizeof(float*)*n_kernels);
    c->d_kernels = (float**)malloc(sizeof(float*)*n_kernels);
    c->d1_kernels = (float**)malloc(sizeof(float*)*n_kernels);
    c->d2_kernels = (float**)malloc(sizeof(float*)*n_kernels);
    c->biases = (float*)calloc(n_kernels,sizeof(float));
    c->d_biases = (float*)calloc(n_kernels,sizeof(float));
    c->d1_biases = (float*)calloc(n_kernels,sizeof(float));
    c->d2_biases = (float*)calloc(n_kernels,sizeof(float));
    c->convolutional_flag = convolutional_flag;
    if(!bool_is_real((float)((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows)))
        c->rows1 = 0;
    else
        c->rows1 = ((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows);
    if(convolutional_flag == CONVOLUTION){
        if(!bool_is_real((float)((((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows) - pooling_rows)/stride2_rows + 1 + 2*padding2_rows)))
            c->rows2 = 0;
        else
            c->rows2 = ((((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows) - pooling_rows)/stride2_rows + 1 + 2*padding2_rows);
        }
    else{
        if(!bool_is_real((float)((input_rows-pooling_rows)/stride2_rows +1 + 2*padding2_rows)))
            c->rows2 = 0;
        else
            c->rows2 = ((input_rows-pooling_rows)/stride2_rows +1 + 2*padding2_rows);
    }
    if(!bool_is_real((float)((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols)))
        c->cols1 = 0;
    else
        c->cols1 = ((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols);
    if(convolutional_flag == CONVOLUTION){
        if(!bool_is_real((float)((((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols) - pooling_cols)/stride2_cols + 1 + 2*padding2_cols)))
            c->cols2 = 0;
        else
            c->cols2 = ((((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols) - pooling_cols)/stride2_cols + 1 + 2*padding2_cols);
        }
    else{
        if(!bool_is_real((float)((input_cols-pooling_cols)/stride2_cols +1 + 2*padding2_cols)))
            c->cols2 = 0;
        else
            c->cols2 = ((input_cols-pooling_cols)/stride2_cols +1 + 2*padding2_cols);
    }
    
    c->temp = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    c->temp2 = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    c->temp3 = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    c->error2 = (float*)calloc(channels*input_rows*input_cols,sizeof(float));
    
    
    c->pre_activation = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    
    c->post_activation = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));

    

    c->post_normalization = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));



    c->post_pooling = (float*)calloc(n_kernels*c->rows2*c->cols2,sizeof(float));

    
    
    for(i = 0; i < n_kernels; i++){
        c->kernels[i] = (float*)malloc(sizeof(float)*channels*kernel_rows*kernel_cols);
        c->d_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
        c->d1_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
        c->d2_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
        for(j = 0; j < channels*kernel_rows*kernel_cols; j++){
            c->kernels[i][j] = random_general_gaussian(0, (float)channels*input_rows*input_cols);
        }
    }
    return c;
}

/* Given a cl* structure this function frees the space allocated by this structure*/
void free_convolutional(cl* c){
    if(c == NULL){
        return;
    }
    
    int i;
    for(i = 0; i < c->n_kernels; i++){
        free(c->kernels[i]);
        free(c->d_kernels[i]);
        free(c->d1_kernels[i]);
        free(c->d2_kernels[i]);
    }
    free(c->kernels);
    free(c->d_kernels);
    free(c->d1_kernels);
    free(c->d2_kernels);
    free(c->biases);
    free(c->d_biases);
    free(c->d1_biases);
    free(c->d2_biases);
    free(c->pre_activation);
    free(c->post_activation);
    free(c->post_normalization);
    free(c->post_pooling);
    free(c->temp);
    free(c->temp2);
    free(c->temp3);
    free(c->error2);
    free(c);
}

/* This function builds a residual layer according to the rl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int channels:= n. channels of the current layer
 *             @ int input_rows:= n. rows per channel of the current layer
 *             @ int input_cols:= n. columns per channel of the current layer
 *             @ int n_cl:= number of cls structure in this residual layer
 *             @ cl** cls:= the cls structures of the layer
 * 
 * */
rl* residual(int channels, int input_rows, int input_cols, int n_cl, cl** cls){
    if(!channels || !input_rows || !input_cols || (!n_cl) || (!n_cl && cls != NULL)){
        fprintf(stderr,"Error: channels, input rows, input cols params must be > 0 and or n_cl or n_fcl must be > 0\n");
        exit(1);
    }
    rl* r = (rl*)malloc(sizeof(rl));
    r->channels = channels;
    r->input_rows = input_rows;
    r->input_cols = input_cols;
    r->n_cl = n_cl;
    r->cls =cls;
    r->input = (float*)calloc(channels*input_rows*input_cols,sizeof(float));
    r->cl_output = convolutional(channels,input_rows,input_cols,1,1,channels,1,1,0,0,1,1,0,0,0,0,0,RELU,0,cls[n_cl-1]->layer,CONVOLUTION);
    return r;
    
}

/* Given a rl* structure this function frees the space allocated by this structure*/
void free_residual(rl* r){
    int i;
    for(i = 0; i < r->n_cl; i++){
        free_convolutional(r->cls[i]);
    }
    
    free(r->cls);
    free(r->input);
    free_convolutional(r->cl_output);
    free(r);
}


/* this functions build a batch normalization layer
 * 
 * Input:
 * 
 *             @ int batch_size:= the batch size used
 *             @ int vector_input_dimension:= the dimension of the input of this layer, or the output dimension of the previous layer
 * 
 * */
bn* batch_normalization(int batch_size, int vector_input_dimension, int layer, int activation_flag){
    if(batch_size <= 1 || vector_input_dimension < 1){
        fprintf(stderr,"Error: remember if you are using online learning (batch_size = 1) batch normalization is useless, and remember also that vector input dimension must be >= 1\n");
        exit(1);
    }
    int i;
    bn* b = (bn*)malloc(sizeof(bn));
    b->layer = layer;
    b->batch_size = batch_size; 
    b->vector_dim = vector_input_dimension;
    
    b->input_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp_vectors = (float**)malloc(sizeof(float*)*batch_size); 
    b->error2 = (float**)malloc(sizeof(float*)*batch_size); 
    b->temp1 = (float**)malloc(sizeof(float*)*batch_size); 
    b->outputs = (float**)malloc(sizeof(float*)*batch_size);
    b->post_activation = (float**)malloc(sizeof(float*)*batch_size);
    
    b->gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_gamma = (float*)calloc(vector_input_dimension,sizeof(float));
    b->beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d1_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->d2_beta = (float*)calloc(vector_input_dimension,sizeof(float));
    b->mean = (float*)calloc(vector_input_dimension,sizeof(float));
    b->var = (float*)calloc(vector_input_dimension,sizeof(float));
    b->temp2 = (float*)calloc(vector_input_dimension,sizeof(float));
    b->final_mean = (float*)calloc(vector_input_dimension,sizeof(float));
    b->final_var = (float*)calloc(vector_input_dimension,sizeof(float));
    b->mode_flag = BATCH_NORMALIZATION_TRAINING_MODE;
    
    for(i = 0; i < batch_size; i++){
        b->input_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp_vectors[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->error2[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->temp1[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->outputs[i] = (float*)calloc(vector_input_dimension,sizeof(float));
        b->post_activation[i] = (float*)calloc(vector_input_dimension,sizeof(float));
    }
    
    for(i = 0; i < vector_input_dimension; i++){
        b->gamma[i] = 1;
    }
    
    b->epsilon = EPSILON;
    
    return b;
}

/* This functions deallocates the space allocated by a bn structure
 * 
 * Input:
 * 
 *             @ bn* b:= the structure
 * 
 * */
void free_batch_normalization(bn* b){
    if(b == NULL)
        return;
    int i;
    for(i = 0; i < b->batch_size; i++){
        free(b->input_vectors[i]);
        free(b->temp_vectors[i]);
        free(b->error2[i]);
        free(b->temp1[i]);
        free(b->outputs[i]);
        free(b->post_activation[i]);
    }
    free(b->input_vectors);
    free(b->temp_vectors);
    free(b->error2);
    free(b->temp1);
    free(b->outputs);
    free(b->post_activation);
    
    free(b->gamma);
    free(b->d_gamma);
    free(b->d1_gamma);
    free(b->d2_gamma);
    free(b->beta);
    free(b->d_beta);
    free(b->d1_beta);
    free(b->d2_beta);
    free(b->temp2);
    free(b->final_mean);
    free(b->final_var);
    free(b->mean);
    free(b->var);
    free(b);
}

/* This function saves a batch normalized layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ bn* b:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_bn(bn* b, int n){
    if(b == NULL)
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
    
    i = fwrite(&b->layer,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->batch_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->vector_dim,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(&b->activation_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->gamma,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    
    i = fwrite(b->beta,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->final_mean,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fwrite(b->final_var,sizeof(float)*(b->vector_dim),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a bn layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    free(s);
    
    
}

bn* load_bn(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int batch_size = 0,vector_dim = 0, layer = 0, activation_flag;
    float* gamma;
    float* beta;
    float* final_mean;
    float* final_var;
    
    i = fread(&layer,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&batch_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&vector_dim,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&activation_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    gamma = (float*)malloc(sizeof(float)*vector_dim);
    beta = (float*)malloc(sizeof(float)*vector_dim);
    final_mean = (float*)malloc(sizeof(float)*vector_dim);
    final_var = (float*)malloc(sizeof(float)*vector_dim);
    
    i = fread(&gamma,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    i = fread(&beta,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    i = fread(&final_mean,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    i = fread(&final_var,sizeof(float)*vector_dim,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a bn layer\n");
        exit(1);
    }
    
    
    
    bn* b = batch_normalization(batch_size,vector_dim, layer, activation_flag);
    
    copy_array(gamma,b->gamma,vector_dim);
    copy_array(beta,b->beta,vector_dim);
    copy_array(final_mean,b->final_mean,vector_dim);
    copy_array(final_var,b->final_var,vector_dim);
    
    free(gamma);
    free(beta);
    free(final_mean);
    free(final_var);
    
    return b;
    
}
/* This function saves a fully-connected layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ fcl* f:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_fcl(fcl* f, int n){
    if(f == NULL)
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
    
    i = fwrite(&f->input,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->output,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->layer,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->dropout_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->activation_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->dropout_threshold,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->weights,sizeof(float)*(f->input)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->biases,sizeof(float)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    free(s);
    
}

/* This function copies the values in weights and biases vector in the weights 
 * and biases vector of a fcl structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the structure
 *             @ float* weights:= the weights that must be copied (size = f->output*f->input)
 *             @ float* biases:= the biases that must be copied (size = f->output)
 * 
 * */
void copy_fcl_params(fcl* f, float* weights, float* biases){
    int i,j;
    for(i = 0; i < f->output; i++){
        for(j = 0; j < f->input; j++){
            f->weights[i*f->input+j] = weights[i*f->input+j];
        }
        f->biases[i] = biases[i];
    }
}

/* This function loads a fully-connected layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
fcl* load_fcl(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int input = 0,output = 0,layer = 0,dropout_flag = 0,activation_flag = 0;
    float dropout_threshold = 0;
    float* weights;
    float* biases;
    
    i = fread(&input,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&output,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&layer,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&dropout_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&activation_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&dropout_threshold,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    weights = (float*)malloc(sizeof(float)*input*output);
    biases = (float*)malloc(sizeof(float)*output);
    
    i = fread(weights,sizeof(float)*(input)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(biases,sizeof(float)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    fcl* f = fully_connected(input,output,layer,dropout_flag,activation_flag,dropout_threshold);
    copy_fcl_params(f,weights,biases);
    
    free(weights);
    free(biases);
    return f;
}

/* This function saves a convolutional layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ cl* f:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_cl(cl* f, int n){
    if(f == NULL)
        return;
    int i,k;
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
    
    i = fwrite(&f->convolutional_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->channels,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->input_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->input_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->layer,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->kernel_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->kernel_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->n_kernels,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->stride1_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->stride1_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->padding1_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->padding1_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->stride2_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->stride2_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->padding2_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->padding2_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->pooling_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->pooling_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->normalization_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->activation_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->pooling_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->rows1,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->cols1,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->rows2,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->cols2,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    for(k = 0; k < f->n_kernels; k++){
        i = fwrite((f->kernels[k]),sizeof(float)*f->channels*f->kernel_rows*f->kernel_cols,1,fw);

    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a cl layer\n");
            exit(1);
        }
    }
    
    i = fwrite(f->biases,sizeof(float)*f->n_kernels,1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    free(s);
    
}

/* This function copies the values in weights and biases vector in the weights 
 * and biases vector of a cl structure
 * 
 * Input:
 * 
 *             @ cl* f:= the structure
 *             @ float* kernels:= the weights that must be copied (size = f->n_kernels - f->channles*f->kernel_rows*f->kernel_cols)
 *             @ float* biases:= the biases that must be copied (size = f->output)
 * 
 * */
void copy_cl_params(cl* f, float** kernels, float* biases){
    int i,j,k,z;
    for(i = 0; i < f->n_kernels; i++){
        for(j = 0; j < f->channels; j++){
            for(k = 0; k < f->kernel_rows; k++){
                for(z = 0; z < f->kernel_cols; z++){
                    f->kernels[i][j*f->kernel_rows*f->kernel_cols + k*f->kernel_cols + z] = kernels[i][j*f->kernel_rows*f->kernel_cols + k*f->kernel_cols + z];
                }
            }
            
        }
        f->biases[i] = biases[i];
    }
}

/* This function loads a convolutional layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
cl* load_cl(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i,k;
    
    int channels = 0, input_rows = 0, input_cols = 0,layer = 0, convolutional_flag;
    int kernel_rows = 0, kernel_cols = 0, n_kernels = 0;
    int stride1_rows = 0, stride1_cols = 0, padding1_rows = 0, padding1_cols = 0;
    int stride2_rows = 0, stride2_cols = 0, padding2_rows = 0, padding2_cols = 0;
    int pooling_rows = 0, pooling_cols = 0;
    int normalization_flag = 0, activation_flag = 0, pooling_flag = 0;
    int rows1 = 0, cols1 = 0, rows2 = 0,cols2 = 0;
    float** kernels;
    float* biases;
    
    i = fread(&convolutional_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&channels,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&input_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&input_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&layer,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&kernel_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&kernel_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&n_kernels,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&stride1_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&stride1_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&padding1_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&padding1_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&stride2_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&stride2_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&padding2_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&padding2_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&pooling_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&pooling_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&normalization_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&activation_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&pooling_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&rows1,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&cols1,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&rows2,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&cols2,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    kernels = (float**)malloc(sizeof(float*)*n_kernels);
    biases = (float*)malloc(sizeof(float)*n_kernels);
    
    for(k = 0; k < n_kernels; k++){
        kernels[k] = (float*)malloc(sizeof(float)*channels*kernel_rows*kernel_cols);
        i = fread(kernels[k],sizeof(float)*channels*kernel_rows*kernel_cols,1,fr);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a cl layer\n");
            exit(1);
        }
    }
    
    i = fread(biases,sizeof(float)*n_kernels,1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    cl* f = convolutional(channels, input_rows, input_cols, kernel_rows, kernel_cols, n_kernels, stride1_rows, stride1_cols, padding1_rows, padding1_cols, stride2_rows, stride2_cols, padding2_rows, padding2_cols, pooling_rows, pooling_cols, normalization_flag, activation_flag, pooling_flag, layer, convolutional_flag);
    copy_cl_params(f,kernels,biases);
    
    for(i= 0; i < n_kernels; i++){
        free(kernels[i]);
    }
    free(kernels);
    free(biases);
    return f;
}


/* This function saves a residual layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ rl* f:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_rl(rl* f, int n){
    if(f == NULL)
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
    
    i = fwrite(&f->channels,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->input_rows,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->input_cols,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->n_cl,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    for(i = 0; i < f->n_cl; i++){
        save_cl(f->cls[i],n);
    }
    
    free(s);
}


/* This function loads a residual layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
rl* load_rl(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int channels = 0,input_rows = 0,input_cols = 0,n_cl = 0;
    cl** cls;
    
    i = fread(&channels,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&input_rows,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&input_cols,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&n_cl,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    cls = (cl**)malloc(sizeof(cl*)*n_cl);
    for(i = 0; i < n_cl; i++){
        cls[i] = load_cl(fr);
    }
    
    rl* f = residual(channels,input_rows,input_cols,n_cl,cls);
    return f;
}



/* This function returns a bn* layer that is the same copy of the input f
 * arrays used during the feed forward and backpropagation
 * 
 * Input:
 * 
 *             @ bn* f:= the batch normalized layer that must be copied
 * 
 * */ 
bn* copy_bn(bn* b){
    if(b == NULL)
        return NULL;
    bn* copy = batch_normalization(b->batch_size,b->vector_dim, b->layer, b->activation_flag);
    copy_array(b->gamma,copy->gamma,b->vector_dim);
    copy_array(b->d_gamma,copy->d_gamma,b->vector_dim);
    copy_array(b->d1_gamma,copy->d1_gamma,b->vector_dim);
    copy_array(b->d2_gamma,copy->d2_gamma,b->vector_dim);
    copy_array(b->beta,copy->beta,b->vector_dim);
    copy_array(b->d_beta,copy->d_beta,b->vector_dim);
    copy_array(b->d1_beta,copy->d1_beta,b->vector_dim);
    copy_array(b->d2_beta,copy->d2_beta,b->vector_dim);
    copy_array(b->final_mean,copy->final_mean,b->vector_dim);
    copy_array(b->final_var,copy->final_var,b->vector_dim);
    
    return copy;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array.
 * You have a fcl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 * 
 * */
fcl* copy_fcl(fcl* f){
    if(f == NULL)
        return NULL;
    fcl* copy = fully_connected(f->input, f->output,f->layer, f->dropout_flag,f->activation_flag,f->dropout_threshold);
    copy_array(f->weights,copy->weights,f->output*f->input);
    copy_array(f->d_weights,copy->d_weights,f->output*f->input);
    copy_array(f->d1_weights,copy->d1_weights,f->output*f->input);
    copy_array(f->d2_weights,copy->d2_weights,f->output*f->input);
    copy_array(f->biases,copy->biases,f->output);
    copy_array(f->d_biases,copy->d_biases,f->output);
    copy_array(f->d1_biases,copy->d1_biases,f->output);
    copy_array(f->d2_biases,copy->d2_biases,f->output);
    return copy;
}

/* This function returns a cl* layer that is the same copy of the input f
 * except for the activation arrays , the post normalization and post polling arrays
 * You have a cl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms 
 * 
 * Input:
 * 
 *             @ cl* f:= the convolutional layer that must be copied
 * 
 * */
cl* copy_cl(cl* f){
    if(f == NULL)
        return NULL;
    cl* copy = convolutional(f->channels,f->input_rows,f->input_cols,f->kernel_rows,f->kernel_cols,f->n_kernels,f->stride1_rows,f->stride1_cols,f->padding1_rows,f->padding1_cols,f->stride2_rows,f->stride2_cols,f->padding2_rows,f->padding2_cols,f->pooling_rows,f->pooling_cols,f->normalization_flag,f->activation_flag,f->pooling_flag,f->layer, f->convolutional_flag);
    
    int i;
    for(i = 0; i < f->n_kernels; i++){
        copy_array(f->kernels[i],copy->kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d_kernels[i],copy->d_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d1_kernels[i],copy->d1_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d2_kernels[i],copy->d2_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
    }
    
    copy_array(f->biases,copy->biases,f->n_kernels);
    copy_array(f->d_biases,copy->d_biases,f->n_kernels);
    copy_array(f->d1_biases,copy->d1_biases,f->n_kernels);
    copy_array(f->d2_biases,copy->d2_biases,f->n_kernels);
    
    return copy;
}


/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * You have a rl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 * 
 * */
rl* copy_rl(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    cl** cls = (cl**)malloc(sizeof(cl*)*f->n_cl);
    for(i = 0; i < f->n_cl; i++){
        cls[i] = copy_cl(f->cls[i]);
    }
    
    rl* copy = residual(f->channels, f->input_rows, f->input_cols, f->n_cl, cls);
    return copy;
}


/* this function reset all the arrays of a batch normalized layer
 * 
 * 
 * Input:
 * 
 *             @ bn* b:= a bn* f layer
 * 
 * */
bn* reset_bn(bn* b){
    if(b == NULL)
        return NULL;
    int i,j;
    for(i = 0; i < b->vector_dim; i++){
        for(j = 0; j < b->batch_size; j++){
            b->input_vectors[j][i] = 0;
            b->temp_vectors[j][i] = 0;
            b->outputs[j][i] = 0;
            b->post_activation[j][i] = 0;
            b->error2[j][i] = 0;
            b->temp1[j][i] = 0;
        }
        
        b->d_gamma[i] = 0; 
        b->d_beta[i] = 0; 
        b->temp2[i] = 0; 
        b->mean[i] = 0; 
        b->var[i] = 0; 
    } 
}


/* this function reset all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation
 * You have a fcl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * 
 * Input:
 * 
 *             @ fcl* f:= a fcl* f layer
 * 
 * */
fcl* reset_fcl(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    for(i = 0; i < f->output*f->input; i++){
        if(i < f->output){
            f->pre_activation[i] = 0;
            f->post_activation[i] = 0;
            f->d_biases[i] = 0;
            if(f->dropout_flag)
                f->dropout_mask[i] = 1;
            f->dropout_temp[i] = 0;
            f->temp[i] = 0;
            f->temp3[i] = 0;
            
        }
        if(i < f->input){
            f->temp2[i] = 0;
            f->error2[i] = 0;
        }
        f->d_weights[i] = 0;
    }
    return f;
}


/* this function reset all the arrays of a convolutional layer
 * used during the feed forward and backpropagation
 * You have a cl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ cl* f:= a cl* f layer
 * 
 * */
cl* reset_cl(cl* f){
    if(f == NULL)
        return NULL;
    
    int i,j;
    for(i = 0; i < f->n_kernels; i++){
        for(j = 0; j < f->channels*f->kernel_rows*f->kernel_cols; j++){
            f->d_kernels[i][j] = 0;
        }
        
        f->d_biases[i] = 0;
    }
    
    for(i = 0; i < f->n_kernels*f->rows1*f->cols1; i++){
        f->pre_activation[i] = 0;
        f->post_activation[i] = 0;
        f->post_normalization[i] = 0;
        f->temp[i] = 0;
        f->temp2[i] = 0;
        f->temp3[i] = 0;
    }
    
    for(i = 0; i < f->n_kernels*f->rows2*f->cols2; i++){
        f->post_pooling[i] = 0;
    }
    
    for(i = 0; i < f->channels*f->input_rows*f->input_cols; i++){
        f->error2[i] = 0;
    }
    
    return f;
}


/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl(f->cls[i]);
    }
    
    reset_cl(f->cl_output);
    
    for(i = 0; i < f->channels*f->input_rows*f->input_cols; i++){
        f->input[i] = 0;
    }
    
    return f;
}

/* this function compute the space allocated by the arrays of f
 * 
 * Input:
 * 
 *             fcl* f:= the fully-connected layer f
 * 
 * */
unsigned long long int size_of_fcls(fcl* f){
    unsigned long long int sum = 0;
    sum += ((unsigned long long int)(f->input*f->output*4*sizeof(float)));
    sum += ((unsigned long long int)(f->output*10*sizeof(float)));
    sum += ((unsigned long long int)(f->input*2*sizeof(float)));
    return sum;
}


/* this function compute the space allocated by the arrays of f
 * 
 * Input:
 * 
 *             cl* f:= the convolutional layer f
 * 
 * */
unsigned long long int size_of_cls(cl* f){
    unsigned long long int sum = 0;
    sum += ((unsigned long long int)(f->n_kernels*f->channels*f->kernel_cols*f->kernel_rows*4*sizeof(float)));
    sum += ((unsigned long long int)(f->n_kernels*4*sizeof(float)));
    sum += ((unsigned long long int)(f->n_kernels*f->rows1*f->cols1*6*sizeof(float)));
    sum += ((unsigned long long int)(f->n_kernels*f->rows2*f->cols2*sizeof(float)));
    sum += ((unsigned long long int)(f->channels*f->input_rows*f->input_cols*sizeof(float)));
    return sum;
}


/* this function compute the space allocated by the arrays of f
 * 
 * Input:
 * 
 *             rl* f:= the residual layer f
 * 
 * */
unsigned long long int size_of_rls(rl* f){
    unsigned long long int i,sum = 0;
    for(i = 0; i < f->n_cl; i++){
        sum+= size_of_cls(f->cls[i]);
    }
    
    sum+= ((unsigned long long int)(f->channels*f->input_cols*f->input_rows*sizeof(float)));
    sum+= size_of_cls(f->cl_output);
    return sum;
    
}


/* this function compute the space allocated by the arrays of f
 * 
 * Input:
 * 
 *             bn* b:= the batch normalized layer b
 * 
 * */
unsigned long long int size_of_bn(bn* b){
    unsigned long long int sum = 0;
    sum+= (b->batch_size*b->vector_dim*5);
    sum+= (b->vector_dim*13);
    return sum;
}


/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_fcl(fcl* f,fcl* copy){
    if(f == NULL)
        return;
    copy_array(f->weights,copy->weights,f->output*f->input);
    copy_array(f->d_weights,copy->d_weights,f->output*f->input);
    copy_array(f->d1_weights,copy->d1_weights,f->output*f->input);
    copy_array(f->d2_weights,copy->d2_weights,f->output*f->input);
    copy_array(f->biases,copy->biases,f->output);
    copy_array(f->d_biases,copy->d_biases,f->output);
    copy_array(f->d1_biases,copy->d1_biases,f->output);
    copy_array(f->d2_biases,copy->d2_biases,f->output);
    return;
}

/* This function returns a fcl* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 *                @ float tau:= the tau param
 * */
void slow_paste_fcl(fcl* f,fcl* copy, float tau){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->output*f->input; i++){
        if(i < f->output)
            copy->biases[i] = tau*f->biases[i] + (1-tau)*copy->biases[i];
        copy->weights[i] = tau*f->weights[i] + (1-tau)*copy->weights[i];
    }
    return;
}

/* This function returns a cl* layer that is the same copy of the input f
 * except for the activation arrays , the post normalization and post polling arrays
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ cl* f:= the convolutional layer that must be copied
 *             @ cl* copy:= the convolutional layer where f is copied
 * 
 * */
void paste_cl(cl* f, cl* copy){
    if(f == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_kernels; i++){
        copy_array(f->kernels[i],copy->kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d_kernels[i],copy->d_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d1_kernels[i],copy->d1_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d2_kernels[i],copy->d2_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
    }
    
    copy_array(f->biases,copy->biases,f->n_kernels);
    copy_array(f->d_biases,copy->d_biases,f->n_kernels);
    copy_array(f->d1_biases,copy->d1_biases,f->n_kernels);
    copy_array(f->d2_biases,copy->d2_biases,f->n_kernels);
    
    return;
}

/* This function returns a cl* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ cl* f:= the convolutional layer that must be copied
 *             @ cl* copy:= the convolutional layer where f is copied
 *                @ float tau:= the tau param
 * */
void slow_paste_cl(cl* f, cl* copy,float tau){
    if(f == NULL)
        return;
    
    int i,j;
    for(i = 0; i < f->n_kernels; i++){
        for(j = 0; j < f->channels*f->kernel_rows*f->kernel_cols; j++){
            copy->kernels[i][j] = tau*f->kernels[i][j] + (1-tau)*copy->kernels[i][j];
        }
        
        copy->biases[i] = tau*f->biases[i] + (1-tau)*copy->biases[i];
    }
    
    return;
}


/* This function returns a bn* layer that is the same copy of the input b1
 * except for temp arrays used for feed forward and backprop 
 * Input:
 * 
 *             @ bn* b1:= the batch normalized layer that must be copied
 *             @ bn* b2:= the batch normalized layer where b1 is copied
 * 
 * */
void paste_bn(bn* b1, bn* b2){
    if(b1 == NULL || b2 == NULL)
        return;
    
    copy_array(b1->gamma,b2->gamma,b1->vector_dim);
    copy_array(b1->d_gamma,b2->d_gamma,b1->vector_dim);
    copy_array(b1->d1_gamma,b2->d1_gamma,b1->vector_dim);
    copy_array(b1->d2_gamma,b2->d2_gamma,b1->vector_dim);
    copy_array(b1->beta,b2->beta,b1->vector_dim);
    copy_array(b1->d_beta,b2->d_beta,b1->vector_dim);
    copy_array(b1->d1_beta,b2->d1_beta,b1->vector_dim);
    copy_array(b1->d2_beta,b2->d2_beta,b1->vector_dim);
    copy_array(b1->final_mean,b2->final_mean,b1->vector_dim);
    copy_array(b1->final_var,b2->final_var,b1->vector_dim);
    
    return;
}

/* This function returns a bn* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ bn* f:= the batch normalized layer that must be copied
 *             @ bn* copy:= the batch normalized layer where f is copied
 *             @ float tau:= the tau param
 * */
void slow_paste_bn(bn* f, bn* copy,float tau){
    if(f == NULL)
        return;
    
    int i,j;
    for(i = 0; i < f->vector_dim; i++){
        copy->gamma[i] = tau*f->gamma[i] + (1-tau)*copy->gamma[i];
        copy->beta[i] = tau*f->beta[i] + (1-tau)*copy->beta[i];
    }
    
    return;
}


/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 *             @ rl* copy:= the residual layer where f is copied
 * 
 * */
void paste_rl(rl* f, rl* copy){
    if(f == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        paste_cl(f->cls[i],copy->cls[i]);
    }
    
    return;
}

/* This function returns a rl* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 *             @ rl* copy:= the residual layer where f is copied
 *                @ float tau:= the tau param
 * */
void slow_paste_rl(rl* f, rl* copy,float tau){
    if(f == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        slow_paste_cl(f->cls[i],copy->cls[i],tau);
    }
    
    return;
}

