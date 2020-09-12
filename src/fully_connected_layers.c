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

/* This function builds a fully-connected layer according to the fcl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int input:= number of neurons of the previous layer
 *             @ int output:= number of neurons of the current layer
 *             @ int layer:= number of sequential layer [0,∞)
 *             @ int dropout_flag:= is set to 0 if you don't want to apply dropout, NO_DROPOUT (flag)
 *             @ int activation_flag:= is set to 0 if you don't want to apply the activation function else read in llab.h
 *             @ float dropout_threshold:= [0,1]
 *                @ int n_groups:= a number that divides the output in tot group for the layer normalization
 *                @ int normalization_flag:= either NO_NORMALIZATION or LAYER_NORMALIZATION 
 * */
fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag){
    if(!input || !output || layer < 0){
        fprintf(stderr,"Error: input, output params must be > 0 and layer > -1\n");
        exit(1);
    }
    
    if(normalization_flag == GROUP_NORMALIZATION)
        normalization_flag = LAYER_NORMALIZATION;
    
    if(normalization_flag == LAYER_NORMALIZATION){
        if(n_groups == 0 || output==n_groups){
            fprintf(stderr,"Error: your groups must perfectly divide your output neurons\n");
            exit(1);
        }
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
    f->ex_d_weights_diff_grad = (float*)calloc(output*input,sizeof(float));
    f->d1_weights = (float*)calloc(output*input,sizeof(float));
    f->d2_weights = (float*)calloc(output*input,sizeof(float));
    f->d3_weights = (float*)calloc(output*input,sizeof(float));
    f->biases = (float*)calloc(output,sizeof(float));
    f->d_biases = (float*)calloc(output,sizeof(float));
    f->ex_d_biases_diff_grad = (float*)calloc(output,sizeof(float));
    f->d1_biases = (float*)calloc(output,sizeof(float));
    f->d2_biases = (float*)calloc(output,sizeof(float));
    f->d3_biases = (float*)calloc(output,sizeof(float));
    f->pre_activation = (float*)calloc(output,sizeof(float));
    f->dropout_temp = (float*)calloc(output,sizeof(float));
    f->temp = (float*)calloc(output,sizeof(float));
    f->temp3 = (float*)calloc(output,sizeof(float));
    f->temp2 = (float*)calloc(input,sizeof(float));
    f->error2 = (float*)calloc(input,sizeof(float));
    f->training_mode = GRADIENT_DESCENT;
    f->scores = (float*)calloc(output*input,sizeof(float));
    f->d_scores = (float*)calloc(output*input,sizeof(float));
    f->ex_d_scores_diff_grad = (float*)calloc(output*input,sizeof(float));
    f->d1_scores = (float*)calloc(output*input,sizeof(float));
    f->d2_scores = (float*)calloc(output*input,sizeof(float));
    f->d3_scores = (float*)calloc(output*input,sizeof(float));
    f->indices = (int*)calloc(output*input,sizeof(int));
    f->active_output_neurons = (int*)calloc(output,sizeof(int));
    f->post_activation = (float*)calloc(output,sizeof(float));
    f->post_normalization = (float*)calloc(output,sizeof(float));
    f->dropout_mask = (float*)calloc(output,sizeof(float));
    f->feed_forward_flag = FULLY_FEED_FORWARD;
    f->k_percentage = 1;
    for(i = 0; i < output; i++){
        f->active_output_neurons[i] = 1;
        for(j = 0; j < input; j++){
            f->indices[i*input+j] = i*input+j;
            f->weights[i*input+j] = random_general_gaussian(0, (float)input);
        }
        if(dropout_flag)
            f->dropout_mask[i] = 1;
    }
    
    if(normalization_flag == LAYER_NORMALIZATION){
        f->layer_norm = batch_normalization(n_groups,output/n_groups,0,0);
    }
    else{
        f->layer_norm = NULL;
    }
    
    f->normalization_flag = normalization_flag;
    f->n_groups = n_groups;
    
    return f;
}

/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected(fcl* f){
    if(f == NULL){
        return;
    }
    
    free(f->weights);
    free(f->d_weights);
    free(f->ex_d_weights_diff_grad);
    free(f->d1_weights);
    free(f->d2_weights);
    free(f->d3_weights);
    free(f->biases);
    free(f->d_biases);
    free(f->ex_d_biases_diff_grad);
    free(f->d1_biases);
    free(f->d2_biases);
    free(f->d3_biases);
    free(f->pre_activation);
    free(f->post_activation);
    free(f->post_normalization);
    free(f->dropout_mask);
    free(f->dropout_temp);
    free(f->temp);
    free(f->temp2);
    free(f->temp3);
    free(f->error2);
    free(f->scores);
    free(f->d_scores);
    free(f->ex_d_scores_diff_grad);
    free(f->d1_scores);
    free(f->d2_scores);
    free(f->d3_scores);
    free(f->indices);
    free(f->active_output_neurons);
    free_batch_normalization(f->layer_norm);
    free(f);    
}

/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected_for_edge_popup(fcl* f){
    if(f == NULL){
        return;
    }
    
    free(f->d_weights);
    free(f->ex_d_weights_diff_grad);
    free(f->d1_weights);
    free(f->d2_weights);
    free(f->d3_weights);
    free(f->d_biases);
    free(f->ex_d_biases_diff_grad);
    free(f->d1_biases);
    free(f->d2_biases);
    free(f->d3_biases);
}

/* Given a fcl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ fcl* f:= the fcl structure that must be deallocated
 * */
void free_fully_connected_complementary_edge_popup(fcl* f){
    if(f == NULL){
        return;
    }
    free(f->weights);
    free(f->biases);
    free(f->pre_activation);
    free(f->post_activation);
    free(f->post_normalization);
    free(f->dropout_mask);
    free(f->dropout_temp);
    free(f->temp);
    free(f->temp2);
    free(f->temp3);
    free(f->error2);
    free(f->scores);
    free(f->d_scores);
    free(f->ex_d_scores_diff_grad);
    free(f->d1_scores);
    free(f->d2_scores);
    free(f->d3_scores);
    free(f->indices);
    free(f->active_output_neurons);
    free_batch_normalization(f->layer_norm);
    free(f);
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
    
    i = fwrite(&f->n_groups,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->normalization_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->feed_forward_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->training_mode,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
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
    
    i = fwrite(f->scores,sizeof(float)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->indices,sizeof(int)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->active_output_neurons,sizeof(int)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        save_bn(f->layer_norm,n);
    
    free(s);
    
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
void heavy_save_fcl(fcl* f, int n){
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
    
    
    i = fwrite(&f->n_groups,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->normalization_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->feed_forward_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->training_mode,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
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
    
    i = fwrite(f->d1_weights,sizeof(float)*(f->input)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d2_weights,sizeof(float)*(f->input)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d3_weights,sizeof(float)*(f->input)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->ex_d_weights_diff_grad,sizeof(float)*(f->input)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->biases,sizeof(float)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d1_biases,sizeof(float)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d2_biases,sizeof(float)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d3_biases,sizeof(float)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->ex_d_biases_diff_grad,sizeof(float)*(f->output),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->scores,sizeof(float)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d1_scores,sizeof(float)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d2_scores,sizeof(float)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->d3_scores,sizeof(float)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->ex_d_scores_diff_grad,sizeof(float)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->indices,sizeof(int)*(f->output)*(f->input),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    
    i = fwrite(f->active_output_neurons,sizeof(int)*(f->output),1,fw);
    
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
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        heavy_save_bn(f->layer_norm,n);
    
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
    copy_array(weights,f->weights,f->input*f->output);
    copy_array(biases,f->biases,f->output);
}

/* This function loads a fully-connected layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
fcl* heavy_load_fcl(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int input = 0,output = 0,layer = 0,dropout_flag = 0,activation_flag = 0, training_mode = 0,feed_forward_flag = 0, normalization_flag = 0, n_groups = 0;
    float dropout_threshold = 0;
    float* weights;
    float* d1_weights;
    float* d2_weights;
    float* d3_weights;
    float* ex_d_weights_diff_grad;
    float* biases;
    float* d1_biases;
    float* d2_biases;
    float* d3_biases;
    float* ex_d_biases_diff_grad;
    float* scores;
    float* d1_scores;
    float* d2_scores;
    float* d3_scores;
    float* ex_d_scores_diff_grad;
    int* indices;
    int* active_output_neurons;
    bn* layer_norm = NULL;
    i = fread(&n_groups,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&normalization_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    
    i = fread(&feed_forward_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&training_mode,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
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
    d1_weights = (float*)malloc(sizeof(float)*input*output);
    d2_weights = (float*)malloc(sizeof(float)*input*output);
    d3_weights = (float*)malloc(sizeof(float)*input*output);
    ex_d_weights_diff_grad = (float*)malloc(sizeof(float)*input*output);
    scores = (float*)malloc(sizeof(float)*input*output);
    d1_scores = (float*)malloc(sizeof(float)*input*output);
    d2_scores = (float*)malloc(sizeof(float)*input*output);
    d3_scores = (float*)malloc(sizeof(float)*input*output);
    ex_d_scores_diff_grad = (float*)malloc(sizeof(float)*input*output);
    indices = (int*)malloc(sizeof(int)*input*output);
    active_output_neurons = (int*)malloc(sizeof(int)*output);
    biases = (float*)malloc(sizeof(float)*output);
    d1_biases = (float*)malloc(sizeof(float)*output);
    d2_biases = (float*)malloc(sizeof(float)*output);
    d3_biases = (float*)malloc(sizeof(float)*output);
    ex_d_biases_diff_grad = (float*)malloc(sizeof(float)*output);
    
    i = fread(weights,sizeof(float)*(input)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d1_weights,sizeof(float)*(input)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d2_weights,sizeof(float)*(input)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d3_weights,sizeof(float)*(input)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(ex_d_weights_diff_grad,sizeof(float)*(input)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    i = fread(biases,sizeof(float)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d1_biases,sizeof(float)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d2_biases,sizeof(float)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d3_biases,sizeof(float)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(ex_d_biases_diff_grad,sizeof(float)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    i = fread(scores,sizeof(float)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d1_scores,sizeof(float)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d2_scores,sizeof(float)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(d3_scores,sizeof(float)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(ex_d_scores_diff_grad,sizeof(float)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    i = fread(indices,sizeof(int)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(active_output_neurons,sizeof(int)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    if(normalization_flag == LAYER_NORMALIZATION)
        layer_norm = heavy_load_bn(fr);
    fcl* f = fully_connected(input,output,layer,dropout_flag,activation_flag,dropout_threshold, n_groups, normalization_flag);
    copy_fcl_params(f,weights,biases);
    copy_array(scores,f->scores,input*output);
    copy_array(d1_weights,f->d1_weights,input*output);
    copy_array(d2_weights,f->d2_weights,input*output);
    copy_array(d3_weights,f->d3_weights,input*output);
    copy_array(ex_d_weights_diff_grad,f->ex_d_weights_diff_grad,input*output);
    copy_array(d1_biases,f->d1_biases,output);
    copy_array(d2_biases,f->d2_biases,output);
    copy_array(d3_biases,f->d3_biases,output);
    copy_array(ex_d_biases_diff_grad,f->ex_d_biases_diff_grad,output);
    copy_array(d1_scores,f->d1_scores,input*output);
    copy_array(d2_scores,f->d2_scores,input*output);
    copy_array(d3_scores,f->d3_scores,input*output);
    copy_array(ex_d_scores_diff_grad,f->ex_d_scores_diff_grad,input*output);
    copy_int_array(indices,f->indices,input*output);
    copy_int_array(active_output_neurons,f->active_output_neurons,output);
    f->training_mode = training_mode;
    f->feed_forward_flag = feed_forward_flag;
    free(weights);
    free(d1_weights);
    free(d2_weights);
    free(d3_weights);
    free(ex_d_weights_diff_grad);
    free(biases);
    free(d1_biases);
    free(d2_biases);
    free(d3_biases);
    free(ex_d_biases_diff_grad);
    free(indices);
    free(d1_scores);
    free(d2_scores);
    free(d3_scores);
    free(ex_d_scores_diff_grad);
    free(active_output_neurons);
    free(scores);
    if(layer_norm != NULL)
        paste_bn(layer_norm,f->layer_norm);
    free_batch_normalization(layer_norm);
    return f;
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
    
    int input = 0,output = 0,layer = 0,dropout_flag = 0,activation_flag = 0, training_mode = 0,feed_forward_flag = 0, n_groups = 0, normalization_flag = 0;
    float dropout_threshold = 0;
    float* weights;
    float* biases;
    float* scores;
    float* ex_d_scores_diff_grad;
    int* indices;
    int* active_output_neurons;
    bn* layer_norm = NULL;
    
    i = fread(&n_groups,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&normalization_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&feed_forward_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&training_mode,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
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
    scores = (float*)malloc(sizeof(float)*input*output);
    indices = (int*)malloc(sizeof(int)*input*output);
    active_output_neurons = (int*)malloc(sizeof(int)*output);
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
    
    i = fread(scores,sizeof(float)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(indices,sizeof(int)*(output)*(input),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(active_output_neurons,sizeof(int)*(output),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    fcl* f = fully_connected(input,output,layer,dropout_flag,activation_flag,dropout_threshold, n_groups, normalization_flag);
    copy_fcl_params(f,weights,biases);
    copy_array(scores,f->scores,input*output);
    copy_int_array(indices,f->indices,input*output);
    copy_int_array(active_output_neurons,f->active_output_neurons,output);
    f->training_mode = training_mode;
    f->feed_forward_flag = feed_forward_flag;
    free(weights);
    free(biases);
    free(indices);
    free(active_output_neurons);
    free(scores);
    
    if(normalization_flag == LAYER_NORMALIZATION){
        layer_norm = load_bn(fr);
        paste_bn(layer_norm,f->layer_norm);
    }
    free_batch_normalization(layer_norm);
    return f;
}

/* This function loads a fully-connected layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
fcl* light_load_fcl(FILE* fr){
    fcl* f = load_fcl(fr);
    free_fully_connected_for_edge_popup(f);
    return f;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array, and all the arrays used by ff and bp.
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
    fcl* copy = fully_connected(f->input, f->output,f->layer, f->dropout_flag,f->activation_flag,f->dropout_threshold,f->n_groups,f->normalization_flag);
    copy_array(f->weights,copy->weights,f->output*f->input);
    copy_array(f->d_weights,copy->d_weights,f->output*f->input);
    copy_array(f->ex_d_weights_diff_grad,copy->ex_d_weights_diff_grad,f->output*f->input);
    copy_array(f->d1_weights,copy->d1_weights,f->output*f->input);
    copy_array(f->d2_weights,copy->d2_weights,f->output*f->input);
    copy_array(f->d3_weights,copy->d3_weights,f->output*f->input);
    copy_array(f->biases,copy->biases,f->output);
    copy_array(f->d_biases,copy->d_biases,f->output);
    copy_array(f->ex_d_biases_diff_grad,copy->ex_d_biases_diff_grad,f->output);
    copy_array(f->d1_biases,copy->d1_biases,f->output);
    copy_array(f->d2_biases,copy->d2_biases,f->output);
    copy_array(f->d3_biases,copy->d3_biases,f->output);
    copy_array(f->scores,copy->scores,f->input*f->output);
    copy_array(f->d_scores,copy->d_scores,f->input*f->output);
    copy_array(f->ex_d_scores_diff_grad,copy->ex_d_scores_diff_grad,f->input*f->output);
    copy_array(f->d1_scores,copy->d1_scores,f->input*f->output);
    copy_array(f->d2_scores,copy->d2_scores,f->input*f->output);
    copy_array(f->d3_scores,copy->d3_scores,f->input*f->output);
    copy_int_array(f->indices,copy->indices,f->input*f->output);
    copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
    copy->training_mode = f->training_mode;
    copy->feed_forward_flag = f->feed_forward_flag;
    if(f->normalization_flag == LAYER_NORMALIZATION){
        paste_bn(f->layer_norm,copy->layer_norm);
    }
    return copy;
}


/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array, and all the arrays used by ff and bp.
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
fcl* copy_light_fcl(fcl* f){
    fcl* copy = copy_fcl(f);
    free_fully_connected_for_edge_popup(copy);
    return copy;
}

/* this function resets all the arrays of a fully-connected layer
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
            f->post_normalization[i] = 0;
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
        
        if(f->training_mode == EDGE_POPUP){
            f->indices[i] = i;
            f->d_scores[i] = 0;
        }
    }
    if(f->training_mode == EDGE_POPUP){
        quick_sort(f->scores,f->indices,0,f->output*f->input-1);
        free(f->active_output_neurons);
        f->active_output_neurons = get_used_outputs(f,NULL,FCLS,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}

/* this function resets all the arrays of a fully-connected layer
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
fcl* reset_fcl_without_dwdb(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    for(i = 0; i < f->output*f->input; i++){
        if(i < f->output){
            f->pre_activation[i] = 0;
            f->post_activation[i] = 0;
            f->post_normalization[i] = 0;
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
        
        if(f->training_mode == EDGE_POPUP){
            f->indices[i] = i;
            f->d_scores[i] = 0;
        }
    }
    if(f->training_mode == EDGE_POPUP){
        quick_sort(f->scores,f->indices,0,f->output*f->input-1);
        free(f->active_output_neurons);
        f->active_output_neurons = get_used_outputs(f,NULL,FCLS,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}
/* this function resets all the arrays of a fully-connected layer
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
fcl* reset_fcl_for_edge_popup(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    for(i = 0; i < f->output*f->input; i++){
        if(i < f->output){
            f->pre_activation[i] = 0;
            f->post_activation[i] = 0;
            f->post_normalization[i] = 0;
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
        
        if(f->training_mode == EDGE_POPUP){
            f->d_scores[i] = 0;
        }
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}

/* this function resets all the arrays of a fully-connected layer
 * used during the feed forward and backpropagation, doesn't care about partial derivatives of weights and biases
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
fcl* light_reset_fcl(fcl* f){
    if(f == NULL)
        return NULL;
    int i;
    for(i = 0; i < f->output*f->input; i++){
        if(i < f->output){
            f->pre_activation[i] = 0;
            f->post_activation[i] = 0;
            f->post_normalization[i] = 0;
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
        
        if(f->training_mode == EDGE_POPUP){
            f->indices[i] = i;
            f->d_scores[i] = 0;
        }
    }
    if(f->training_mode == EDGE_POPUP){
        quick_sort(f->scores,f->indices,0,f->output*f->input-1);
        free(f->active_output_neurons);
        f->active_output_neurons = get_used_outputs(f,NULL,FCLS,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        reset_bn(f->layer_norm);
    return f;
}
/* this function returns the space allocated by the arrays of f (more or less)
 * 
 * Input:
 * 
 *             fcl* f:= the fully-connected layer f
 * 
 * */
unsigned long long int size_of_fcls(fcl* f){
    unsigned long long int sum = 0;
    sum += ((unsigned long long int)(f->input*f->output*13*sizeof(float)));
    sum += ((unsigned long long int)(f->input*f->output*sizeof(int)));
    sum += ((unsigned long long int)(f->output*12*sizeof(float)));
    sum += ((unsigned long long int)(f->input*2*sizeof(float)));
    if(f->normalization_flag == LAYER_NORMALIZATION)
        sum += size_of_bn(f->layer_norm);
    return sum;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * the edge popup params are pasted only if feedforwardflag or training mode is set to edge popup
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_fcl(fcl* f,fcl* copy){
    if(f == NULL)
        return;
    copy->k_percentage = f->k_percentage;
    copy_array(f->weights,copy->weights,f->output*f->input);
    copy_array(f->d_weights,copy->d_weights,f->output*f->input);
    copy_array(f->ex_d_weights_diff_grad,copy->ex_d_weights_diff_grad,f->output*f->input);
    copy_array(f->d1_weights,copy->d1_weights,f->output*f->input);
    copy_array(f->d2_weights,copy->d2_weights,f->output*f->input);
    copy_array(f->d3_weights,copy->d3_weights,f->output*f->input);
    copy_array(f->biases,copy->biases,f->output);
    copy_array(f->d_biases,copy->d_biases,f->output);
    copy_array(f->ex_d_biases_diff_grad,copy->ex_d_biases_diff_grad,f->output);
    copy_array(f->d1_biases,copy->d1_biases,f->output);
    copy_array(f->d2_biases,copy->d2_biases,f->output);
    copy_array(f->d3_biases,copy->d3_biases,f->output);
    if(f->training_mode == EDGE_POPUP || f->feed_forward_flag == EDGE_POPUP){
        copy_array(f->scores,copy->scores,f->input*f->output);
        copy_array(f->d_scores,copy->d_scores,f->input*f->output);
        copy_array(f->ex_d_scores_diff_grad,copy->ex_d_scores_diff_grad,f->input*f->output);
        copy_array(f->d1_scores,copy->d1_scores,f->input*f->output);
        copy_array(f->d2_scores,copy->d2_scores,f->input*f->output);
        copy_array(f->d3_scores,copy->d3_scores,f->input*f->output);
        copy_int_array(f->indices,copy->indices,f->input*f->output);
        copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION){
        paste_bn(f->layer_norm,copy->layer_norm);
    }
    return;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * the edge popup params are pasted only if feedforwardflag or training mode is set to edge popup
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_fcl_for_edge_popup(fcl* f,fcl* copy){
    if(f == NULL)
        return;
    
    if(f->training_mode == EDGE_POPUP || f->feed_forward_flag == EDGE_POPUP){
        copy_array(f->scores,copy->scores,f->input*f->output);
        copy_array(f->d_scores,copy->d_scores,f->input*f->output);
        copy_array(f->ex_d_scores_diff_grad,copy->ex_d_scores_diff_grad,f->input*f->output);
        copy_array(f->d1_scores,copy->d1_scores,f->input*f->output);
        copy_array(f->d2_scores,copy->d2_scores,f->input*f->output);
        copy_array(f->d3_scores,copy->d3_scores,f->input*f->output);
        copy_int_array(f->indices,copy->indices,f->input*f->output);
        copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION){
        paste_bn(f->layer_norm,copy->layer_norm);
    }
    return;
}

/* This function returns a fcl* layer that is the same copy of the input f
 * except for the activation arrays and the dropout mask array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * the edge popup params are pasted only if feedforwardflag or training mode is set to edge popup
 * Input:
 * 
 *             @ fcl* f:= the fully-connected layer that must be copied
 *             @ fcl* copy:= the fully-connected layer where f is copied
 * 
 * */
void paste_w_fcl(fcl* f,fcl* copy){
    if(f == NULL)
        return;
    copy_array(f->weights,copy->weights,f->output*f->input);
    copy_array(f->biases,copy->biases,f->output);
    if(f->training_mode == EDGE_POPUP || f->feed_forward_flag == EDGE_POPUP){
        copy_array(f->scores,copy->scores,f->input*f->output);
        copy_int_array(f->indices,copy->indices,f->input*f->output);
        copy_int_array(f->active_output_neurons,copy->active_output_neurons,f->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        paste_w_bn(f->layer_norm,copy->layer_norm);
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
        if(i < f->output){
            copy->biases[i] = tau*f->biases[i] + (1-tau)*copy->biases[i];
            copy->d1_biases[i] = tau*f->d1_biases[i] + (1-tau)*copy->d1_biases[i];
            copy->d2_biases[i] = tau*f->d2_biases[i] + (1-tau)*copy->d2_biases[i];
            copy->d3_biases[i] = tau*f->d3_biases[i] + (1-tau)*copy->d3_biases[i];
            copy->ex_d_biases_diff_grad[i] = tau*f->ex_d_biases_diff_grad[i] + (1-tau)*copy->ex_d_biases_diff_grad[i];
        }
        copy->weights[i] = tau*f->weights[i] + (1-tau)*copy->weights[i];
        copy->d1_weights[i] = tau*f->d1_weights[i] + (1-tau)*copy->d1_weights[i];
        copy->d2_weights[i] = tau*f->d2_weights[i] + (1-tau)*copy->d2_weights[i];
        copy->d3_weights[i] = tau*f->d3_weights[i] + (1-tau)*copy->d3_weights[i];
        copy->ex_d_weights_diff_grad[i] = tau*f->ex_d_weights_diff_grad[i] + (1-tau)*copy->ex_d_weights_diff_grad[i];
        if(f->training_mode == EDGE_POPUP || f->feed_forward_flag == EDGE_POPUP){
            copy->scores[i] = tau*f->scores[i] + (1-tau)*copy->scores[i];
            copy->d1_scores[i] = tau*f->d1_scores[i] + (1-tau)*copy->d1_scores[i];
            copy->d2_scores[i] = tau*f->d2_scores[i] + (1-tau)*copy->d2_scores[i];
            copy->d3_scores[i] = tau*f->d3_scores[i] + (1-tau)*copy->d3_scores[i];
            copy->ex_d_scores_diff_grad[i] = tau*f->ex_d_scores_diff_grad[i] + (1-tau)*copy->ex_d_scores_diff_grad[i];
            copy->indices[i] = i;
        }
        
    }
    
    for(i = 0; i < f->input*f->output; i++){
        copy->indices[i] = i;
    }
    if(f->training_mode == EDGE_POPUP || f->feed_forward_flag == EDGE_POPUP){
        quick_sort(copy->scores,copy->indices,0,f->output*f->input-1);
        free(copy->active_output_neurons);
        copy->active_output_neurons = get_used_outputs(copy,NULL,FCLS,copy->output);
    }
    
    if(f->normalization_flag == LAYER_NORMALIZATION)
        slow_paste_bn(f->layer_norm,copy->layer_norm,tau);
    return;
}


/* this function gives the number of float params for biases and weights in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
int get_array_size_params(fcl* f){
    int sum = 0;
    if(f->normalization_flag == LAYER_NORMALIZATION){
        sum += f->layer_norm->vector_dim*2;
    }
    return f->input*f->output+f->output+sum;
}


/* this function gives the number of float params for biases and weights in a fcl
 * 
 * Input:
 * 
 * 
 *                 @ flc* f:= the fully-connected layer
 * */
int get_array_size_weights(fcl* f){
    return f->input*f->output;
}
/* this function pastes the weights and biases from a vector into in a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_params(fcl* f, float* vector){
    memcpy(f->weights,vector,f->input*f->output*sizeof(float));
    memcpy(f->biases,&vector[f->input*f->output],f->output*sizeof(float));
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(f->layer_norm->gamma,&vector[f->input*f->output+f->output],f->layer_norm->vector_dim*sizeof(float));
        memcpy(f->layer_norm->beta,&vector[f->input*f->output+f->output + f->layer_norm->vector_dim],f->layer_norm->vector_dim*sizeof(float));
    }
}

/* this function pastes the weights and biases from a vector into in a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_scores(fcl* f, float* vector){
    memcpy(f->scores,vector,f->input*f->output*sizeof(float));
}

/* this function pastes the the weights and biases from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_params_to_vector(fcl* f, float* vector){
    memcpy(vector,f->weights,f->input*f->output*sizeof(float));
    memcpy(&vector[f->input*f->output],f->biases,f->output*sizeof(float));
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(&vector[f->input*f->output+f->output],f->layer_norm->gamma,f->layer_norm->vector_dim*sizeof(float));
        memcpy(&vector[f->input*f->output+f->output + f->layer_norm->vector_dim],f->layer_norm->beta,f->layer_norm->vector_dim*sizeof(float));
    }
}

/* this function pastes the the weights from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_weights_to_vector(fcl* f, float* vector){
    memcpy(vector,f->weights,f->input*f->output*sizeof(float));
}

/* this function pastes the the weights from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_weights(fcl* f, float* vector){
    memcpy(f->weights,vector,f->input*f->output*sizeof(float));
}

/* this function pastes the the weights and biases from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_scores_to_vector(fcl* f, float* vector){
    memcpy(vector,f->scores,f->input*f->output*sizeof(float));
}
/* this function pastes the dweights and dbiases from a vector into in a fcl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_derivative_params(fcl* f, float* vector){
    memcpy(f->d_weights,vector,f->input*f->output*sizeof(float));
    memcpy(f->d_biases,&vector[f->input*f->output],f->output*sizeof(float));
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(f->layer_norm->d_gamma,&vector[f->input*f->output+f->output],f->layer_norm->vector_dim*sizeof(float));
        memcpy(f->layer_norm->d_beta,&vector[f->input*f->output+f->output + f->layer_norm->vector_dim],f->layer_norm->vector_dim*sizeof(float));
    }
}


/* this function pastes the the dweights and dbiases from a fcl structure into a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ fcl* f:= the fully-connecteed layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_derivative_params_to_vector(fcl* f, float* vector){
    memcpy(vector,f->d_weights,f->input*f->output*sizeof(float));
    memcpy(&vector[f->input*f->output],f->d_biases,f->output*sizeof(float));
    if(f->normalization_flag == LAYER_NORMALIZATION){
        memcpy(&vector[f->input*f->output+f->output],f->layer_norm->d_gamma,f->layer_norm->vector_dim*sizeof(float));
        memcpy(&vector[f->input*f->output+f->output + f->layer_norm->vector_dim],f->layer_norm->d_beta,f->layer_norm->vector_dim*sizeof(float));
    }
}

/* setting the biases to 0
 * Inpout:
 *             @ fcl* f:= the fully connected layer
 * */
void set_fully_connected_biases_to_zero(fcl* f){
    int i;
    for(i = 0; i < f->output; i++){
        f->biases[i] = 0;
    }
}

/* setting the unused weights to 0
 * Inpout:
 *             @ fcl* f:= the fully connected layer
 * */
void set_fully_connected_unused_weights_to_zero(fcl* f){
    int i;
    for(i = 0; i < f->output*f->input-f->output*f->input*f->k_percentage; i++){
        f->weights[f->indices[i]] = 0;
    }
}

/* This function minimizes k percentage and the used weights if there are some used weights attached to inactive neurons
 * 
 * Input:
 * 
 *             @ fcl* f:= the fully connected layer
 *             @ int* used_input:= an array of the input used, can be dimension (n_feature_maps) if layer_flag = CLS f->input*f->output dimension otherwise
 *             @ int* used_output:= the active neurons of the next layer, f->output dimension
 *             @ int layer_flag:= it says if the previous layer was a convolution/residual (CONVOLUTION) or a fcl layer (FULLY_CONNECTED)
 *             @ int input_size:= the size of used_input array
 * */
int fcl_adjusting_weights_after_edge_popup(fcl* f, int* used_input, int* used_output, int layer_flag, int input_size){
    int i,j,z,flag = 0, lower = f->input*f->output-f->input*f->output*f->k_percentage;
    for(i = f->input*f->output-f->input*f->output*f->k_percentage; i < f->input*f->output; i++){
        if(layer_flag == CLS){
            int n_per_feature_map = f->input/input_size;
            for(j = 0; j < input_size; j++){
                if(((int)(f->indices[i]%f->input)< n_per_feature_map*(j+1) && (int)(f->indices[i]%f->input) >= n_per_feature_map*(j)) || (!used_output[(int)(f->indices[i]%f->output)])){
                    if(!used_input[j]){
                        flag = 1;
                        for(z = i; z > lower; z--){
                            int temp = f->indices[z-1];
                            f->indices[z-1] = f->indices[z];
                            f->indices[z] = temp;
                        }
                        lower++;
                    }
                }
            }
            
        }
        
        else{
            if((!used_input[(int)(f->indices[i]%f->input)]) || (!used_output[(int)(f->indices[i]%f->output)])){
                flag = 1;
                for(z = i; z > lower; z--){
                    int temp = f->indices[z-1];
                    f->indices[z-1] = f->indices[z];
                    f->indices[z] = temp;
                }
                lower++;
            }
        }
    }
    
    f->k_percentage = (float)(1-(float)(((double)(lower))/((double)(f->input*f->output))));
    return flag;
}


/* This function returns the input surely used by current weights
 * 
 * Inputs:
 *             
 *             @ fcl* f:= the fully-connected layer
 *             @ int* used_input:= the used input
 *             @ int flag == convolution or fully connected
 *             @ int input_size:= the size of the array used_input
 * */
int* get_used_inputs(fcl* f, int* used_input, int flag, int input_size){
    int i,j;
    int* ui;
    if(used_input == NULL)
        ui = (int*)calloc(input_size,sizeof(int));
    else
        ui = used_input;
    for(i = 0; i < input_size; i++){
        ui[i] = 0;
    }
    
    for(i = f->input*f->output-f->input*f->output*f->k_percentage; i < f->input*f->output; i++){
        if(flag == CLS){
            int n_per_feature_map = f->input/input_size;
            for(j = 0; j < input_size; j++){
                if(((int)(f->indices[i]%f->input)< n_per_feature_map*(j+1) && (int)(f->indices[i]%f->input) >= n_per_feature_map*(j)))
                ui[j] = 1;
            }
        }
            
        else{
            ui[f->indices[i]%f->input] = 1;
        }
    }
    
    return ui;    
}

/* This function returns the output surely used by current weights
 * 
 * Inputs:
 *             
 *             @ fcl* f:= the fully-connected layer
 *             @ int* used_input:= the used output
 *             @ int flag == convolution or fully connected
 *             @ int input_size:= the size of the array used_output
 * */
int* get_used_outputs(fcl* f, int* used_output, int flag, int output_size){
    int i,j;
    int* uo;
    if(used_output == NULL)
        uo= (int*)calloc(output_size,sizeof(int));
    else
        uo = used_output;
    
    for(i = 0; i < output_size; i++){
        uo[i] = 0;
    }
    
    for(i = f->input*f->output-f->input*f->output*f->k_percentage; i < f->input*f->output; i++){
        if(flag == CLS){
            int n_per_feature_map = f->output/output_size;
            for(j = 0; j < output_size; j++){
                if(((int)(f->indices[i]%f->output)< n_per_feature_map*(j+1) && (int)(f->indices[i]%f->output) >= n_per_feature_map*(j)))
                uo[j] = 1;
            }
        }
            
        else{
            uo[(int)((f->indices[i]/f->input))] = 1;
        }
    }
    
    return uo;  
    
}


/* this function sum up the scores in input1 and input2 in output
 * 
 * Input:
 * 
 * 
 *                 @ fcl* input1:= the first input fcl layer
 *                 @ fcl* input2:= the second input fcl layer
 *                 @ fcl* output:= the output fcl layer
 * */
void sum_score_fcl(fcl* input1, fcl* input2, fcl* output){
    sum1D(input1->scores,input2->scores,output->scores,input1->input*input1->output);
}

/* this function sum up the scores in input1 and input2 in output
 * 
 * Input:
 * 
 * 
 *                 @ fcl* input1:= the first input fcl layer
 *                 @ fcl* input2:= the second input fcl layer
 *                 @ fcl* output:= the output fcl layer
 * */
void compare_score_fcl(fcl* input1, fcl* input2, fcl* output){
    int i;
    for(i = 0; i < input1->input*input1->output; i++){
        if(input1->scores[i] > input2->scores[i])
            output->scores[i] = input1->scores[i];
        else
            output->scores[i] = input2->scores[i];
    }
}

/* this function divides the score with value
 * 
 * Input:
 * 
 *                 @ fcl* f:= the fcl layer
 *                 @ float value:= the value that is gonna divide the scores
 * */
void dividing_score_fcl(fcl* f, float value){
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->scores[i]/=value;
    }
}

/* this function set the feed forward flag to only dropout checking the restriction needed
 * 
 * Input:
 * 
 * 
 *                 @ fcl* f:= the fully connected layer
 * 
 * */
void set_fcl_only_dropout(fcl* f){
    if(!f->dropout_flag){
        fprintf(stderr,"Error: if you use this layer only for dropout you should set dropou flag!\n");
        exit(1);
    }
    
    if(f->input!= f->output){
        fprintf(stderr,"Error: if you use only dropout then your input and output should match!\n");
        exit(1);
    }
    
    f->feed_forward_flag = ONLY_DROPOUT;
}


/* this function reset all the scores of the fcl layer to 0
 * 
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 * */
void reset_score_fcl(fcl* f){
    if(f->feed_forward_flag == NO_DROPOUT)
        return;
    int i;
    for(i = 0; i < f->input*f->output; i++){
        f->scores[i] = 0;
    }
    
}

/* thif function reinitialize the weights under the goodness function only if
 * they are among the f->input*f->output*percentage worst weights according to the scores
 * percentage and goodness should range in [0,1]
 * 
 * Input:
 * 
 *                 @ fcl* f:= the fully connected layer
 *                 @ float percentage:= the percentage of the worst weights
 *                 @ float goodness:= the goodness function
 * */
void reinitialize_scores_fcl(fcl* f, float percentage, float goodness){
    if(f->feed_forward_flag == ONLY_DROPOUT)
        return;
    int i;
    for(i = 0; i < f->input*f->output; i++){
        if(i >= f->input*f->output*percentage)
            return;
        if(f->scores[f->indices[i]] < goodness)
            f->weights[f->indices[i]] = random_general_gaussian(0, (float)f->input);
    }
}
