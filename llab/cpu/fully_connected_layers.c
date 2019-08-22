/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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
