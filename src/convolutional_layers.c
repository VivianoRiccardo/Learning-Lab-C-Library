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
 *             @ int normalization_flag:= is set to 1 if you wan't to apply local response normalization, 0 for no normalization 3 for group normalization
 *             @ int activation_flag:= is set to 1 if you want to apply activation function
 *             @ int pooling_flag:= is set to 1 if you want to apply pooling
 *                @ int group_norm_channels:= the number of the grouped channels during the group normalization if there is anyone
 *                @ int convolutional_flag:= NO_CONVOLUTION to apply only pooling otherwise CONVOLUTION,TRANSPOSED CONVOLUTION
 *                   @ int training_mode:= can be FREEZE_TRAINING, ONLY_FF, GRADIENT_DESCENT, EDGE_POPUP
 *                   @ int feed_forward_flag:= can be FULLY_FEED_FORWARD, EDGE_POPUP
 *                @ int layer:= the layer index
 * 
 * */
 
cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer){
    if(!channels || !input_rows || !input_cols || !kernel_rows || !kernel_cols || !n_kernels || !stride1_rows || !stride1_cols || (pooling_flag && (!stride2_rows || !stride2_cols))){
        fprintf(stderr,"Error: channles, input_rows, input_cols, kernel_rows, kernel_cols, n_kernels, stride2_rows stride2_cols, stride2_rows, stride2_cols params must be > 0\n");
        exit(1);
    }
    
    if(padding1_rows!=padding1_cols || padding2_rows != padding2_cols){
        fprintf(stderr,"Error: padding1_rows must be equal to padding1_cols and padding2_rows must be equal to padding2_cols\n");
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
    
    if(convolutional_flag != NO_CONVOLUTION && normalization_flag == GROUP_NORMALIZATION){
        if(n_kernels%group_norm_channels){
            fprintf(stderr,"Error: your normalization channels doesn't divide perfectly the number of kernels of this layer\n");
            exit(1);
        }
    }
    
    if(convolutional_flag == NO_CONVOLUTION && normalization_flag){
        fprintf(stderr,"Error: you cannot use the convolutional layer only for normalization and pooling, you can use it for only pooling, or only convolution plus activation-normalization-pooling\n");
        exit(1);
    }
    
    
    
    int i,j;
    cl* c = (cl*)malloc(sizeof(cl));
    c->layer = layer;
    c->k_percentage = 1;
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
    if(convolutional_flag != NO_CONVOLUTION){
        c->kernels = (float**)malloc(sizeof(float*)*n_kernels);
        c->used_kernels = (int*)malloc(sizeof(int)*n_kernels);
    }
    else{
        c->used_kernels = NULL;
        c->kernels = NULL;
    }
    if(training_mode != EDGE_POPUP && training_mode != ONLY_FF && convolutional_flag != NO_CONVOLUTION){
        c->d_kernels = (float**)malloc(sizeof(float*)*n_kernels);
        c->ex_d_kernels_diff_grad = (float**)malloc(sizeof(float*)*n_kernels);
        c->d1_kernels = (float**)malloc(sizeof(float*)*n_kernels);
        c->d2_kernels = (float**)malloc(sizeof(float*)*n_kernels);
        c->d3_kernels = (float**)malloc(sizeof(float*)*n_kernels);
    }
    else{
        c->d_kernels = NULL;
        c->ex_d_kernels_diff_grad = NULL;
        c->d1_kernels = NULL;
        c->d2_kernels = NULL;
        c->d3_kernels = NULL;
    }
    if(convolutional_flag != NO_CONVOLUTION)
        c->biases = (float*)calloc(n_kernels,sizeof(float));
    else
        c->biases = NULL;
    if(training_mode != EDGE_POPUP && training_mode != ONLY_FF && convolutional_flag != NO_CONVOLUTION){
        c->d_biases = (float*)calloc(n_kernels,sizeof(float));
        c->ex_d_biases_diff_grad = (float*)calloc(n_kernels,sizeof(float));
        c->d1_biases = (float*)calloc(n_kernels,sizeof(float));
        c->d2_biases = (float*)calloc(n_kernels,sizeof(float));
        c->d3_biases = (float*)calloc(n_kernels,sizeof(float));
    }
    c->convolutional_flag = convolutional_flag;
    c->group_norm_channels = group_norm_channels;
    if(convolutional_flag == NO_CONVOLUTION)
        c->pooltemp = (float*)calloc(channels*input_rows*input_cols,sizeof(float));
    else
        c->pooltemp = NULL;
    c->training_mode = training_mode;
    c->feed_forward_flag = feed_forward_flag;
    
    if(convolutional_flag != NO_CONVOLUTION && (feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP)){
        c->scores = (float*)calloc(n_kernels,sizeof(float));
        c->d_scores = (float*)calloc(n_kernels,sizeof(float));
        c->ex_d_scores_diff_grad = (float*)calloc(n_kernels,sizeof(float));
        c->d1_scores = (float*)calloc(n_kernels,sizeof(float));
        c->d2_scores = (float*)calloc(n_kernels,sizeof(float));
        c->d3_scores = (float*)calloc(n_kernels,sizeof(float));
        c->indices = (int*)calloc(n_kernels,sizeof(int));
        
    }
    else{
        c->scores = NULL;
        c->d_scores = NULL;
        c->ex_d_scores_diff_grad = NULL;
        c->d1_scores = NULL;
        c->d2_scores = NULL;
        c->d3_scores = NULL;
        c->indices = NULL;
    }
    if(convolutional_flag == CONVOLUTION || convolutional_flag == NO_CONVOLUTION){
        if(!bool_is_real((float)((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows)))
            c->rows1 = 0;
        else
            c->rows1 = ((input_rows-kernel_rows)/stride1_rows +1 + 2*padding1_rows);
    }
    
    else if(convolutional_flag == TRANSPOSED_CONVOLUTION){
        if(!bool_is_real((float)((input_rows-1)*stride1_rows +kernel_rows - 2*padding1_rows)))
            c->rows1 = 0;
        else
            c->rows1 = ((input_rows-1)*stride1_rows +kernel_rows - 2*padding1_rows);
    }
    
    if(convolutional_flag == CONVOLUTION || convolutional_flag == TRANSPOSED_CONVOLUTION){
        if(!bool_is_real((float)((c->rows1 - pooling_rows)/stride2_rows + 1 + 2*padding2_rows)))
            c->rows2 = 0;
        else
            c->rows2 = ((c->rows1 - pooling_rows)/stride2_rows + 1 + 2*padding2_rows);
        }
    else{
        if(!bool_is_real((float)((input_rows-pooling_rows)/stride2_rows +1 + 2*padding2_rows)))
            c->rows2 = 0;
        else
            c->rows2 = ((input_rows-pooling_rows)/stride2_rows +1 + 2*padding2_rows);
    }
    if(convolutional_flag == CONVOLUTION || convolutional_flag == NO_CONVOLUTION){
        if(!bool_is_real((float)((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols)))
            c->cols1 = 0;
        else
            c->cols1 = ((input_cols-kernel_cols)/stride1_cols +1 + 2*padding1_cols);
    }
    
    else if(convolutional_flag == TRANSPOSED_CONVOLUTION){
        if(!bool_is_real((float)((input_cols-1)*stride1_cols +kernel_cols - 2*padding1_cols)))
            c->cols1 = 0;
        else
            c->cols1 = ((input_cols-1)*stride1_cols +kernel_cols - 2*padding1_cols);
    }
    if(convolutional_flag == CONVOLUTION || convolutional_flag == TRANSPOSED_CONVOLUTION){
        if(!bool_is_real((float)((c->cols1 - pooling_cols)/stride2_cols + 1 + 2*padding2_cols)))
            c->cols2 = 0;
        else
            c->cols2 = ((c->cols1 - pooling_cols)/stride2_cols + 1 + 2*padding2_cols);
        }
    else{
        if(!bool_is_real((float)((input_cols-pooling_cols)/stride2_cols +1 + 2*padding2_cols)))
            c->cols2 = 0;
        else
            c->cols2 = ((input_cols-pooling_cols)/stride2_cols +1 + 2*padding2_cols);
    }
    
    if(training_mode != ONLY_FF){
        c->temp = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
        c->temp2 = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
        c->temp3 = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
        c->error2 = (float*)calloc(channels*input_rows*input_cols,sizeof(float));
    }
    
    else{
        c->temp = NULL;
        c->temp2 = NULL;
        c->temp3 = NULL;
        c->error2 = NULL;
    }
    
    if(convolutional_flag != NO_CONVOLUTION)
    c->pre_activation = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    else
    c->pre_activation = NULL;
    
    if(activation_flag != NO_ACTIVATION)
    c->post_activation = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    else
    c->post_activation = NULL;
    
    if(normalization_flag != NO_NORMALIZATION)
    c->post_normalization = (float*)calloc(n_kernels*c->rows1*c->cols1,sizeof(float));
    else
    c->post_normalization = NULL;

    if(pooling_flag == MAX_POOLING || pooling_flag == AVARAGE_POOLING)
    c->post_pooling = (float*)calloc(n_kernels*c->rows2*c->cols2,sizeof(float));
    else
    c->post_pooling = NULL;
    
    if(convolutional_flag != NO_CONVOLUTION){
        for(i = 0; i < n_kernels; i++){
            c->used_kernels[i] = 1;
            c->kernels[i] = (float*)malloc(sizeof(float)*channels*kernel_rows*kernel_cols);
            if(training_mode == GRADIENT_DESCENT || training_mode == FREEZE_TRAINING || training_mode == FREEZE_BIASES){
                c->d_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
                c->ex_d_kernels_diff_grad[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
                c->d1_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
                c->d2_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
                c->d3_kernels[i] = (float*)calloc(channels*kernel_rows*kernel_cols,sizeof(float));
            }
            else if(training_mode == EDGE_POPUP || feed_forward_flag == EDGE_POPUP){
                c->indices[i] = i;
            }
            for(j = 0; j < channels*kernel_rows*kernel_cols; j++){
                c->kernels[i][j] = random_general_gaussian_xavier_init(channels*input_rows*input_cols);
            }
        }
    }
    
    if(normalization_flag != GROUP_NORMALIZATION){
        c->group_norm = NULL;
        c->group_norm_channels = 0;
    }
    
    else{
        c->group_norm = (bn**)malloc(sizeof(bn*)*n_kernels/group_norm_channels);
        for(i = 0; i < n_kernels/group_norm_channels; i++){
            if(convolutional_flag == CONVOLUTION)
            c->group_norm[i] = batch_normalization(group_norm_channels,c->rows1*c->cols1-2*padding1_rows-2*padding1_cols,i,NO_ACTIVATION);
            else if(convolutional_flag == TRANSPOSED_CONVOLUTION)
            c->group_norm[i] = batch_normalization(group_norm_channels,c->rows1*c->cols1,i,NO_ACTIVATION);
        }
    }
    return c;
}

/* these functions just give an insight on what is allocated or not*/

int exists_d_kernels_cl(cl* c){
    return c->training_mode != EDGE_POPUP && c->training_mode != ONLY_FF && c->convolutional_flag != NO_CONVOLUTION;
}

int exists_d_biases_cl(cl* c){
    return c->training_mode != EDGE_POPUP && c->training_mode != ONLY_FF && c->convolutional_flag != NO_CONVOLUTION;
}

int exists_kernels_cl(cl* c){
    return c->convolutional_flag != NO_CONVOLUTION;
}

int exists_biases_cl(cl* c){
        return c->convolutional_flag != NO_CONVOLUTION;
}

int exists_pre_activation_cl(cl* c){
    return c->convolutional_flag != NO_CONVOLUTION;
}

int exists_post_activation_cl(cl* c){
    return c->activation_flag != NO_ACTIVATION;
}

int exists_normalization_cl(cl* c){
    return c->normalization_flag != NO_NORMALIZATION;
}

int exists_pooling(cl* c){
    return c->pooling_flag == MAX_POOLING || c->pooling_flag == AVARAGE_POOLING;
}

int exists_edge_popup_stuff_cl(cl * c){
    return c->convolutional_flag != NO_CONVOLUTION && (c->feed_forward_flag == EDGE_POPUP || c->training_mode == EDGE_POPUP);
}

int exists_edge_popup_stuff_with_only_training_mode_cl(cl * c){
    return c->convolutional_flag != NO_CONVOLUTION && c->training_mode == EDGE_POPUP;
}

int exists_bp_handler_arrays(cl* c){
    return c->training_mode != ONLY_FF;
}

/* Given a cl* structure this function frees the space allocated by this structure
 * 
 * Input:
 * 
 *             @ cl* c:= the convolutional structure
 * 
 * */
void free_convolutional(cl* c){
    if(c == NULL){
        return;
    }
    
    int i;
    for(i = 0; i < c->n_kernels; i++){
        if(exists_kernels_cl(c))
        free(c->kernels[i]);
        if(exists_d_kernels_cl(c)){
            free(c->d_kernels[i]);
            free(c->ex_d_kernels_diff_grad[i]);
            free(c->d1_kernels[i]);
            free(c->d2_kernels[i]);
            free(c->d3_kernels[i]);
        }
    }
    
    free(c->kernels);
    free(c->used_kernels);
    free(c->d_kernels);
    free(c->ex_d_kernels_diff_grad);
    free(c->d1_kernels);
    free(c->d2_kernels);
    free(c->biases);
    free(c->d_biases);
    free(c->ex_d_biases_diff_grad);
    free(c->d1_biases);
    free(c->d2_biases);
    free(c->d3_biases);
    free(c->pre_activation);
    free(c->post_activation);
    free(c->post_normalization);
    free(c->post_pooling);
    free(c->temp);
    free(c->temp2);
    free(c->temp3);
    free(c->pooltemp);
    free(c->error2);
    free(c->indices);
    free(c->scores);
    free(c->d_scores);
    free(c->ex_d_scores_diff_grad);
    free(c->d1_scores);
    free(c->d2_scores);
    free(c->d3_scores);
    if(c->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < c->n_kernels/c->group_norm_channels; i++){
            free_batch_normalization(c->group_norm[i]);
        }
        free(c->group_norm);
    }
    free(c);
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
    
    i = fwrite(&f->k_percentage,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->feed_forward_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->training_mode,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
        exit(1);
    }
    
    i = fwrite(&f->group_norm_channels,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a cl layer\n");
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
    
    if(exists_kernels_cl(f)){
        for(k = 0; k < f->n_kernels; k++){
            i = fwrite((f->kernels[k]),sizeof(float)*f->channels*f->kernel_rows*f->kernel_cols,1,fw);

        
            if(i != 1){
                fprintf(stderr,"Error: an error occurred saving a cl layer\n");
                exit(1);
            }
        }
    }
        
    if(exists_biases_cl(f)){
        i = fwrite(f->biases,sizeof(float)*f->n_kernels,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a cl layer\n");
            exit(1);
        }
    }
    
    if(exists_edge_popup_stuff_cl(f)){
        
        i = fwrite(f->scores,sizeof(float)*f->n_kernels,1,fw);
        
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a cl layer\n");
            exit(1);
        }
        
        i = fwrite(f->indices,sizeof(int)*f->n_kernels,1,fw);
        
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a cl layer\n");
            exit(1);
        }
    }
    
    if(exists_kernels_cl(f)){
    
        i = fwrite(f->used_kernels,sizeof(int)*f->n_kernels,1,fw);
        
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a cl layer\n");
            exit(1);
        }
    }
    
    i = fclose(fw);
    
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(k = 0; k < f->n_kernels/f->group_norm_channels; k++){
            save_bn(f->group_norm[k],n);
        }
    }
    
    free(s);
    
}


/* This function copies the values in weights and biases vectors in the weights 
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
    if(exists_biases_cl(f) && exists_kernels_cl(f)){
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
    
    else{
        fprintf(stderr,"Error: this layer is used only for convolution doesn not contain any kernel or bias\n");
        exit(1);
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
    int group_norm_channels = 0;
    int training_mode = 0,feed_forward_flag = 0;
    float** kernels = NULL;
    float* biases = NULL;
    float* scores = NULL;
    int* indices = NULL;
    int* used_kernels = NULL;
    bn** group_norm = NULL;
    float k_percentage;
    
    i = fread(&k_percentage,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    i = fread(&feed_forward_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&training_mode,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
    i = fread(&group_norm_channels,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a cl layer\n");
        exit(1);
    }
    
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
    
    if(convolutional_flag != NO_CONVOLUTION){
        kernels = (float**)malloc(sizeof(float*)*n_kernels);
        biases = (float*)malloc(sizeof(float)*n_kernels);
        used_kernels = (int*)malloc(sizeof(int)*n_kernels);
        if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
            scores = (float*)malloc(sizeof(float)*n_kernels);
            indices = (int*)malloc(sizeof(int)*n_kernels);
        }
    }
    
    if(convolutional_flag != NO_CONVOLUTION){
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
    
        if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
            i = fread(scores,sizeof(float)*n_kernels,1,fr);
            
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a cl layer\n");
                exit(1);
            }
            
            i = fread(indices,sizeof(int)*n_kernels,1,fr);
            
            if(i != 1){
                fprintf(stderr,"Error: an error occurred loading a cl layer\n");
                exit(1);
            }
        }
        
        i = fread(used_kernels,sizeof(int)*n_kernels,1,fr);
        
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a cl layer\n");
            exit(1);
        }
        
    }
    
    if(normalization_flag == GROUP_NORMALIZATION){
        group_norm = (bn**)malloc(sizeof(bn*)*n_kernels/group_norm_channels);
        for(k = 0; k < n_kernels/group_norm_channels; k++){
            group_norm[k] = load_bn(fr);
        }
    }
    
    cl* f = convolutional(channels, input_rows, input_cols, kernel_rows, kernel_cols, n_kernels, stride1_rows, stride1_cols, padding1_rows, padding1_cols, stride2_rows, stride2_cols, padding2_rows, padding2_cols, pooling_rows, pooling_cols, normalization_flag, activation_flag, pooling_flag, group_norm_channels, convolutional_flag,training_mode,feed_forward_flag,layer);
    if(convolutional_flag != NO_CONVOLUTION)
    copy_cl_params(f,kernels,biases);
    
    if(normalization_flag == GROUP_NORMALIZATION){
        for(k = 0; k < n_kernels/group_norm_channels; k++){
            free_batch_normalization(f->group_norm[k]);
        }
        free(f->group_norm);
        f->group_norm = group_norm;
    }
    
    if(convolutional_flag != NO_CONVOLUTION){
        for(i= 0; i < n_kernels; i++){
            free(kernels[i]);
        }
        copy_int_array(used_kernels,f->used_kernels,n_kernels);
        if(feed_forward_flag == EDGE_POPUP || training_mode == EDGE_POPUP){
            copy_array(scores,f->scores,n_kernels);
            copy_int_array(indices,f->indices,n_kernels);
            
        }
    }
    free(kernels);
    free(biases);
    free(scores);
    free(indices);
    free(used_kernels);
    f->k_percentage = k_percentage;
    return f;
}

/* This function returns a cl* layer that is the same copy of the input f
 * except for the activation arrays , the post normalization and post pooling arrays
 * and all the temporary arrays used for the feed forward and back propagation
 * You have a cl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure.
 * 
 * Input:
 * 
 *             @ cl* f:= the convolutional layer that must be copied
 * 
 * */
cl* copy_cl(cl* f){
    if(f == NULL)
        return NULL;
    cl* copy = convolutional(f->channels,f->input_rows,f->input_cols,f->kernel_rows,f->kernel_cols,f->n_kernels,f->stride1_rows,f->stride1_cols,f->padding1_rows,f->padding1_cols,f->stride2_rows,f->stride2_cols,f->padding2_rows,f->padding2_cols,f->pooling_rows,f->pooling_cols,f->normalization_flag,f->activation_flag,f->pooling_flag,f->group_norm_channels, f->convolutional_flag,f->training_mode,f->feed_forward_flag,f->layer);
    
    int i;
    for(i = 0; i < f->n_kernels; i++){
        if(exists_kernels_cl(f)){
            copy_array(f->kernels[i],copy->kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
            copy_int_array(f->used_kernels,copy->used_kernels,f->n_kernels);
        }
        if(exists_d_kernels_cl(f)){
            copy_array(f->d_kernels[i],copy->d_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
            copy_array(f->ex_d_kernels_diff_grad[i],copy->ex_d_kernels_diff_grad[i],f->channels*f->kernel_rows*f->kernel_cols);
            copy_array(f->d1_kernels[i],copy->d1_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
            copy_array(f->d2_kernels[i],copy->d2_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
            copy_array(f->d3_kernels[i],copy->d3_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        }
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            paste_bn(f->group_norm[i],copy->group_norm[i]);
        }
    }
    
    if(exists_biases_cl(f))
    copy_array(f->biases,copy->biases,f->n_kernels);
    if(exists_d_biases_cl(f)){
        copy_array(f->d_biases,copy->d_biases,f->n_kernels);
        copy_array(f->ex_d_biases_diff_grad,copy->ex_d_biases_diff_grad,f->n_kernels);
        copy_array(f->d1_biases,copy->d1_biases,f->n_kernels);
        copy_array(f->d2_biases,copy->d2_biases,f->n_kernels);
        copy_array(f->d3_biases,copy->d3_biases,f->n_kernels);
    }
    
    if(exists_edge_popup_stuff_cl(f)){
        copy_array(f->scores,copy->scores,f->n_kernels);
        copy_array(f->d_scores,copy->d_scores,f->n_kernels);
        copy_array(f->ex_d_scores_diff_grad,copy->ex_d_scores_diff_grad,f->n_kernels);
        copy_array(f->d1_scores,copy->d1_scores,f->n_kernels);
        copy_array(f->d2_scores,copy->d2_scores,f->n_kernels);
        copy_array(f->d3_scores,copy->d3_scores,f->n_kernels);
        copy_int_array(f->indices,copy->indices,f->n_kernels);
        
    }
    return copy;
}



/* this function resets all the arrays of a convolutional layer
 * used during the feed forward and backpropagation
 * You have a cl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
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
    if(exists_d_kernels_cl(f) || exists_d_biases_cl(f)){
        for(i = 0; i < f->n_kernels; i++){
            if(exists_d_kernels_cl(f)){
                for(j = 0; j < f->channels*f->kernel_rows*f->kernel_cols; j++){
                    f->d_kernels[i][j] = 0;
                }
            }
            if(exists_d_biases_cl(f))
            f->d_biases[i] = 0;
        }
    }
    
    if(exists_bp_handler_arrays(f) || exists_pre_activation_cl(f) || exists_post_activation_cl(f) || exists_normalization_cl(f)){
        for(i = 0; i < f->n_kernels*f->rows1*f->cols1; i++){
            if(exists_pre_activation_cl(f))
            f->pre_activation[i] = 0;
            if(exists_post_activation_cl(f))
            f->post_activation[i] = 0;
            if(exists_normalization_cl(f))
            f->post_normalization[i] = 0;
            if(exists_bp_handler_arrays(f)){
                f->temp[i] = 0;
                f->temp2[i] = 0;
                f->temp3[i] = 0;
            }
        }
    }
    
    if(exists_pooling(f)){
        for(i = 0; i < f->n_kernels*f->rows2*f->cols2; i++){
            f->post_pooling[i] = 0;
        }
    }
    
    
    
    for(i = 0; i < f->channels*f->input_rows*f->input_cols; i++){
        f->error2[i] = 0;
        if(f->convolutional_flag == NO_CONVOLUTION)
        f->pooltemp[i] = 0;
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            reset_bn(f->group_norm[i]);
        }
    }
    
    if(exists_edge_popup_stuff_with_only_training_mode_cl(f)){
        for(i = 0; i < f->n_kernels; i++){
            f->d_scores[i] = 0;
            f->indices[i] = i;
        }
        for(i = 0; i < f->n_kernels; i++){
            f->used_kernels[i] = 0;
        }
        
        sort(f->scores,f->indices,0,f->n_kernels-1);

        
        for(i = f->n_kernels-f->n_kernels*f->k_percentage; i < f->n_kernels; i++){
            f->used_kernels[(int)(f->indices[i])] = 1;
        }
    }
    return f;
}
/* Is the same as the other resets except fot he partial derivatives (also the scores are not taken in consideration)
 * 
 * Input:
 * 
 *             @ cl* f:= a cl* f layer
 * 
 * */
cl* reset_cl_except_partial_derivatives(cl* f){
    if(f == NULL)
        return NULL;
    
    int i,j;
    
    if(exists_bp_handler_arrays(f) || exists_pre_activation_cl(f) || exists_post_activation_cl(f) || exists_normalization_cl(f)){
        for(i = 0; i < f->n_kernels*f->rows1*f->cols1; i++){
            if(exists_pre_activation_cl(f))
            f->pre_activation[i] = 0;
            if(exists_post_activation_cl(f))
            f->post_activation[i] = 0;
            if(exists_normalization_cl(f))
            f->post_normalization[i] = 0;
            if(exists_bp_handler_arrays(f)){
                f->temp[i] = 0;
                f->temp2[i] = 0;
                f->temp3[i] = 0;
            }
        }
    }
    
    if(exists_pooling(f)){
        for(i = 0; i < f->n_kernels*f->rows2*f->cols2; i++){
            f->post_pooling[i] = 0;
        }
    }
    
    
    
    for(i = 0; i < f->channels*f->input_rows*f->input_cols; i++){
        f->error2[i] = 0;
        if(f->convolutional_flag == NO_CONVOLUTION)
        f->pooltemp[i] = 0;
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            reset_bn_except_partial_derivatives(f->group_norm[i]);
        }
    }

    return f;
}

/* as all the others but the derivatives of w and b are not resets
 * 
 * Input:
 * 
 *             @ cl* f:= a cl* f layer
 * 
 * */
cl* reset_cl_without_dwdb(cl* f){
    if(f == NULL)
        return NULL;
    
    int i,j;
    
    if(exists_bp_handler_arrays(f) || exists_pre_activation_cl(f) || exists_post_activation_cl(f) || exists_normalization_cl(f)){
        for(i = 0; i < f->n_kernels*f->rows1*f->cols1; i++){
            if(exists_pre_activation_cl(f))
            f->pre_activation[i] = 0;
            if(exists_post_activation_cl(f))
            f->post_activation[i] = 0;
            if(exists_normalization_cl(f))
            f->post_normalization[i] = 0;
            if(exists_bp_handler_arrays(f)){
                f->temp[i] = 0;
                f->temp2[i] = 0;
                f->temp3[i] = 0;
            }
        }
    }
    
    if(exists_pooling(f)){
        for(i = 0; i < f->n_kernels*f->rows2*f->cols2; i++){
            f->post_pooling[i] = 0;
        }
    }
    
    
    
    for(i = 0; i < f->channels*f->input_rows*f->input_cols; i++){
        f->error2[i] = 0;
        if(f->convolutional_flag == NO_CONVOLUTION)
        f->pooltemp[i] = 0;
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            reset_bn(f->group_norm[i]);
        }
    }
    
    if(exists_edge_popup_stuff_with_only_training_mode_cl(f)){
        for(i = 0; i < f->n_kernels; i++){
            f->d_scores[i] = 0;
            f->indices[i] = i;
        }
        for(i = 0; i < f->n_kernels; i++){
            f->used_kernels[i] = 0;
        }
        
        sort(f->scores,f->indices,0,f->n_kernels-1);

        
        for(i = f->n_kernels-f->n_kernels*f->k_percentage; i < f->n_kernels; i++){
            f->used_kernels[(int)(f->indices[i])] = 1;
        }
    }
    return f;
}

/* This function resets the temop arrays + just partial derivatives of d_Scores
 * 
 * Input:
 * 
 *             @ cl* f:= a cl* f layer
 * 
 * */
cl* reset_cl_for_edge_popup(cl* f){
    if(f == NULL)
        return NULL;
    
    int i,j;
    
    if(exists_bp_handler_arrays(f) || exists_pre_activation_cl(f) || exists_post_activation_cl(f) || exists_normalization_cl(f)){
        for(i = 0; i < f->n_kernels*f->rows1*f->cols1; i++){
            if(exists_pre_activation_cl(f))
            f->pre_activation[i] = 0;
            if(exists_post_activation_cl(f))
            f->post_activation[i] = 0;
            if(exists_normalization_cl(f))
            f->post_normalization[i] = 0;
            if(exists_bp_handler_arrays(f)){
                f->temp[i] = 0;
                f->temp2[i] = 0;
                f->temp3[i] = 0;
            }
        }
    }
    
    if(exists_pooling(f)){
        for(i = 0; i < f->n_kernels*f->rows2*f->cols2; i++){
            f->post_pooling[i] = 0;
        }
    }
    
    
    
    for(i = 0; i < f->channels*f->input_rows*f->input_cols; i++){
        f->error2[i] = 0;
        if(f->convolutional_flag == NO_CONVOLUTION)
        f->pooltemp[i] = 0;
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            reset_bn(f->group_norm[i]);
        }
    }
    
    if(exists_edge_popup_stuff_with_only_training_mode_cl(f)){
        for(i = 0; i < f->n_kernels; i++){
            f->d_scores[i] = 0;
        }
    }
    return f;
}

/* this function returns the space allocated by the arrays of f (more or less)
 * 
 * Input:
 * 
 *             cl* f:= the convolutional layer f
 * 
 * */
uint64_t size_of_cls(cl* f){
    uint64_t sum = 0;
    if(exists_biases_cl(f))
        sum+=f->n_kernels*sizeof(float);
    if(exists_bp_handler_arrays(f))
        sum+=(3*f->n_kernels*f->rows1*f->cols1 + f->channels*f->input_rows*f->input_cols)*sizeof(float);
    if(exists_d_biases_cl(f))
        sum+=5*f->n_kernels*sizeof(float);
    if(exists_d_kernels_cl(f))
        sum+=5*f->n_kernels*f->kernel_rows*f->kernel_cols*sizeof(float);
    if(exists_edge_popup_stuff_cl(f))
        sum+=6*f->n_kernels*sizeof(float)*f->n_kernels*sizeof(int);
    if(exists_kernels_cl(f))
        sum+=f->n_kernels*f->kernel_rows*f->kernel_cols*sizeof(float) + f->n_kernels*sizeof(int);
    if(exists_pre_activation_cl(f))
        sum+=f->n_kernels*f->rows1*f->cols1*sizeof(float);
    if(exists_post_activation_cl(f))
        sum+=f->n_kernels*f->rows1*f->cols1*sizeof(float);
    if(exists_normalization_cl(f))
        sum+=f->n_kernels*f->rows1*f->cols1*sizeof(float);
    if(exists_pooling(f))
        sum+=f->n_kernels*f->rows2*f->cols2*sizeof(float);
    if(f->normalization_flag == GROUP_NORMALIZATION)
        sum+=size_of_bn(f->group_norm[0])*f->n_kernels/f->group_norm_channels;
    
    return sum;
}

/* This function pastes f in copy
 * except for the activation arrays , the post normalization and post pooling arrays
 * and all the arrays used by the feed forward and backpropagation.
 * the d_arrays are copied
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
    copy->k_percentage = f->k_percentage;
    if(exists_kernels_cl(f) || exists_d_kernels_cl(f)){
        if(exists_kernels_cl(f))
        copy_int_array(f->used_kernels,copy->used_kernels,f->n_kernels);
        for(i = 0; i < f->n_kernels; i++){
            if(exists_kernels_cl(f)){
                copy_array(f->kernels[i],copy->kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
            }
            if(exists_d_kernels_cl(f)){
                copy_array(f->d_kernels[i],copy->d_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
                copy_array(f->ex_d_kernels_diff_grad[i],copy->ex_d_kernels_diff_grad[i],f->channels*f->kernel_rows*f->kernel_cols);
                copy_array(f->d1_kernels[i],copy->d1_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
                copy_array(f->d2_kernels[i],copy->d2_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
                copy_array(f->d3_kernels[i],copy->d3_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
            }
        }
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            paste_bn(f->group_norm[i],copy->group_norm[i]);
        }
    }
    
    if(exists_biases_cl(f))
    copy_array(f->biases,copy->biases,f->n_kernels);
    if(exists_d_biases_cl(f)){
        copy_array(f->d_biases,copy->d_biases,f->n_kernels);
        copy_array(f->ex_d_biases_diff_grad,copy->ex_d_biases_diff_grad,f->n_kernels);
        copy_array(f->d1_biases,copy->d1_biases,f->n_kernels);
        copy_array(f->d2_biases,copy->d2_biases,f->n_kernels);
        copy_array(f->d3_biases,copy->d3_biases,f->n_kernels);
    }
    if(exists_edge_popup_stuff_cl(f)){
        copy_int_array(f->indices,copy->indices,f->n_kernels);
        copy_array(f->scores,copy->scores,f->n_kernels);
        copy_array(f->d_scores,copy->d_scores,f->n_kernels);
        copy_array(f->ex_d_scores_diff_grad,copy->ex_d_scores_diff_grad,f->n_kernels);
        copy_array(f->d1_scores,copy->d1_scores,f->n_kernels);
        copy_array(f->d2_scores,copy->d2_scores,f->n_kernels);
        copy_array(f->d3_scores,copy->d3_scores,f->n_kernels);
    }
    return;
}


/* This function pastes f in copy only for the w,w norm, b norm, scores, indices
 * Input:
 * 
 *             @ cl* f:= the convolutional layer that must be copied
 *             @ cl* copy:= the convolutional layer where f is copied
 * 
 * */
void paste_w_cl(cl* f, cl* copy){
    if(f == NULL)
        return;
    copy->k_percentage = f->k_percentage;
    int i;
    if(exists_kernels_cl(f)){
        copy_int_array(f->used_kernels,copy->used_kernels,f->n_kernels);
        for(i = 0; i < f->n_kernels; i++){
            copy_array(f->kernels[i],copy->kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        }
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            paste_w_bn(f->group_norm[i],copy->group_norm[i]);
        }
    }
    if(exists_biases_cl(f))
    copy_array(f->biases,copy->biases,f->n_kernels);
    if(exists_edge_popup_stuff_cl(f)){
        copy_int_array(f->indices,copy->indices,f->n_kernels);
        copy_array(f->scores,copy->scores,f->n_kernels);
    }
    return;
}

/* This function pastes f in copy with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ cl* f:= the convolutional layer that must be copied
 *             @ cl* copy:= the convolutional layer where f is copied
 *                @ float tau:= the tau param \in [0,1]
 * */
void slow_paste_cl(cl* f, cl* copy,float tau){
    if(f == NULL)
        return;
    
    int i,j;
    if(exists_kernels_cl(f) || exists_biases_cl(f)){
        if(exists_kernels_cl(f))
        copy_int_array(f->used_kernels,copy->used_kernels,f->n_kernels);
        for(i = 0; i < f->n_kernels; i++){
            if(exists_kernels_cl(f)){
                for(j = 0; j < f->channels*f->kernel_rows*f->kernel_cols; j++){
                    copy->kernels[i][j] = tau*f->kernels[i][j] + (1-tau)*copy->kernels[i][j];
                }
            }
            if(exists_biases_cl(f))
            copy->biases[i] = tau*f->biases[i] + (1-tau)*copy->biases[i];
        }
    }
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            slow_paste_bn(f->group_norm[i],copy->group_norm[i],tau);
        }
    }
    
     if(exists_edge_popup_stuff_cl(f)){
         for(i = 0; i < f->n_kernels; i++){
             copy->scores[i] = tau*f->scores[i] + (1-tau)*copy->scores[i];
             copy->indices[i] = i;
             copy->used_kernels[i] = 0;
         }
         sort(copy->scores,copy->indices,0,f->n_kernels-1);

        
         for(i = copy->n_kernels-copy->n_kernels*copy->k_percentage; i < copy->n_kernels; i++){
            copy->used_kernels[(int)(copy->indices[i])] = 1;
         }
     }
    return;
}


/* this function gives the number of float params for biases and weights in a cl
 * 
 * Input:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 * */
uint64_t get_array_size_params_cl(cl* f){
    
    uint64_t sum = 0;
    int i;
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            sum+=(uint64_t)f->group_norm[i]->vector_dim*2;
        }
    }
    return sum+(uint64_t)f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels;
}

/* this function gives the number of float params for weights in a cl
 * 
 * Input:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 * */
uint64_t get_array_size_weights_cl(cl* f){
    uint64_t sum = 0;
    int i;
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            sum+=(uint64_t)f->group_norm[i]->vector_dim*2;
        }
    }
    return sum+(uint64_t)f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols;
}
/* this function gives the number of scores of the convolutional layer
 * 
 * Inputs:
 * 
 * 
 *             @ cl* f:= the convolutional layer
 * */
uint64_t get_array_size_scores_cl(cl* f){
    return (uint64_t)f->n_kernels;
}

/* this function pastes the weights and biases from a vector in a cl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_params_cl(cl* f, float* vector){
    int i;
    for(i = 0; i < f->n_kernels; i++){
        memcpy(f->kernels[i],&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->channels*f->kernel_rows*f->kernel_cols*sizeof(float));    
    }
    
    memcpy(f->biases,&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->n_kernels*sizeof(float));
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(f->group_norm[i]->gamma,&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+i*f->group_norm[i]->vector_dim],f->group_norm[i]->vector_dim*sizeof(float));
        }
        
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(f->group_norm[i]->beta,&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+(f->n_kernels/f->group_norm_channels+i)*f->group_norm[i]->vector_dim],f->group_norm[i]->vector_dim*sizeof(float));
        }
    }
}


/* this function pastes the cl structure weights and biases in a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_params_to_vector_cl(cl* f, float* vector){
    int i;
    for(i = 0; i < f->n_kernels; i++){
        memcpy(&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->kernels[i],f->channels*f->kernel_rows*f->kernel_cols*sizeof(float));    
    }
    
    memcpy(&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->biases,f->n_kernels*sizeof(float));
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+i*f->group_norm[i]->vector_dim],f->group_norm[i]->gamma,f->group_norm[i]->vector_dim*sizeof(float));
        }
        
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+(f->n_kernels/f->group_norm_channels+i)*f->group_norm[i]->vector_dim],f->group_norm[i]->beta,f->group_norm[i]->vector_dim*sizeof(float));
        }
    }
}

/* this function pastes the weights from a vector in a cl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copied everything
 * */
void memcopy_vector_to_weights_cl(cl* f, float* vector){
    int i;
    for(i = 0; i < f->n_kernels; i++){
        memcpy(f->kernels[i],&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->channels*f->kernel_rows*f->kernel_cols*sizeof(float));    
    }
    
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(f->group_norm[i]->gamma,&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+i*f->group_norm[i]->vector_dim],f->group_norm[i]->vector_dim*sizeof(float));
        }
        
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(f->group_norm[i]->beta,&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+(f->n_kernels/f->group_norm_channels+i)*f->group_norm[i]->vector_dim],f->group_norm[i]->vector_dim*sizeof(float));
        }
    }
}


/* this function pastes the cl structure weights in a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_weights_to_vector_cl(cl* f, float* vector){
    int i;
    for(i = 0; i < f->n_kernels; i++){
        memcpy(&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->kernels[i],f->channels*f->kernel_rows*f->kernel_cols*sizeof(float));    
    }
    
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+i*f->group_norm[i]->vector_dim],f->group_norm[i]->gamma,f->group_norm[i]->vector_dim*sizeof(float));
        }
        
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+(f->n_kernels/f->group_norm_channels+i)*f->group_norm[i]->vector_dim],f->group_norm[i]->beta,f->group_norm[i]->vector_dim*sizeof(float));
        }
    }
}

/* this function pastes the cl structure weights and biases in a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_scores_to_vector_cl(cl* f, float* vector){
    memcpy(vector,f->scores,f->n_kernels*sizeof(float));    
    
}

/* this function pastes the weights and biases from a vector in a cl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_scores_cl(cl* f, float* vector){
    memcpy(f->scores,vector,f->n_kernels*sizeof(float));    
}

/* this function pastes the vector in the the dweights and dbiases of a cl structure
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_derivative_params_cl(cl* f, float* vector){
    int i;
    for(i = 0; i < f->n_kernels; i++){
        memcpy(f->d_kernels[i],&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->channels*f->kernel_rows*f->kernel_cols*sizeof(float));    
    }
    
    memcpy(f->d_biases,&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->n_kernels*sizeof(float));
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(f->group_norm[i]->d_gamma,&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+i*f->group_norm[i]->vector_dim],f->group_norm[i]->vector_dim*sizeof(float));
        }
        
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(f->group_norm[i]->d_beta,&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+(f->n_kernels/f->group_norm_channels+i)*f->group_norm[i]->vector_dim],f->group_norm[i]->vector_dim*sizeof(float));
        }
    }
}


/* this function pastes the dweights and dbiases of the cl in a vector
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* f:= the convolutional layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_derivative_params_to_vector_cl(cl* f, float* vector){
    int i;
    for(i = 0; i < f->n_kernels; i++){
        memcpy(&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->d_kernels[i],f->channels*f->kernel_rows*f->kernel_cols*sizeof(float));    
    }
    
    memcpy(&vector[i*f->channels*f->kernel_rows*f->kernel_cols],f->d_biases,f->n_kernels*sizeof(float));

    if(f->normalization_flag == GROUP_NORMALIZATION){
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+i*f->group_norm[i]->vector_dim],f->group_norm[i]->d_gamma,f->group_norm[i]->vector_dim*sizeof(float));
        }
        
        for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
            memcpy(&vector[f->n_kernels*f->channels*f->kernel_rows*f->kernel_cols+f->n_kernels+(f->n_kernels/f->group_norm_channels+i)*f->group_norm[i]->vector_dim],f->group_norm[i]->d_beta,f->group_norm[i]->vector_dim*sizeof(float));
        }
    }
}

/* setting the biases to 0
 * 
 * Input:
 *             @ cl* c:= the convolutional layer
 * */
void set_convolutional_biases_to_zero(cl* c){
    int i;
    for(i = 0; i < c->n_kernels; i++){
        c->biases[i] = 0;
    }
}

/* setting the unused weights to 0
 * 
 * Input:
 *             @ cl* c:= the convolutional layer
 * */
void set_convolutional_unused_weights_to_zero(cl* c){
    int i,j;
    for(i = 0; i < c->n_kernels-c->n_kernels*c->k_percentage; i++){
        for(j = 0; j < c->channels*c->kernel_rows*c->kernel_cols; j++){
            c->kernels[(c->indices[i])][j] = 0;
        }
    }
}


/* this function, given 2 input convolutional layers sum up the scores in he output convolutional layer
 * 
 * Inputs:
 * 
 * 
 *             @ cl* input1:= the first input convolutional layer
 *             @ cl* input2:= the second input convolutional layer
 *             @ cl* output:= the output convolutional layer
 * */
void sum_score_cl(cl* input1, cl* input2, cl* output){
    sum1D(input1->scores,input2->scores,output->scores,input1->n_kernels);
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
void compare_score_cl(cl* input1, cl* input2, cl* output){
    int i;
    for(i = 0; i < input1->n_kernels; i++){
        if(input1->scores[i] > input2->scores[i])
            output->scores[i] = input1->scores[i];
        else
            output->scores[i] = input2->scores[i];
    }
}

/* this function sum up the scores in input1 and input2 in output
 * 
 * Input:
 * 
 * 
 *                 @ fcl* input1:= the first input fcl layer
 *                 @ float* input2:= the vector
 *                 @ fcl* output:= the output fcl layer
 * */
void compare_score_cl_with_vector(cl* input1, float* input2, cl* output){
    int i;
    for(i = 0; i < input1->n_kernels; i++){
        if(input1->scores[i] > input2[i])
            output->scores[i] = input1->scores[i];
        else
            output->scores[i] = input2[i];
    }
}
    
/* This function divides all the scores with value
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ cl* c:= the convolutional layer 
 *                 @ float value:= the value that is gonna divide the scores
 * 
 * */
void dividing_score_cl(cl* c,float value){
    int i;
    for(i = 0; i < c->n_kernels;i++){
        c->scores[i]/=value;
    }
}


/* This function set all the scores to 0
 * 
 * Input:
 * 
 * 
 *                 @ cl* f:= the convolutional input layer
 * */
void reset_score_cl(cl* f){
    if(f->convolutional_flag == NO_CONVOLUTION)
        return;
    int i;
    for(i = 0; i < f->n_kernels; i++){
        f->scores[i] = 0;
    }
    
}


/* This function re initialize the weights which scores is < of a goodness (range [0,1])
 * or the weights are among the worst scores in the total_scores*percentage (range percentage [0,1])
 * The re initialization happens with the signed kaiming constant (the best initialization for edge-popup according to the paper)
 * 
 * Input:
 * 
 *                     @ cl* f:= the convolutional layer
 *                     @ float percentage:= the percentage
 *                    @ float goodness:= the goodness value
 * */
void reinitialize_weights_according_to_scores_cl(cl* f, float percentage, float goodness){
    if(f->convolutional_flag == NO_CONVOLUTION)
        return;
    int i,j;
    for(i = 0; i < f->n_kernels; i++){
        if(i >= f->n_kernels*percentage)
            return;
        if(f->scores[f->indices[i]] < goodness){
            for(j = 0; j < f->channels*f->kernel_rows*f->kernel_cols; j++){
                f->kernels[f->indices[i]][j] = signed_kaiming_constant((float)f->channels*f->input_rows*f->input_cols);
            }
        } 
    }
}


/* this function re initializes the weights of a convolutional layers (all the weights)
 * 
 * Inputs:
 * 
 * 
 *             @ cl* f:= the convolutional layer
 * */
void reinitialize_w_cl(cl* f){
    int i,j;
    for(i = 0; i < f->n_kernels; i++){
        for(j = 0; j < f->channels*f->kernel_cols*f->kernel_rows; j++){
            f->kernels[i][j] = random_general_gaussian_xavier_init((float)f->channels*f->input_rows*f->input_cols);
        }
    }
}

/* this function reset all the arrays of partial derivatives of the scores
 * 
 * Inputs:
 * 
 *             @ cl* f:= the convolutional layer which derivative scores must be re initialized
 * */
cl* reset_edge_popup_d_cl(cl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_kernels; i++){
        f->d_scores[i] = 0;
        f->d1_scores[i] = 0;
        f->d2_scores[i] = 0;
        f->d3_scores[i] = 0;
        f->ex_d_scores_diff_grad[i] = 0;
    }
    return f;
}

/* this function set all the scores to -99999
 * 
 * Inputs:
 * 
 *             @ cl* f:= the convolutional layer which scores must be set to a low value
 * */
void set_low_score_cl(cl* f){
    if(f->convolutional_flag == NO_CONVOLUTION)
        return;
    int i;
    for(i = 0; i < f->n_kernels; i++){
        f->scores[i] = -99999;
    }
    
}
