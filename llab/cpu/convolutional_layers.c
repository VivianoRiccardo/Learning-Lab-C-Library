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
 * 			   @ int group_norm_channels:= the number of the grouped channels during the group normalization if there is anyone
 * 			   @ int convolutional_flag:= NO_CONVOLUTION to apply only pooling otherwise CONVOLUTION
 * 			   @ int layer:= the layer index
 * 
 * */
 
cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int layer){
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
    
    if(convolutional_flag == CONVOLUTION && normalization_flag == GROUP_NORMALIZATION){
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
    c->group_norm_channels = group_norm_channels;
    
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
    
    if(normalization_flag != GROUP_NORMALIZATION){
		c->group_norm = NULL;
		c->group_norm_channels = 0;
	}
	
	else{
		c->group_norm = (bn**)malloc(sizeof(bn*)*n_kernels/group_norm_channels);
		for(i = 0; i < n_kernels/group_norm_channels; i++){
			c->group_norm[i] = batch_normalization(group_norm_channels,c->rows1*c->cols1,i,NO_ACTIVATION);
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
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
		for(k = 0; k < f->n_kernels/f->group_norm_channels; k++){
			save_bn(f->group_norm[k],n);
		}
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
    int group_norm_channels = 0;
    float** kernels;
    float* biases;
    bn** group_norm = NULL;
    
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
    
    if(normalization_flag == GROUP_NORMALIZATION){
		group_norm = (bn**)malloc(sizeof(bn*)*n_kernels/group_norm_channels);
		for(k = 0; k < n_kernels/group_norm_channels; k++){
			group_norm[k] = load_bn(fr);
		}
	}
    
    cl* f = convolutional(channels, input_rows, input_cols, kernel_rows, kernel_cols, n_kernels, stride1_rows, stride1_cols, padding1_rows, padding1_cols, stride2_rows, stride2_cols, padding2_rows, padding2_cols, pooling_rows, pooling_cols, normalization_flag, activation_flag, pooling_flag, group_norm_channels, convolutional_flag,layer);
    copy_cl_params(f,kernels,biases);
    
    if(normalization_flag == GROUP_NORMALIZATION){
		for(k = 0; k < n_kernels/group_norm_channels; k++){
			free_batch_normalization(f->group_norm[k]);
		}
		free(f->group_norm);
		f->group_norm = group_norm;
	}
    
    for(i= 0; i < n_kernels; i++){
        free(kernels[i]);
    }
    free(kernels);
    free(biases);
    return f;
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
    cl* copy = convolutional(f->channels,f->input_rows,f->input_cols,f->kernel_rows,f->kernel_cols,f->n_kernels,f->stride1_rows,f->stride1_cols,f->padding1_rows,f->padding1_cols,f->stride2_rows,f->stride2_cols,f->padding2_rows,f->padding2_cols,f->pooling_rows,f->pooling_cols,f->normalization_flag,f->activation_flag,f->pooling_flag,f->group_norm_channels, f->convolutional_flag,f->layer);
    
    int i;
    for(i = 0; i < f->n_kernels; i++){
        copy_array(f->kernels[i],copy->kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d_kernels[i],copy->d_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d1_kernels[i],copy->d1_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
        copy_array(f->d2_kernels[i],copy->d2_kernels[i],f->channels*f->kernel_rows*f->kernel_cols);
    }
    if(f->normalization_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
			free_batch_normalization(copy->group_norm[i]);
			copy->group_norm[i] = copy_bn(f->group_norm[i]);
		}
	}
    
    copy_array(f->biases,copy->biases,f->n_kernels);
    copy_array(f->d_biases,copy->d_biases,f->n_kernels);
    copy_array(f->d1_biases,copy->d1_biases,f->n_kernels);
    copy_array(f->d2_biases,copy->d2_biases,f->n_kernels);
    
    return copy;
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
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
			reset_bn(f->group_norm[i]);
		}
	}
    
    return f;
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
    if(f->normalization_flag == GROUP_NORMALIZATION)
		sum+=size_of_bn(f->group_norm[0])*f->n_kernels/f->group_norm_channels;
	
    return sum;
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
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
			paste_bn(f->group_norm[i],copy->group_norm[i]);
		}
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
    
    if(f->normalization_flag == GROUP_NORMALIZATION){
		for(i = 0; i < f->n_kernels/f->group_norm_channels; i++){
			slow_paste_bn(f->group_norm[i],copy->group_norm[i],tau);
		}
	}
    
    return;
}
