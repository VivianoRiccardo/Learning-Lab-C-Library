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

/* This function allocates the space for a l2_Scaled_norm struct
 * 
 * Inputs:
 * 
 *                 @ int vector_dimension:= the dimension of the input it receives as well as the
 *                                          dimension of the output
 * */
scaled_l2_norm* scaled_l2_normalization_layer(int vector_dimension){
    if(vector_dimension <= 0){
        fprintf(stderr,"Error vector dimension must be > 0\n");
        exit(1);
    }
    scaled_l2_norm* l2 = (scaled_l2_norm*)malloc(sizeof(scaled_l2_norm));
    l2->input_dimension = vector_dimension;
    l2->output = (float*)calloc(l2->input_dimension,sizeof(float));
    l2->output_error = (float*)calloc(l2->input_dimension,sizeof(float));
    l2->norm = 1;
    l2->learned_g = sqrtf((float)(vector_dimension));
    l2->d_learned_g = 0;
    l2->d1_learned_g = 0;
    l2->d2_learned_g = 0;
    l2->d3_learned_g = 0;
    l2->ex_d_learned_g_diff_grad = 0;
    l2->training_mode = GRADIENT_DESCENT;
    
    return l2;
}

/* This function deallocates the space occupied by a scaled_l2_normalization_layer struct
 * 
 * Inputs:
 *                 @ scaled_l2_norm* l2:= the struct
 * */
void free_scaled_l2_normalization_layer(scaled_l2_norm* l2){
    if(l2 == NULL)
        return;
    free(l2->output);
    free(l2->output_error);
    free(l2);
}

/* This function saves a scaled_l2_norm layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ scaled_l2_norm* f:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_scaled_l2_norm(scaled_l2_norm* f, int n){
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
    
    i = fwrite(&f->input_dimension,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    i = fwrite(&f->training_mode,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a fcl layer\n");
        exit(1);
    }
    i = fwrite(&f->learned_g,sizeof(float),1,fw);
    
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

/* This function loads a scaled_l2_norm layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
scaled_l2_norm* load_scaled_l2_norm(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int input_dimension = 0, training_mode = GRADIENT_DESCENT;
    float learned_g = 0;
    
    i = fread(&input_dimension,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    i = fread(&training_mode,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    i = fread(&learned_g,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a fcl layer\n");
        exit(1);
    }
    
    scaled_l2_norm* l2 =  scaled_l2_normalization_layer(input_dimension);
    l2->learned_g = learned_g;
    l2->training_mode = training_mode;
  
    return l2;
}

/* this function returns a new allocated struct of scaled_l2_norm with the copy
 * of the parameters of the one passed as input
 * 
 * Inputs:
 * 
 *             @ scaled_l2_norm* f:= the struct we want to copy
 * */
scaled_l2_norm* copy_scaled_l2_norm(scaled_l2_norm* f){
    if(f == NULL)
        return NULL;
    scaled_l2_norm* copy = scaled_l2_normalization_layer(f->input_dimension);
    copy->learned_g = f->learned_g;
    copy->d_learned_g = f->d_learned_g;
    copy->d1_learned_g = f->d1_learned_g;
    copy->d2_learned_g = f->d2_learned_g;
    copy->d3_learned_g = f->d3_learned_g;
    copy->ex_d_learned_g_diff_grad = f->ex_d_learned_g_diff_grad;
    copy->training_mode = f->training_mode;
    return copy;
}

/* this function resetes the d_learned_g parameter as well as the output array where we store 
 * the output computed
 * 
 * Inputs:
 * 
 *                 @ scaled_l2_nor,* f:= the struct we want to reset
 * 
 * */
scaled_l2_norm* reset_scaled_l2_norm(scaled_l2_norm* f){
    if (f == NULL)
        return NULL;
    int i;
    f->d_learned_g = 0;
    for(i = 0; i < f->input_dimension; i++){
        f->output[i] = 0;
        f->output_error[i] = 0;
    }
}

/* returns the dimension occupied by this struct (more or less)
 * 
 * Inputs:
 * 
 * 
 *             @ scaled_l2_norm* f:= the struct we want to know the size that occupies
 * 
 * */
unsigned long long int size_of_scaled_l2_norm(scaled_l2_norm* f){
    return f->input_dimension*2;
}

/* this function copies the g parameter
 * 
 * Input:
 * 
 *             @ scaled_l2_norm* f:= the struct we want to copy from
 *             @ scaled_l2_norm* copy:= where we store the coping parameters
 * */
void paste_scaled_l2_norm(scaled_l2_norm* f,scaled_l2_norm* copy){
    if (f == NULL || copy == NULL)
        return;
    copy->learned_g = f->learned_g;
    copy->d_learned_g = f->d_learned_g;
    copy->d1_learned_g = f->d1_learned_g;
    copy->d2_learned_g = f->d2_learned_g;
    copy->d3_learned_g = f->d3_learned_g;
    copy->ex_d_learned_g_diff_grad = f->ex_d_learned_g_diff_grad;
    copy->training_mode = f->training_mode;
}



void slow_paste_scaled_l2_norm(scaled_l2_norm* f,scaled_l2_norm* copy, float tau){
    if (copy == NULL || f == NULL)
        return;
    copy->learned_g = tau*f->learned_g + (1-tau)*copy->learned_g;
    copy->d_learned_g = tau*f->d_learned_g + (1-tau)*copy->d_learned_g;
    copy->d1_learned_g = tau*f->d1_learned_g + (1-tau)*copy->d1_learned_g;
    copy->d2_learned_g = tau*f->d2_learned_g + (1-tau)*copy->d2_learned_g;
    copy->d3_learned_g = tau*f->d3_learned_g + (1-tau)*copy->d3_learned_g;
    copy->ex_d_learned_g_diff_grad = tau*f->ex_d_learned_g_diff_grad + (1-tau)*copy->ex_d_learned_g_diff_grad;
    copy->training_mode = f ->training_mode;
}

