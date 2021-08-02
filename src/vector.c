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

/*
 * 
 * float* v:= a vector pre created maybe with some random values (like gaussian ones) to make multiplication (think about autoencoder gaussian variance
 * int:= v_size:= the size of v
 * int output_size:= the size of the output of the structure
 * int action:= multiplication/addition/concatenate/resize
 * int activation flag:= you know...
 * int dropout_flag:= dropout or not. if you want dropout test just fill the v vector
 * int index:= this index is used for the concatanetion and the resize, look at the feed forward
 * float dropout_threshold:= the threshold for dropout
 * int input_size:= is used only for the resize or concatenate, sum of the input sizes, or multiplication or sum of 2 inputs
 * 
 * */


vector_struct* create_vector(float* v, int v_size, int output_size, int action, int activation_flag, int dropout_flag, int index, float dropout_threshold, int input_size){

    vector_struct* ve = (vector_struct*)malloc(sizeof(vector_struct));
    ve->v = v;
    ve->v_size = v_size;
    ve->action = action;
    ve->activation_flag = activation_flag;
    ve->dropout_flag = dropout_flag;
    ve->index = index;
    
    ve->dropout_threshold = dropout_threshold;
    if(action != RESIZE)
    ve->output = (float*)calloc(output_size,sizeof(float));
    ve->output_size = output_size;
    ve->input_error = (float*)calloc(input_size,sizeof(float));
    ve->input_size = input_size;
    return ve;
}


void free_vector(vector_struct* v){
    free(v->v);
    free(v->output);
    free(v->input_error);
    free(v);
    return;
}

void reset_vector(vector_struct* v){
    int i;
    if(v->action != RESIZE){
        for(i = 0; i < v->output_size; i++){
            v->output[i] = 0;
        }
    }
    else
    set_vector_with_value(0.0,v->input_error,v->input_size);
}


vector_struct* copy_vector(vector_struct* v){
    float* vec = NULL;
    if(v->v != NULL){
        vec = (float*)malloc(sizeof(float)*v->v_size);
        copy_array(v->v,vec,v->v_size);
    }
    return create_vector(vec,v->v_size,v->output_size,v->action, v->activation_flag, v->dropout_flag, v->index, v->dropout_threshold, v->input_size);
}


void paste_vector(vector_struct* v, vector_struct* copy){
    copy_array(v->v,copy->v,v->v_size);
    return;
}

uint64_t size_of_vector(vector_struct* v){
    return (uint64_t)(v->v_size+v->output_size);
}

void save_vector(vector_struct* v, int n){
    if (v == NULL) return;
    int i,j;
    
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* tt = ".bin";
    s = itoa(n,s);
    s = strcat(s,tt);
    
    fw = fopen(s,"a+");
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    
    
    i = fwrite(&v->input_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    i = fwrite(&v->v_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    i = fwrite(&v->output_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    
    i = fwrite(&v->action,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    i = fwrite(&v->activation_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    i = fwrite(&v->dropout_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    i = fwrite(&v->index,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    i = fwrite(&v->dropout_threshold,sizeof(float),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a vector\n");
        exit(1);
    }
    
    int is_null = 1;
    if(v->v == NULL){
        i = fwrite(&is_null,sizeof(int),1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a vector\n");
            exit(1);
        }
    }
    else{
        is_null = 0;
        i = fwrite(&is_null,sizeof(int),1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a vector\n");
            exit(1);
        }
        i = fwrite(v->v,sizeof(float)*v->v_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving a vector\n");
            exit(1);
        }
    }
    
    
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    free(s);
    
    
    return;
}

vector_struct* load_vector(FILE* fr){
    if(fr == NULL)
        return NULL;
    int v_size, output_size,action, activation_flag, dropout_flag, index, input_size,i, is_null;
    float dropout_threshold;
    float* v = NULL;
    
    i = fread(&input_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&v_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&output_size,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&action,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&activation_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&dropout_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&index,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    i = fread(&dropout_threshold,sizeof(float),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    
    i = fread(&is_null,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a vector\n");
        exit(1);
    }
    
    if(!is_null){
        float* v = (float*)calloc(v_size,sizeof(float));
        i = fread(v,sizeof(float)*v_size,1,fr);
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading a vector\n");
            exit(1);
        }
    }
    
    return create_vector(v,v_size,output_size,action,activation_flag,dropout_flag,index, dropout_threshold, input_size);
}

void ff_vector(float* input1,float* input2, vector_struct* v){
    if(input2 == NULL){
        if(v->dropout_flag == DROPOUT){
            if(v->v_size != v->output_size){
                fprintf(stderr,"Error: you have dropout, we use the v vector to store the mask, v_size must be equal to output_size\n");
                exit(1);
            }
            set_vector_with_value(1.0,v->v,v->v_size);
            set_dropout_mask(v->output_size,v->v, v->dropout_threshold);
            dot1D(input1,v->v,v->output,v->output_size);
        }
       
        else if(v->dropout_flag == DROPOUT_TEST){
            mul_value(input1,1 - v->dropout_threshold,v->output,v->output_size);
        }
        
        else if(v->action == ADDITION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            sum1D(input1,v->v,v->output,v->output_size);
        }
        else if(v->action == SUBTRACTION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            sub1D(input1,v->v,v->output,v->output_size);
        }
        
        else if(v->action == MULTIPLICATION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            
            dot1D(input1,v->v,v->output,v->output_size);
        }
        
        else if(v->action == DIVISION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            
            div1D(input1,v->v,v->output,v->output_size);
        }
        else if(v->action == INVERSE){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            
            inverse(input1,v->output,v->output_size);
        }
        else if(v->action == CHANGE_SIGN){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            
            mul_value(input1,-1.0,v->output,v->output_size);
        }
        
        else if(v->action == GET_MAX){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            int index = -1,i;
            float max = -1;
            for(i = 0; i < v->output_size; i++){
                if(input1[i] > max){
                    max = input1[i];
                    index = i;
                }
            }
            v->output[index] = 1;
        }
        
        else if(v->action == RESIZE){
            v->output = &input1[v->index];
            //copy_array(&input1[v->index],v->output,min(v->input_size-v->index,v->output_size));
        }
        
        else if(v->activation_flag == SIGMOID){
            sigmoid_array(input1,v->output,v->output_size);
        }
        else if(v->activation_flag == TANH){
            tanhh_array(input1,v->output,v->output_size);
        }
        else if(v->activation_flag == RELU){
            relu_array(input1,v->output,v->output_size);
        }
        else if(v->activation_flag == LEAKY_RELU){
            leaky_relu_array(input1,v->output,v->output_size);
        }
        else if(v->activation_flag == ELU){
            elu_array(input1,v->output,v->output_size,ELU_THRESHOLD);
        }
        else if(v->activation_flag == SOFTMAX){
            softmax(input1,v->output,v->output_size);
        }
    }
    
    else{
        if(v->action == ADDITION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            sum1D(input1,input2,v->output,v->output_size);
        }
        
        else if(v->action == SUBTRACTION){
            sub1D(input1,input2,v->output,v->output_size);
        }
        else if(v->action == MULTIPLICATION){
            
            dot1D(input1,input2,v->output,v->output_size);
        }
        else if(v->action == DIVISION){
            
            div1D(input1,input2,v->output,v->output_size);
        }
        
        else if(v->action = CONCATENATE){
            copy_array(input1,v->output,v->index);//v->index is the index where the second should start and is the length of the first
            copy_array(input2,&v->output[v->index],v->output_size-v->index);
        }
    }
    
    return;
        
}

float* bp_vector(float* input1,float* input2, vector_struct* v, float* output_error){
    if(v->action == ADDITION || v->action == CONCATENATE)
        return output_error;
        
    if(input2 == NULL){
        float* temp = NULL;
        if(v->action == DROPOUT){
            dot1D(output_error,v->v,v->input_error,v->output_size);
        }
        //dropout test does not make sense since is used only for inference
        
        else if(v->action == MULTIPLICATION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            
            dot1D(output_error,v->v,v->input_error,v->output_size);
        }
        else if(v->action == DIVISION){
            if(v->output_size != v->v_size){
                fprintf(stderr,"Error: your v_size != output_size\n");
                exit(1);
            }
            
            div1D(output_error,v->v,v->input_error,v->output_size);
        }
        else if(v->action == SUBTRACTION){
            return output_error;
        }
        
        else if(v->action == INVERSE){
            div1D(output_error,input1,v->input_error,v->output_size);
            div1D(v->input_error,input1,v->input_error,v->output_size);
            mul_value(v->input_error,-1.0,v->input_error,v->output_size);
        }
        else if(v->action == CHANGE_SIGN){
            mul_value(output_error,-1.0,v->input_error,v->output_size);
        }
        
        else if(v->action == RESIZE){
            copy_array(output_error,&v->input_error[v->index],min(v->output_size,v->input_size));
        }
        
        
        else if(v->activation_flag == SIGMOID){
            temp = (float*)calloc(v->output_size,sizeof(float));
            derivative_sigmoid_array(input1,temp,v->output_size);
            dot1D(temp,output_error,v->input_error,v->output_size);
        }
        else if(v->activation_flag == TANH){
            temp = (float*)calloc(v->output_size,sizeof(float));
            derivative_tanhh_array(input1,temp,v->output_size);
            dot1D(temp,output_error,v->input_error,v->output_size);
        }
        else if(v->activation_flag == RELU){
            temp = (float*)calloc(v->output_size,sizeof(float));
            derivative_relu_array(input1,temp,v->output_size);
            dot1D(temp,output_error,v->input_error,v->output_size);
        }
        else if(v->activation_flag == LEAKY_RELU){
            temp = (float*)calloc(v->output_size,sizeof(float));
            derivative_leaky_relu_array(input1,temp,v->output_size);
            dot1D(temp,output_error,v->input_error,v->output_size);
        }
        else if(v->activation_flag == ELU){
            temp = (float*)calloc(v->output_size,sizeof(float));
            derivative_elu_array(input1,temp,v->output_size,ELU_THRESHOLD);
            dot1D(temp,output_error,v->input_error,v->output_size);
        }
        else if(v->activation_flag == SOFTMAX){
            derivative_softmax(v->input_error,v->output,output_error,v->output_size);
        }
        free(temp);
    }
    
    else{
        if(v->action == MULTIPLICATION){
            dot1D(output_error,input2,v->input_error,v->output_size);
            dot1D(output_error,input1,v->input_error+v->output_size,v->output_size);
        }
        else if(v->action == DIVISION){
            div1D(output_error,input2,v->input_error,v->output_size);
            dot1D(output_error,input1,v->input_error+v->output_size,v->output_size);
            div1D(v->input_error+v->output_size,input2,v->input_error+v->output_size,v->output_size);
            div1D(v->input_error+v->output_size,input2,v->input_error+v->output_size,v->output_size);
            mul_value(v->input_error+v->output_size,-1.0,v->input_error+v->output_size,v->output_size);
        }
        else if(v->action == SUBTRACTION){
            copy_array(output_error,v->input_error,v->output_size);
            mul_value(output_error,-1.0,v->input_error+v->output_size,v->output_size);
        }
        
    }
    
    return v->input_error;
}
