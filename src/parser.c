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


float** get_inputs_from_multiple_instances_single_char_binary_file_with_single_softmax_output(char* filename,int input_dimension, int instances){
    int i,size=0,j;
    char** ksource = (char**)malloc(sizeof(char*));
    read_file_in_char_vector(ksource,filename,&size);
    char temp[2];
    temp[1] = '\0';
    float** inputs = (float**)malloc(sizeof(float*)*instances);
    for(i = 0; i < instances; i++){
        inputs[i] = (float*)calloc(input_dimension,sizeof(float));
        for(j = 0; j < input_dimension; j++){
            temp[0] = ksource[0][i*(input_dimension+1)+j];
            inputs[i][j] = atof(temp);
        }
    }
    free(ksource);
    return inputs;
}

float** get_outputs_from_multiple_instances_single_char_binary_file_with_single_softmax_output(char* filename,int input_dimension,int output_dimension, int instances){
    int i,size=0,j;
    char** ksource = (char**)malloc(sizeof(char*));
    read_file_in_char_vector(ksource,filename,&size);
    
    char temp[2];
    temp[1] = '\0';
    float** outputs = (float**)malloc(sizeof(float*)*instances);
    for(i = 0; i < instances; i++){
        outputs[i] = (float*)calloc(output_dimension,sizeof(float));
        temp[0] = ksource[0][i*(input_dimension+1)+input_dimension];
        outputs[i][atoi(temp)] = 1;
    }
    return outputs;
}
/* Given a csv file split in input and output this function can take that file and put
 * the input and output in float vectors, let's make an example. Imagine a regression problem
 * and a file split in 3 input features and 1 single output:
 * 1.3452;4.6793;2.3003;8.0000;
 * 
 * The first 3 numbers are the input feature and the fourth is the output
 * 
 * PAY ATTENTION: after the end of the output should there be a ';' character
 * 
 * Inputs:
 * 
 *             @ float* input:= the float vector where will be stored the input from the file
 *             @ float* output:= the float vector where will be stored the output from the file
 *             @ char* filename:= the filename
 *             @ int input_size:= the size of the input. (is not required output size, 'cause the end of the file will handle this factor)
 * */
int single_instance_single_csv_file_parser(float* input, float* output,char* filename,int input_size){
    
    char* ksource;//ksource
    int size = 0,counter = 0, counter2 = 0, z;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return ret;
        
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    for(z = 0; z < size; z++){
        
        if(ksource[z] != ';'){
            temp2[counter] = ksource[z];
            counter++;
        }
        else{
            if(counter2 < input_size){
                temp2[counter] = '\0';
                counter = 0;
                input[counter2]=atof(temp2);
                counter2++;
            }
            else{
                temp2[counter] = '\0';
                counter = 0;
                output[counter2-input_size] = atof(temp2);
                counter2++;
            }
        }
    
    }
    
    free(ksource);
    free(temp2);
    
    return 0;
    
}

/* This function is the same of the one above except that you do what the above function does but for a number n of files
 * 
Inputs:
 * 
 *             @ float** input:= the float vector where will be stored the input from the files, dimension: n_filesxinput_size 
 *             @ float** output:= the float vector where will be stored the output from the files, dimension: n_filesxinput_size
 *             @ char** filename:= the filename, dimension: n_filesx...
 *             @ int input_size:= the size of the input. (is not required output size, 'cause the end of the file will handle this factor)
 *             @ int n_files:= the number of files
 * */        
int single_instance_multiple_csv_file_parser(float** input, float** output,char** filename,int input_size, int n_files){
    int i,ret;
    for(i = 0; i < n_files; i++){
        ret = single_instance_single_csv_file_parser(input[i], output[i],filename[i],input_size);
        if(ret)
            return ret;
    }
    
    return 0;
}

/* Is the same of the first function, but in this case in a single file there are multiple instances per line
 * 
Inputs:
 * 
 *             @ float** input:= the float vector where will be stored the input from the files, dimension: n_filesxinput_size 
 *             @ float** output:= the float vector where will be stored the output from the files, dimension: n_filesxoutput_size
 *             @ char* filename:= the filename
 *             @ int input_size:= the size of the input. (is not required output size, 'cause the end of the line will handle this factor)
 * */        
int multiple_instance_single_csv_file_parser(float** input, float** output,char* filename,int input_size){
    char* ksource;//ksource
    int size = 0,counter = 0, counter2 = 0, z,counter3 = 0;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return ret;
        
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    for(z = 0; z < size; z++){
        
        if(ksource[z] != ';' && ksource[z] != '\n'){
            temp2[counter] = ksource[z];
            counter++;
        }
        else if(ksource[z] != '\n'){
            if(counter2 < input_size){
                temp2[counter] = '\0';
                counter = 0;
                input[counter3][counter2]=atof(temp2);
                counter2++;
            }
            else{
                temp2[counter] = '\0';
                counter = 0;
                output[counter3][counter2-input_size] = atof(temp2);
                counter2++;
            }
        }
        
        else{
            counter3++;
            counter2 = 0;
            counter = 0;
        }
    
    }
    
    free(ksource);
    free(temp2);
    
    return 0;
}

/* Given a file split in input and output this function can take that file and put
 * the input and output in float vectors. each character in the mentioned file should be a feature of input or output let's make an example.
 * Imagine a file with 3 input feature and 3 output feature then:
 * 101001
 * 
 * Inputs:
 * 
 *             @ float* input:= the float vector where will be stored the input from the file
 *             @ float* output:= the float vector where will be stored the output from the file
 *             @ char* filename:= the filename
 *             @ int input_size:= the size of the input. (is not required output size, 'cause the end of the file will handle this factor)
 * */
int single_instance_single_file_parser(float* input, float* output,char* filename,int input_size){
    char* ksource;//ksource
    int size = 0,counter = 0, counter2 = 0, z;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return ret;
        
    char* temp2 = (char*)malloc(sizeof(char)*2);// temp2 = (256)
    for(z = 0; z < size; z++){
        
        temp2[0] = ksource[z];
        temp2[1] = '\0';

        if(z < input_size)
            input[z]=atof(temp2);
        else
            output[z-input_size]=atof(temp2);
        
    }
    
    free(ksource);
    free(temp2);
    
    return 0;
}

/* This function is the same of the one above except that you do what the above function does but for a number n of files
 * 
Inputs:
 * 
 *             @ float** input:= the float vector where will be stored the input from the files, dimension: n_filesxinput_size 
 *             @ float** output:= the float vector where will be stored the output from the files, dimension: n_filesxinput_size
 *             @ char** filename:= the filename, dimension: n_filesx...
 *             @ int input_size:= the size of the input. (is not required output size, 'cause the end of the file will handle this factor)
 *             @ int n_files:= the number of files
 * */        
int single_instance_multiple_file_parser(float** input, float** output,char** filename,int input_size, int n_files){
    int i,ret;
    for(i = 0; i < n_files; i++){
        ret = single_instance_single_file_parser(input[i], output[i],filename[i],input_size);
        if(ret)
            return ret;
    }
    
    return 0;
}

/* Is the same of the fourth function, but in this case in a single file there are multiple instances per line
 * 
Inputs:
 * 
 *             @ float** input:= the float vector where will be stored the input from the files, dimension: n_filesxinput_size 
 *             @ float** output:= the float vector where will be stored the output from the files, dimension: n_filesxinput_size
 *             @ char* filename:= the filename
 *             @ int input_size:= the size of the input. (is not required output size, 'cause the end of the line will handle this factor)
 * */        
int multiple_instance_single_file_parser(float** input, float** output,char* filename,int input_size){
    char* ksource;//ksource
    int size = 0, counter2 = 0, z,counter3 = 0;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return ret;
        
    char* temp2 = (char*)malloc(sizeof(char)*2);// temp2 = (256)
    for(z = 0; z < size; z++){
        
        if(ksource[z] != '\n'){
            temp2[0] = ksource[z];
            temp2[1] = '\0';
            if(counter2 < input_size)
                input[counter3][counter2]=atof(temp2);
            else
                output[counter3][counter2-input_size]=atof(temp2);
            counter2++;
        }
        
        
        else{
            counter3++;
            counter2 = 0;
        }
    
    }
    
    free(ksource);
    free(temp2);
    
    return 0;
}

// number_of_fcls;number_of_cls;number_of_total_cls;number_of_rls\n
// flc\n
// param1;param2,...,paramn\n
// cl
// param1;param2,...,paramn\n
// rcl
// param1;param2,...,paramn\n
model* parse_model_file(char* filename){
    char* ksource;//ksource
    int size = 0,i,counter,flag, flag_n,counter_int;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return NULL;
    
    int n_fcls = 0, n_cls = 0, n_rls = 0, n_total_cls = 0;
    int c_fcls = 0, c_cls = 0;
    
    float* temp_float = (float*)malloc(sizeof(float)*256);
    int* cl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag2= (int*)malloc(sizeof(int)*1024);
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    
    char* res = "residual";
    char* conv = "convolutional";
    char* rconv = "rconvolutional";
    char* full = "fully-connected";
    
    model* m;
    fcl** fcls = NULL;
    cl** cls = NULL;
    rl** rls = NULL;
    for(i = 0,counter=0, counter_int = 0,flag = 0,flag_n=0; i < size; i++){
        
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp2[counter] = ksource[i];
            counter++;
        }
        
        else if(ksource[i] == ';'){
            temp2[counter] = '\0';
            if (!flag_n){
                if(!strcmp(conv,temp2)){
                    n_cls++;
                    n_total_cls++;
                }
                else if(!strcmp(full,temp2))
                    n_fcls++;
                else if(!strcmp(rconv,temp2))
                    n_total_cls++;
                else if(!strcmp(res,temp2))
                    n_rls++;
            }
            
            else{
                temp_float[counter_int] = atof(temp2);
                counter_int++;
            }
            
            counter = 0;
        }
        
        
        
        else if(ksource[i] == '\n'){
            if(!flag_n){
                if(n_fcls)
                    fcls = (fcl**)malloc(sizeof(fcl*)*n_fcls);
                if(n_total_cls)
                    cls = (cl**)malloc(sizeof(cl*)*n_total_cls);
                if(n_rls)
                    rls = (rl**)malloc(sizeof(rl*)*n_rls);
            }
            
            else{
                if(!flag){
                    if(!strcmp(conv,temp2)){
                        flag = 1;
                        cl_flag[c_cls] = flag;
                    }
                    else if(!strcmp(full,temp2))
                        flag = 2;
                    else if(!strcmp(rconv,temp2)){
                        flag = 3;
                        cl_flag[c_cls] = flag;
                    }
                }
                
                else if(flag == 2){
                    fcls[c_fcls] = fully_connected((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9]);
                    c_fcls++;
                    flag = 0;
                }
                else if(flag == 1){
                    cls[c_cls] = convolutional((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    c_cls++;
                    flag = 0;
                }
                else if(flag == 3){
                    cls[c_cls] = convolutional((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    rl_flag[c_cls] = (int)temp_float[24];
                    rl_flag2[c_cls] = (int)temp_float[25];
                    c_cls++;    
                    flag = 0;
                }
                
                
            }
            counter_int = 0;
            counter = 0;
            flag_n = 1;
        }
    
    }
    
    cl** real_cls = NULL;
    if(n_cls)
     real_cls = (cl**)malloc(sizeof(cl*)*n_cls);
    cl** cl_handler = NULL;
    int j,k,counter_rl, prev_rl = -1;
    //printf("total_cls: %d\n",n_total_cls);
    for(counter_rl = 0, flag = 0, counter = 0,counter_int = 0, i = 0; i < n_total_cls; i++){
        if(cl_flag[i] == 1){
            if(flag == 1){
                cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                for(k = j; k < j+counter_int; k++){
                    cl_handler[k-j] = cls[k];
                }
                rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                counter_rl++;
            }
            real_cls[counter] = cls[i];
            counter++;
            flag = 0;
            counter_int = 0;
        }
        else{
            if(!flag){
                prev_rl = rl_flag2[i];
                j = i;
                flag = 1;
            }
            else{
                if(prev_rl != rl_flag2[i]){
                    cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                    for(k = j; k < j+counter_int; k++){
                        cl_handler[k-j] = cls[k];
                    }
                    rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                    rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                    counter_rl++;
                    j = i;
                    counter_int = 0;
                    prev_rl = rl_flag2[i];
                }
            }
            counter_int++;
        }
    }
    
    if(flag == 1){
        cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
        for(k = j; k < j+counter_int; k++){
            cl_handler[k-j] = cls[k];
        }
        rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
        rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
    }
    
    //printf("%d\n",n_fcls);
    m = network(n_fcls+n_total_cls,n_rls,n_cls,n_fcls,rls,real_cls,fcls); 
    free(ksource);
    free(temp2);
    free(temp_float);
    free(cl_flag);
    free(rl_flag);
    free(rl_flag2);
    free(cls);
    return m;
    
}
// number_of_fcls;number_of_cls;number_of_total_cls;number_of_rls\n
// flc\n
// param1;param2,...,paramn\n
// cl
// param1;param2,...,paramn\n
// rcl
// param1;param2,...,paramn\n
model* parse_model_without_learning_parameters_file(char* filename){
    char* ksource;//ksource
    int size = 0,i,counter,flag, flag_n,counter_int;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return NULL;
    
    int n_fcls = 0, n_cls = 0, n_rls = 0, n_total_cls = 0;
    int c_fcls = 0, c_cls = 0;
    
    float* temp_float = (float*)malloc(sizeof(float)*256);
    int* cl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag2= (int*)malloc(sizeof(int)*1024);
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    
    char* res = "residual";
    char* conv = "convolutional";
    char* rconv = "rconvolutional";
    char* full = "fully-connected";
    
    model* m;
    fcl** fcls = NULL;
    cl** cls = NULL;
    rl** rls = NULL;
    for(i = 0,counter=0, counter_int = 0,flag = 0,flag_n=0; i < size; i++){
        
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp2[counter] = ksource[i];
            counter++;
        }
        
        else if(ksource[i] == ';'){
            temp2[counter] = '\0';
            if (!flag_n){
                if(!strcmp(conv,temp2)){
                    n_cls++;
                    n_total_cls++;
                }
                else if(!strcmp(full,temp2))
                    n_fcls++;
                else if(!strcmp(rconv,temp2))
                    n_total_cls++;
                else if(!strcmp(res,temp2))
                    n_rls++;
            }
            
            else{
                temp_float[counter_int] = atof(temp2);
                counter_int++;
            }
            
            counter = 0;
        }
        
        
        
        else if(ksource[i] == '\n'){
            if(!flag_n){
                if(n_fcls)
                    fcls = (fcl**)malloc(sizeof(fcl*)*n_fcls);
                if(n_total_cls)
                    cls = (cl**)malloc(sizeof(cl*)*n_total_cls);
                if(n_rls)
                    rls = (rl**)malloc(sizeof(rl*)*n_rls);
            }
            
            else{
                if(!flag){
                    if(!strcmp(conv,temp2)){
                        flag = 1;
                        cl_flag[c_cls] = flag;
                    }
                    else if(!strcmp(full,temp2))
                        flag = 2;
                    else if(!strcmp(rconv,temp2)){
                        flag = 3;
                        cl_flag[c_cls] = flag;
                    }
                }
                
                else if(flag == 2){
                    fcls[c_fcls] = fully_connected_without_learning_parameters((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9]);
                    c_fcls++;
                    flag = 0;
                }
                else if(flag == 1){
                    cls[c_cls] = convolutional_without_learning_parameters((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    c_cls++;
                    flag = 0;
                }
                else if(flag == 3){
                    cls[c_cls] = convolutional_without_learning_parameters((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    rl_flag[c_cls] = (int)temp_float[24];
                    rl_flag2[c_cls] = (int)temp_float[25];
                    c_cls++;    
                    flag = 0;
                }
                
                
            }
            counter_int = 0;
            counter = 0;
            flag_n = 1;
        }
    
    }
    
    cl** real_cls = NULL;
    if(n_cls)
     real_cls = (cl**)malloc(sizeof(cl*)*n_cls);
    cl** cl_handler = NULL;
    int j,k,counter_rl, prev_rl = -1;
    for(counter_rl = 0, flag = 0, counter = 0,counter_int = 0, i = 0; i < n_total_cls; i++){
        if(cl_flag[i] == 1){
            if(flag == 1){
                cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                for(k = j; k < j+counter_int; k++){
                    cl_handler[k-j] = cls[k];
                }
                rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                counter_rl++;
            }
            real_cls[counter] = cls[i];
            counter++;
            flag = 0;
            counter_int = 0;
        }
        else{
            if(!flag){
                prev_rl = rl_flag2[i];
                j = i;
                flag = 1;
            }
            else{
                if(prev_rl != rl_flag2[i]){
                    cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                    for(k = j; k < j+counter_int; k++){
                        cl_handler[k-j] = cls[k];
                    }
                    rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                    rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                    counter_rl++;
                    j = i;
                    counter_int = 0;
                    prev_rl = rl_flag2[i];
                }
            }
            counter_int++;
        }
    }
    
    if(flag == 1){
        cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
        for(k = j; k < j+counter_int; k++){
            cl_handler[k-j] = cls[k];
        }
        rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
        rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
    }
    
    
    m = network(n_fcls+n_total_cls,n_rls,n_cls,n_fcls,rls,real_cls,fcls); 
    free(ksource);
    free(temp2);
    free(temp_float);
    free(cl_flag);
    free(rl_flag);
    free(rl_flag2);
    free(cls);
    return m;
    
}
// number_of_fcls;number_of_cls;number_of_total_cls;number_of_rls\n
// flc\n
// param1;param2,...,paramn\n
// cl
// param1;param2,...,paramn\n
// rcl
// param1;param2,...,paramn\n
model* parse_model_without_arrays_file(char* filename){
    char* ksource;//ksource
    int size = 0,i,counter,flag, flag_n,counter_int;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return NULL;
    
    int n_fcls = 0, n_cls = 0, n_rls = 0, n_total_cls = 0;
    int c_fcls = 0, c_cls = 0;
    
    float* temp_float = (float*)malloc(sizeof(float)*256);
    int* cl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag2= (int*)malloc(sizeof(int)*1024);
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    
    char* res = "residual";
    char* conv = "convolutional";
    char* rconv = "rconvolutional";
    char* full = "fully-connected";
    
    model* m;
    fcl** fcls = NULL;
    cl** cls = NULL;
    rl** rls = NULL;
    for(i = 0,counter=0, counter_int = 0,flag = 0,flag_n=0; i < size; i++){
        
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp2[counter] = ksource[i];
            counter++;
        }
        
        else if(ksource[i] == ';'){
            temp2[counter] = '\0';
            if (!flag_n){
                if(!strcmp(conv,temp2)){
                    n_cls++;
                    n_total_cls++;
                }
                else if(!strcmp(full,temp2))
                    n_fcls++;
                else if(!strcmp(rconv,temp2))
                    n_total_cls++;
                else if(!strcmp(res,temp2))
                    n_rls++;
            }
            
            else{
                temp_float[counter_int] = atof(temp2);
                counter_int++;
            }
            
            counter = 0;
        }
        
        
        
        else if(ksource[i] == '\n'){
            if(!flag_n){
                if(n_fcls)
                    fcls = (fcl**)malloc(sizeof(fcl*)*n_fcls);
                if(n_total_cls)
                    cls = (cl**)malloc(sizeof(cl*)*n_total_cls);
                if(n_rls)
                    rls = (rl**)malloc(sizeof(rl*)*n_rls);
            }
            
            else{
                if(!flag){
                    if(!strcmp(conv,temp2)){
                        flag = 1;
                        cl_flag[c_cls] = flag;
                    }
                    else if(!strcmp(full,temp2))
                        flag = 2;
                    else if(!strcmp(rconv,temp2)){
                        flag = 3;
                        cl_flag[c_cls] = flag;
                    }
                }
                
                else if(flag == 2){
                    fcls[c_fcls] = fully_connected_without_arrays((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9]);
                    c_fcls++;
                    flag = 0;
                }
                else if(flag == 1){
                    cls[c_cls] = convolutional_without_arrays((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    c_cls++;
                    flag = 0;
                }
                else if(flag == 3){
                    cls[c_cls] = convolutional_without_arrays((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    rl_flag[c_cls] = (int)temp_float[24];
                    rl_flag2[c_cls] = (int)temp_float[25];
                    c_cls++;    
                    flag = 0;
                }
                
                
            }
            counter_int = 0;
            counter = 0;
            flag_n = 1;
        }
    
    }
    
    cl** real_cls = NULL;
    if(n_cls)
     real_cls = (cl**)malloc(sizeof(cl*)*n_cls);
    cl** cl_handler = NULL;
    int j,k,counter_rl, prev_rl = -1;
    for(counter_rl = 0, flag = 0, counter = 0,counter_int = 0, i = 0; i < n_total_cls; i++){
        if(cl_flag[i] == 1){
            if(flag == 1){
                cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                for(k = j; k < j+counter_int; k++){
                    cl_handler[k-j] = cls[k];
                }
                rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                counter_rl++;
            }
            real_cls[counter] = cls[i];
            counter++;
            flag = 0;
            counter_int = 0;
        }
        else{
            if(!flag){
                prev_rl = rl_flag2[i];
                j = i;
                flag = 1;
            }
            else{
                if(prev_rl != rl_flag2[i]){
                    cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                    for(k = j; k < j+counter_int; k++){
                        cl_handler[k-j] = cls[k];
                    }
                    rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                    rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                    counter_rl++;
                    j = i;
                    counter_int = 0;
                    prev_rl = rl_flag2[i];
                }
            }
            counter_int++;
        }
    }
    
    if(flag == 1){
        cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
        for(k = j; k < j+counter_int; k++){
            cl_handler[k-j] = cls[k];
        }
        rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
        rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
    }
    
    
    m = network(n_fcls+n_total_cls,n_rls,n_cls,n_fcls,rls,real_cls,fcls); 
    free(ksource);
    free(temp2);
    free(temp_float);
    free(cl_flag);
    free(rl_flag);
    free(rl_flag2);
    free(cls);
    return m;
    
}
// number_of_fcls;number_of_cls;number_of_total_cls;number_of_rls\n
// flc\n
// param1;param2,...,paramn\n
// cl
// param1;param2,...,paramn\n
// rcl
// param1;param2,...,paramn\n
model* parse_model_str(char* ksource, int size){
    int i,counter,flag, flag_n,counter_int;
    
    int n_fcls = 0, n_cls = 0, n_rls = 0, n_total_cls = 0;
    int c_fcls = 0, c_cls = 0;
    
    float* temp_float = (float*)malloc(sizeof(float)*256);
    int* cl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag2= (int*)malloc(sizeof(int)*1024);
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    
    char* res = "residual";
    char* conv = "convolutional";
    char* rconv = "rconvolutional";
    char* full = "fully-connected";
    
    model* m;
    fcl** fcls = NULL;
    cl** cls = NULL;
    rl** rls = NULL;
    for(i = 0,counter=0, counter_int = 0,flag = 0,flag_n=0; i < size; i++){
        
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp2[counter] = ksource[i];
            counter++;
        }
        
        else if(ksource[i] == ';'){
            temp2[counter] = '\0';
            if (!flag_n){
                if(!strcmp(conv,temp2)){
                    n_cls++;
                    n_total_cls++;
                }
                else if(!strcmp(full,temp2))
                    n_fcls++;
                else if(!strcmp(rconv,temp2))
                    n_total_cls++;
                else if(!strcmp(res,temp2))
                    n_rls++;
            }
            
            else{
                temp_float[counter_int] = atof(temp2);
                counter_int++;
            }
            
            counter = 0;
        }
        
        
        
        else if(ksource[i] == '\n'){
            if(!flag_n){
                if(n_fcls)
                    fcls = (fcl**)malloc(sizeof(fcl*)*n_fcls);
                if(n_total_cls)
                    cls = (cl**)malloc(sizeof(cl*)*n_total_cls);
                if(n_rls)
                    rls = (rl**)malloc(sizeof(rl*)*n_rls);
            }
            
            else{
                if(!flag){
                    if(!strcmp(conv,temp2)){
                        flag = 1;
                        cl_flag[c_cls] = flag;
                    }
                    else if(!strcmp(full,temp2))
                        flag = 2;
                    else if(!strcmp(rconv,temp2)){
                        flag = 3;
                        cl_flag[c_cls] = flag;
                    }
                }
                
                else if(flag == 2){
                    fcls[c_fcls] = fully_connected((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9]);
                    c_fcls++;
                    flag = 0;
                }
                else if(flag == 1){
                    cls[c_cls] = convolutional((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    c_cls++;
                    flag = 0;
                }
                else if(flag == 3){
                    cls[c_cls] = convolutional((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    rl_flag[c_cls] = (int)temp_float[24];
                    rl_flag2[c_cls] = (int)temp_float[25];
                    c_cls++;    
                    flag = 0;
                }
                
                
            }
            counter_int = 0;
            counter = 0;
            flag_n = 1;
        }
    
    }
    
    cl** real_cls = NULL;
    if(n_cls)
     real_cls = (cl**)malloc(sizeof(cl*)*n_cls);
    cl** cl_handler = NULL;
    int j,k,counter_rl, prev_rl = -1;
    for(counter_rl = 0, flag = 0, counter = 0,counter_int = 0, i = 0; i < n_total_cls; i++){
        if(cl_flag[i] == 1){
            if(flag == 1){
                cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                for(k = j; k < j+counter_int; k++){
                    cl_handler[k-j] = cls[k];
                }
                rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                counter_rl++;
            }
            real_cls[counter] = cls[i];
            counter++;
            flag = 0;
            counter_int = 0;
        }
        else{
            if(!flag){
                prev_rl = rl_flag2[i];
                j = i;
                flag = 1;
            }
            else{
                if(prev_rl != rl_flag2[i]){
                    cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                    for(k = j; k < j+counter_int; k++){
                        cl_handler[k-j] = cls[k];
                    }
                    rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                    rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                    counter_rl++;
                    j = i;
                    counter_int = 0;
                    prev_rl = rl_flag2[i];
                }
            }
            counter_int++;
        }
    }
    
    if(flag == 1){
        cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
        for(k = j; k < j+counter_int; k++){
            cl_handler[k-j] = cls[k];
        }
        rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
        rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
    }
    
    
    m = network(n_fcls+n_total_cls,n_rls,n_cls,n_fcls,rls,real_cls,fcls);
    free(temp2);
    free(temp_float);
    free(cl_flag);
    free(rl_flag);
    free(rl_flag2);
    free(cls);
    return m;
    
}
// number_of_fcls;number_of_cls;number_of_total_cls;number_of_rls\n
// flc\n
// param1;param2,...,paramn\n
// cl
// param1;param2,...,paramn\n
// rcl
// param1;param2,...,paramn\n
model* parse_model_without_learning_parameters_str(char* ksource, int size){
    int i,counter,flag, flag_n,counter_int;
    
    int n_fcls = 0, n_cls = 0, n_rls = 0, n_total_cls = 0;
    int c_fcls = 0, c_cls = 0;
    
    float* temp_float = (float*)malloc(sizeof(float)*256);
    int* cl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag2= (int*)malloc(sizeof(int)*1024);
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    
    char* res = "residual";
    char* conv = "convolutional";
    char* rconv = "rconvolutional";
    char* full = "fully-connected";
    
    model* m;
    fcl** fcls = NULL;
    cl** cls = NULL;
    rl** rls = NULL;
    for(i = 0,counter=0, counter_int = 0,flag = 0,flag_n=0; i < size; i++){
        
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp2[counter] = ksource[i];
            counter++;
        }
        
        else if(ksource[i] == ';'){
            temp2[counter] = '\0';
            if (!flag_n){
                if(!strcmp(conv,temp2)){
                    n_cls++;
                    n_total_cls++;
                }
                else if(!strcmp(full,temp2))
                    n_fcls++;
                else if(!strcmp(rconv,temp2))
                    n_total_cls++;
                else if(!strcmp(res,temp2))
                    n_rls++;
            }
            
            else{
                temp_float[counter_int] = atof(temp2);
                counter_int++;
            }
            
            counter = 0;
        }
        
        
        
        else if(ksource[i] == '\n'){
            if(!flag_n){
                if(n_fcls)
                    fcls = (fcl**)malloc(sizeof(fcl*)*n_fcls);
                if(n_total_cls)
                    cls = (cl**)malloc(sizeof(cl*)*n_total_cls);
                if(n_rls)
                    rls = (rl**)malloc(sizeof(rl*)*n_rls);
            }
            
            else{
                if(!flag){
                    if(!strcmp(conv,temp2)){
                        flag = 1;
                        cl_flag[c_cls] = flag;
                    }
                    else if(!strcmp(full,temp2))
                        flag = 2;
                    else if(!strcmp(rconv,temp2)){
                        flag = 3;
                        cl_flag[c_cls] = flag;
                    }
                }
                
                else if(flag == 2){
                    fcls[c_fcls] = fully_connected_without_learning_parameters((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9]);
                    c_fcls++;
                    flag = 0;
                }
                else if(flag == 1){
                    cls[c_cls] = convolutional_without_learning_parameters((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    c_cls++;
                    flag = 0;
                }
                else if(flag == 3){
                    cls[c_cls] = convolutional_without_learning_parameters((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    rl_flag[c_cls] = (int)temp_float[24];
                    rl_flag2[c_cls] = (int)temp_float[25];
                    c_cls++;    
                    flag = 0;
                }
                
                
            }
            counter_int = 0;
            counter = 0;
            flag_n = 1;
        }
    
    }
    
    cl** real_cls = NULL;
    if(n_cls)
     real_cls = (cl**)malloc(sizeof(cl*)*n_cls);
    cl** cl_handler = NULL;
    int j,k,counter_rl, prev_rl = -1;
    for(counter_rl = 0, flag = 0, counter = 0,counter_int = 0, i = 0; i < n_total_cls; i++){
        if(cl_flag[i] == 1){
            if(flag == 1){
                cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                for(k = j; k < j+counter_int; k++){
                    cl_handler[k-j] = cls[k];
                }
                rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                counter_rl++;
            }
            real_cls[counter] = cls[i];
            counter++;
            flag = 0;
            counter_int = 0;
        }
        else{
            if(!flag){
                prev_rl = rl_flag2[i];
                j = i;
                flag = 1;
            }
            else{
                if(prev_rl != rl_flag2[i]){
                    cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                    for(k = j; k < j+counter_int; k++){
                        cl_handler[k-j] = cls[k];
                    }
                    rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                    rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                    counter_rl++;
                    j = i;
                    counter_int = 0;
                    prev_rl = rl_flag2[i];
                }
            }
            counter_int++;
        }
    }
    
    if(flag == 1){
        cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
        for(k = j; k < j+counter_int; k++){
            cl_handler[k-j] = cls[k];
        }
        rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
        rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
    }
    
    
    m = network(n_fcls+n_total_cls,n_rls,n_cls,n_fcls,rls,real_cls,fcls);
    free(temp2);
    free(temp_float);
    free(cl_flag);
    free(rl_flag);
    free(rl_flag2);
    free(cls);
    return m;
    
}
// number_of_fcls;number_of_cls;number_of_total_cls;number_of_rls\n
// flc\n
// param1;param2,...,paramn\n
// cl
// param1;param2,...,paramn\n
// rcl
// param1;param2,...,paramn\n
model* parse_model_without_arrays_str(char* ksource, int size){
    int i,counter,flag, flag_n,counter_int;
    
    int n_fcls = 0, n_cls = 0, n_rls = 0, n_total_cls = 0;
    int c_fcls = 0, c_cls = 0;
    
    float* temp_float = (float*)malloc(sizeof(float)*256);
    int* cl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag= (int*)malloc(sizeof(int)*1024);
    int* rl_flag2= (int*)malloc(sizeof(int)*1024);
    char* temp2 = (char*)malloc(sizeof(char)*256);// temp2 = (256)
    
    char* res = "residual";
    char* conv = "convolutional";
    char* rconv = "rconvolutional";
    char* full = "fully-connected";
    
    model* m = NULL;
    fcl** fcls = NULL;
    cl** cls = NULL;
    rl** rls = NULL;
    int something_went_wrong = 0;
    for(i = 0,counter=0, counter_int = 0,flag = 0,flag_n=0; i < size; i++){
        
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp2[counter] = ksource[i];
            counter++;
        }
        
        else if(ksource[i] == ';'){
            temp2[counter] = '\0';
            if (!flag_n){
                if(!strcmp(conv,temp2)){
                    n_cls++;
                    n_total_cls++;
                }
                else if(!strcmp(full,temp2))
                    n_fcls++;
                else if(!strcmp(rconv,temp2))
                    n_total_cls++;
                else if(!strcmp(res,temp2))
                    n_rls++;
            }
            
            else{
                temp_float[counter_int] = atof(temp2);
                counter_int++;
            }
            
            counter = 0;
        }
        
        
        
        else if(ksource[i] == '\n'){
            if(!flag_n){
                if(n_fcls)
                    fcls = (fcl**)malloc(sizeof(fcl*)*n_fcls);
                if(n_total_cls)
                    cls = (cl**)malloc(sizeof(cl*)*n_total_cls);
                if(n_rls)
                    rls = (rl**)malloc(sizeof(rl*)*n_rls);
            }
            
            else{
                if(!flag){
                    if(!strcmp(conv,temp2)){
                        flag = 1;
                        cl_flag[c_cls] = flag;
                    }
                    else if(!strcmp(full,temp2))
                        flag = 2;
                    else if(!strcmp(rconv,temp2)){
                        flag = 3;
                        cl_flag[c_cls] = flag;
                    }
                }
                
                else if(flag == 2){
                    fcls[c_fcls] = fully_connected_without_arrays((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9]);
                    if(fcls[c_fcls] == NULL)
                        something_went_wrong = 1;
                    c_fcls++;
                    flag = 0;
                }
                else if(flag == 1){
                    cls[c_cls] = convolutional_without_arrays((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    if(cls[c_cls] == NULL)
                        something_went_wrong = 1;
                    c_cls++;
                    flag = 0;
                }
                else if(flag == 3){
                    cls[c_cls] = convolutional_without_arrays((int)temp_float[0],(int)temp_float[1],(int)temp_float[2],(int)temp_float[3],(int)temp_float[4],temp_float[5],(int)temp_float[6],(int)temp_float[7],(int)temp_float[8],(int)temp_float[9],(int)temp_float[10],(int)temp_float[11],(int)temp_float[12],(int)temp_float[13],(int)temp_float[14],(int)temp_float[15],(int)temp_float[16],(int)temp_float[17],(int)temp_float[18],(int)temp_float[19],(int)temp_float[20],(int)temp_float[21],(int)temp_float[22],(int)temp_float[23]);
                    if(cls[c_cls] == NULL)
                        something_went_wrong = 1;
                    rl_flag[c_cls] = (int)temp_float[24];
                    rl_flag2[c_cls] = (int)temp_float[25];
                    c_cls++;    
                    flag = 0;
                }
                
                
            }
            counter_int = 0;
            counter = 0;
            flag_n = 1;
        }
    
    }
    
    
    
    
    cl** real_cls = NULL;
    if(something_went_wrong){
        for(i = 0; i < n_total_cls; i++){
            free_convolutional_without_arrays(cls[i]);
        }
        for(i = 0; i < n_fcls; i++){
            free_fully_connected_without_arrays(fcls[i]);
        }
        free(fcls);
        free(rls);
    }
    else{
        if(n_cls)
         real_cls = (cl**)malloc(sizeof(cl*)*n_cls);
        cl** cl_handler = NULL;
        int j,k,counter_rl, prev_rl = -1;
        for(counter_rl = 0, flag = 0, counter = 0,counter_int = 0, i = 0; i < n_total_cls; i++){
            if(cl_flag[i] == 1){
                if(flag == 1){
                    cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                    for(k = j; k < j+counter_int; k++){
                        cl_handler[k-j] = cls[k];
                    }
                    rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                    if(rls[counter_rl] == NULL)
                            something_went_wrong = 1;
                    else
                        rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                    counter_rl++;
                }
                real_cls[counter] = cls[i];
                counter++;
                flag = 0;
                counter_int = 0;
            }
            else{
                if(!flag){
                    prev_rl = rl_flag2[i];
                    j = i;
                    flag = 1;
                }
                else{
                    if(prev_rl != rl_flag2[i]){
                        cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
                        for(k = j; k < j+counter_int; k++){
                            cl_handler[k-j] = cls[k];
                        }
                        rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
                        if(rls[counter_rl] == NULL)
                            something_went_wrong = 1;
                        else
                            rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
                        counter_rl++;
                        j = i;
                        counter_int = 0;
                        prev_rl = rl_flag2[i];
                    }
                }
                counter_int++;
            }
        }
        
        if(flag == 1){
            cl_handler = (cl**)malloc(sizeof(cl*)*counter_int);
            for(k = j; k < j+counter_int; k++){
                cl_handler[k-j] = cls[k];
            }
            rls[counter_rl] = residual(cls[j]->channels,cls[j]->input_rows,cls[j]->input_cols,counter_int,cl_handler);
            if(rls[counter_rl] == NULL)
                something_went_wrong = 1;
            else
                rls[counter_rl]->cl_output->activation_flag = rl_flag[j];
        }
        if(something_went_wrong){
            
            for(i = 0; i < n_rls; i++){
                if(rls[i] != NULL){
                    free_convolutional_without_learning_parameters(rls[i]->cl_output);
                    free(rls[i]);
                }
            }
            for(i = 0; i < n_total_cls; i++){
                free_convolutional_without_arrays(cls[i]);
            }
            for(i = 0; i < n_fcls; i++){
                free_fully_connected_without_arrays(fcls[i]);
            }
            free(rls);
            free(real_cls);
            free(fcls);
        }
        else
            m = network(n_fcls+n_total_cls,n_rls,n_cls,n_fcls,rls,real_cls,fcls);
    }
    free(temp2);
    free(temp_float);
    free(cl_flag);
    free(rl_flag);
    free(rl_flag2);
    free(cls);
    return m;
    
}





dueling_categorical_dqn* parse_dueling_categorical_dqn_file(char* filename){
    char* ksource;//ksource
    int size = 0,i;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return NULL;
    
    
    
    char* shared_hidden_layers = "shared_hidden_layers";
    char* v_hidden_layers = "v_hidden_layers";
    char* v_linear_last_layer = "v_linear_last_layer";
    char* a_hidden_layers = "a_hidden_layers";
    char* a_linear_last_layer = "a_linear_last_layer";
    
    char* temp = (char*)malloc(sizeof(char)*256);
    int shar,v_hid,v_lin,a_hid,a_lin,counter,lines;
    int index_shar,index_v, index_vv, index_a, index_aa, len_shar, len_v, len_vv, len_a, len_aa;
    for(shar = 0, v_hid = 0,v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else
                    shar = 1;
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else
                        v_hid = 1;
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else
                            v_lin = 1;
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else
                                a_hid = 1;
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else
                                    a_lin = 1;
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    if(!shar || !v_hid || !v_lin || !a_hid || !a_lin){
        fprintf(stderr,"Error: something went wrong during parsing!\n");
        free(ksource);
        free(temp);
        exit(1);
    }
    
    for(index_shar = 0, index_v = 0, index_vv = 0, index_a = 0, index_aa = 0, len_shar = 0, len_v = 0, len_vv = 0, len_a = 0, len_aa = 0, shar = 0, v_hid = 0, v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else{
                    shar = 1;
                    index_shar = i+2;
                }
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else{
                        v_hid = 1;
                        len_shar = i-strlen(v_hidden_layers)-index_shar;
                        index_v = i+2;
                    }
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else{
                            v_lin = 1;
                            len_v = i-strlen(v_linear_last_layer)-index_v;
                            index_vv = i+2;
                        }
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else{
                                a_hid = 1;
                                len_vv = i-strlen(a_hidden_layers)-index_vv;
                                index_a = i+2;
                            }
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else{
                                    a_lin = 1;
                                    len_a = i-strlen(a_linear_last_layer)-index_a;
                                    index_aa = i+2;
                                    len_aa = size-index_aa;
                                    break;
                                }
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(v_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    free(temp);
    model* shared_hidden_layers_dqn = parse_model_str(ksource+index_shar,len_shar);
    model* v_hidden_layers_dqn = parse_model_str(ksource+index_v,len_v);
    model* v_linear_last_layer_dqn = parse_model_str(ksource+index_vv,len_vv);
    model* a_hidden_layers_dqn = parse_model_str(ksource+index_a,len_a);
    model* a_linear_last_layer_dqn = parse_model_str(ksource+index_aa,len_aa);
    if(a_linear_last_layer_dqn->output_dimension%v_linear_last_layer_dqn->output_dimension){
        fprintf(stderr,"Error: something went wrong defining the output sizes!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init(get_input_layer_size(shared_hidden_layers_dqn),a_linear_last_layer_dqn->output_dimension/v_linear_last_layer_dqn->output_dimension,v_linear_last_layer_dqn->output_dimension,-10,10,shared_hidden_layers_dqn,v_hidden_layers_dqn,a_hidden_layers_dqn,v_linear_last_layer_dqn,a_linear_last_layer_dqn);
    free(ksource);
    return dqn;
    
}

dueling_categorical_dqn* parse_dueling_categorical_dqn_str(char* str, int size){
    char* ksource = str;
    int i;
    
    
    char* shared_hidden_layers = "shared_hidden_layers";
    char* v_hidden_layers = "v_hidden_layers";
    char* v_linear_last_layer = "v_linear_last_layer";
    char* a_hidden_layers = "a_hidden_layers";
    char* a_linear_last_layer = "a_linear_last_layer";
    
    char* temp = (char*)malloc(sizeof(char)*256);
    int shar,v_hid,v_lin,a_hid,a_lin,counter,lines;
    int index_shar,index_v, index_vv, index_a, index_aa, len_shar, len_v, len_vv, len_a, len_aa;
    for(shar = 0, v_hid = 0,v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else
                    shar = 1;
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else
                        v_hid = 1;
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else
                            v_lin = 1;
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else
                                a_hid = 1;
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else
                                    a_lin = 1;
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    if(!shar || !v_hid || !v_lin || !a_hid || !a_lin){
        fprintf(stderr,"Error: something went wrong during parsing!\n");
        free(ksource);
        free(temp);
        exit(1);
    }
    
    for(index_shar = 0, index_v = 0, index_vv = 0, index_a = 0, index_aa = 0, len_shar = 0, len_v = 0, len_vv = 0, len_a = 0, len_aa = 0, shar = 0, v_hid = 0, v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else{
                    shar = 1;
                    index_shar = i+2;
                }
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else{
                        v_hid = 1;
                        len_shar = i-strlen(v_hidden_layers)-index_shar;
                        index_v = i+2;
                    }
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else{
                            v_lin = 1;
                            len_v = i-strlen(v_linear_last_layer)-index_v;
                            index_vv = i+2;
                        }
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else{
                                a_hid = 1;
                                len_vv = i-strlen(a_hidden_layers)-index_vv;
                                index_a = i+2;
                            }
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else{
                                    a_lin = 1;
                                    len_a = i-strlen(a_linear_last_layer)-index_a;
                                    index_aa = i+2;
                                    len_aa = size-index_aa;
                                    break;
                                }
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(v_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    free(temp);
    model* shared_hidden_layers_dqn = parse_model_str(ksource+index_shar,len_shar);
    model* v_hidden_layers_dqn = parse_model_str(ksource+index_v,len_v);
    model* v_linear_last_layer_dqn = parse_model_str(ksource+index_vv,len_vv);
    model* a_hidden_layers_dqn = parse_model_str(ksource+index_a,len_a);
    model* a_linear_last_layer_dqn = parse_model_str(ksource+index_aa,len_aa);
    if(a_linear_last_layer_dqn->output_dimension%v_linear_last_layer_dqn->output_dimension){
        fprintf(stderr,"Error: something went wrong defining the output sizes!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init(get_input_layer_size(shared_hidden_layers_dqn),a_linear_last_layer_dqn->output_dimension/v_linear_last_layer_dqn->output_dimension,v_linear_last_layer_dqn->output_dimension,-10,10,shared_hidden_layers_dqn,v_hidden_layers_dqn,a_hidden_layers_dqn,v_linear_last_layer_dqn,a_linear_last_layer_dqn);
    return dqn;
    
}
dueling_categorical_dqn* parse_dueling_categorical_dqn_without_learning_parameters_file(char* filename){
    char* ksource;//ksource
    int size = 0,i;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return NULL;
    
    
    
    char* shared_hidden_layers = "shared_hidden_layers";
    char* v_hidden_layers = "v_hidden_layers";
    char* v_linear_last_layer = "v_linear_last_layer";
    char* a_hidden_layers = "a_hidden_layers";
    char* a_linear_last_layer = "a_linear_last_layer";
    
    char* temp = (char*)malloc(sizeof(char)*256);
    int shar,v_hid,v_lin,a_hid,a_lin,counter,lines;
    int index_shar,index_v, index_vv, index_a, index_aa, len_shar, len_v, len_vv, len_a, len_aa;
    for(shar = 0, v_hid = 0,v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else
                    shar = 1;
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else
                        v_hid = 1;
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else
                            v_lin = 1;
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else
                                a_hid = 1;
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else
                                    a_lin = 1;
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    if(!shar || !v_hid || !v_lin || !a_hid || !a_lin){
        fprintf(stderr,"Error: something went wrong during parsing!\n");
        free(ksource);
        free(temp);
        exit(1);
    }
    
    for(index_shar = 0, index_v = 0, index_vv = 0, index_a = 0, index_aa = 0, len_shar = 0, len_v = 0, len_vv = 0, len_a = 0, len_aa = 0, shar = 0, v_hid = 0, v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else{
                    shar = 1;
                    index_shar = i+2;
                }
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else{
                        v_hid = 1;
                        len_shar = i-strlen(v_hidden_layers)-index_shar;
                        index_v = i+2;
                    }
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else{
                            v_lin = 1;
                            len_v = i-strlen(v_linear_last_layer)-index_v;
                            index_vv = i+2;
                        }
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else{
                                a_hid = 1;
                                len_vv = i-strlen(a_hidden_layers)-index_vv;
                                index_a = i+2;
                            }
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else{
                                    a_lin = 1;
                                    len_a = i-strlen(a_linear_last_layer)-index_a;
                                    index_aa = i+2;
                                    len_aa = size-index_aa;
                                    break;
                                }
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(v_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    free(temp);
    model* shared_hidden_layers_dqn = parse_model_without_learning_parameters_str(ksource+index_shar,len_shar);
    model* v_hidden_layers_dqn = parse_model_without_learning_parameters_str(ksource+index_v,len_v);
    model* v_linear_last_layer_dqn = parse_model_without_learning_parameters_str(ksource+index_vv,len_vv);
    model* a_hidden_layers_dqn = parse_model_without_learning_parameters_str(ksource+index_a,len_a);
    model* a_linear_last_layer_dqn = parse_model_without_learning_parameters_str(ksource+index_aa,len_aa);
    if(a_linear_last_layer_dqn->output_dimension%v_linear_last_layer_dqn->output_dimension){
        fprintf(stderr,"Error: something went wrong defining the output sizes!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init(get_input_layer_size(shared_hidden_layers_dqn),a_linear_last_layer_dqn->output_dimension/v_linear_last_layer_dqn->output_dimension,v_linear_last_layer_dqn->output_dimension,-10,10,shared_hidden_layers_dqn,v_hidden_layers_dqn,a_hidden_layers_dqn,v_linear_last_layer_dqn,a_linear_last_layer_dqn);
    free(ksource);
    return dqn;
    
}

dueling_categorical_dqn* parse_dueling_categorical_dqn_without_learning_parameters_str(char* str, int size){
    char* ksource = str;
    int i;
    
    
    char* shared_hidden_layers = "shared_hidden_layers";
    char* v_hidden_layers = "v_hidden_layers";
    char* v_linear_last_layer = "v_linear_last_layer";
    char* a_hidden_layers = "a_hidden_layers";
    char* a_linear_last_layer = "a_linear_last_layer";
    
    char* temp = (char*)malloc(sizeof(char)*256);
    int shar,v_hid,v_lin,a_hid,a_lin,counter,lines;
    int index_shar,index_v, index_vv, index_a, index_aa, len_shar, len_v, len_vv, len_a, len_aa;
    for(shar = 0, v_hid = 0,v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else
                    shar = 1;
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else
                        v_hid = 1;
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else
                            v_lin = 1;
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else
                                a_hid = 1;
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else
                                    a_lin = 1;
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    if(!shar || !v_hid || !v_lin || !a_hid || !a_lin){
        fprintf(stderr,"Error: something went wrong during parsing!\n");
        free(ksource);
        free(temp);
        exit(1);
    }
    
    for(index_shar = 0, index_v = 0, index_vv = 0, index_a = 0, index_aa = 0, len_shar = 0, len_v = 0, len_vv = 0, len_a = 0, len_aa = 0, shar = 0, v_hid = 0, v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else{
                    shar = 1;
                    index_shar = i+2;
                }
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else{
                        v_hid = 1;
                        len_shar = i-strlen(v_hidden_layers)-index_shar;
                        index_v = i+2;
                    }
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else{
                            v_lin = 1;
                            len_v = i-strlen(v_linear_last_layer)-index_v;
                            index_vv = i+2;
                        }
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else{
                                a_hid = 1;
                                len_vv = i-strlen(a_hidden_layers)-index_vv;
                                index_a = i+2;
                            }
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else{
                                    a_lin = 1;
                                    len_a = i-strlen(a_linear_last_layer)-index_a;
                                    index_aa = i+2;
                                    len_aa = size-index_aa;
                                    break;
                                }
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(v_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    free(temp);
    model* shared_hidden_layers_dqn = parse_model_without_learning_parameters_str(ksource+index_shar,len_shar);
    model* v_hidden_layers_dqn = parse_model_without_learning_parameters_str(ksource+index_v,len_v);
    model* v_linear_last_layer_dqn = parse_model_without_learning_parameters_str(ksource+index_vv,len_vv);
    model* a_hidden_layers_dqn = parse_model_without_learning_parameters_str(ksource+index_a,len_a);
    model* a_linear_last_layer_dqn = parse_model_without_learning_parameters_str(ksource+index_aa,len_aa);
    if(a_linear_last_layer_dqn->output_dimension%v_linear_last_layer_dqn->output_dimension){
        fprintf(stderr,"Error: something went wrong defining the output sizes!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init(get_input_layer_size(shared_hidden_layers_dqn),a_linear_last_layer_dqn->output_dimension/v_linear_last_layer_dqn->output_dimension,v_linear_last_layer_dqn->output_dimension,-10,10,shared_hidden_layers_dqn,v_hidden_layers_dqn,a_hidden_layers_dqn,v_linear_last_layer_dqn,a_linear_last_layer_dqn);
    return dqn;
    
}
dueling_categorical_dqn* parse_dueling_categorical_dqn_without_arrays_file(char* filename){
    char* ksource;//ksource
    int size = 0,i;
    int ret = read_file_in_char_vector(&ksource,filename,&size);
    
    if(ret)
        return NULL;
    
    
    
    char* shared_hidden_layers = "shared_hidden_layers";
    char* v_hidden_layers = "v_hidden_layers";
    char* v_linear_last_layer = "v_linear_last_layer";
    char* a_hidden_layers = "a_hidden_layers";
    char* a_linear_last_layer = "a_linear_last_layer";
    
    char* temp = (char*)malloc(sizeof(char)*256);
    int shar,v_hid,v_lin,a_hid,a_lin,counter,lines;
    int index_shar,index_v, index_vv, index_a, index_aa, len_shar, len_v, len_vv, len_a, len_aa;
    for(shar = 0, v_hid = 0,v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else
                    shar = 1;
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else
                        v_hid = 1;
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else
                            v_lin = 1;
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else
                                a_hid = 1;
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else
                                    a_lin = 1;
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    if(!shar || !v_hid || !v_lin || !a_hid || !a_lin){
        fprintf(stderr,"Error: something went wrong during parsing!\n");
        free(ksource);
        free(temp);
        exit(1);
    }
    
    for(index_shar = 0, index_v = 0, index_vv = 0, index_a = 0, index_aa = 0, len_shar = 0, len_v = 0, len_vv = 0, len_a = 0, len_aa = 0, shar = 0, v_hid = 0, v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    exit(1);
                }
                else{
                    shar = 1;
                    index_shar = i+2;
                }
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else{
                        v_hid = 1;
                        len_shar = i-strlen(v_hidden_layers)-index_shar;
                        index_v = i+2;
                    }
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else{
                            v_lin = 1;
                            len_v = i-strlen(v_linear_last_layer)-index_v;
                            index_vv = i+2;
                        }
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else{
                                a_hid = 1;
                                len_vv = i-strlen(a_hidden_layers)-index_vv;
                                index_a = i+2;
                            }
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else{
                                    a_lin = 1;
                                    len_a = i-strlen(a_linear_last_layer)-index_a;
                                    index_aa = i+2;
                                    len_aa = size-index_aa;
                                    break;
                                }
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(v_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    exit(1);
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    free(temp);
    model* shared_hidden_layers_dqn = parse_model_without_arrays_str(ksource+index_shar,len_shar);
    model* v_hidden_layers_dqn = parse_model_without_arrays_str(ksource+index_v,len_v);
    model* v_linear_last_layer_dqn = parse_model_without_arrays_str(ksource+index_vv,len_vv);
    model* a_hidden_layers_dqn = parse_model_without_arrays_str(ksource+index_a,len_a);
    model* a_linear_last_layer_dqn = parse_model_without_arrays_str(ksource+index_aa,len_aa);
    if(a_linear_last_layer_dqn->output_dimension%v_linear_last_layer_dqn->output_dimension){
        fprintf(stderr,"Error: something went wrong defining the output sizes!\n");
        exit(1);
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init_without_arrays(get_input_layer_size(shared_hidden_layers_dqn),a_linear_last_layer_dqn->output_dimension/v_linear_last_layer_dqn->output_dimension,v_linear_last_layer_dqn->output_dimension,-10,10,shared_hidden_layers_dqn,v_hidden_layers_dqn,a_hidden_layers_dqn,v_linear_last_layer_dqn,a_linear_last_layer_dqn);
    free(ksource);
    return dqn;
    
}

dueling_categorical_dqn* parse_dueling_categorical_dqn_without_arrays_str(char* str, int size){
    char* ksource = str;
    int i;
    
    
    char* shared_hidden_layers = "shared_hidden_layers";
    char* v_hidden_layers = "v_hidden_layers";
    char* v_linear_last_layer = "v_linear_last_layer";
    char* a_hidden_layers = "a_hidden_layers";
    char* a_linear_last_layer = "a_linear_last_layer";
    
    char* temp = (char*)malloc(sizeof(char)*256);
    int shar,v_hid,v_lin,a_hid,a_lin,counter,lines;
    int index_shar,index_v, index_vv, index_a, index_aa, len_shar, len_v, len_vv, len_a, len_aa;
    for(shar = 0, v_hid = 0,v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    return NULL;
                }
                else
                    shar = 1;
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else
                        v_hid = 1;
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else
                            v_lin = 1;
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else
                                a_hid = 1;
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else
                                    a_lin = 1;
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    return NULL;
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    if(!shar || !v_hid || !v_lin || !a_hid || !a_lin){
        fprintf(stderr,"Error: something went wrong during parsing!\n");
        free(ksource);
        free(temp);
        return NULL;
    }
    
    for(index_shar = 0, index_v = 0, index_vv = 0, index_a = 0, index_aa = 0, len_shar = 0, len_v = 0, len_vv = 0, len_a = 0, len_aa = 0, shar = 0, v_hid = 0, v_lin = 0, a_hid = 0, a_lin = 0,counter = 0, i = 0,lines = 0; i < size; i++){
        if(ksource[i] != ';' && ksource[i] != '\n'){
            temp[counter] = ksource[i];
            counter++;
        }
        else if(ksource[i] == ';'){
            
            temp[counter] = '\0';
            if(!shar){
                if(i >= size-2 || lines || ksource[i+1] != '\n' || memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers))){
                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                    free(ksource);
                    free(temp);
                    return NULL;
                }
                else{
                    shar = 1;
                    index_shar = i+2;
                }
            }
            
            else{
                if(!v_hid){
                    if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_hidden_layers,temp,strlen(v_hidden_layers))){
                        v_hid = 0;
                    }
                    else{
                        v_hid = 1;
                        len_shar = i-strlen(v_hidden_layers)-index_shar;
                        index_v = i+2;
                    }
                }
                else{
                    if(!v_lin){
                        if(i >= size-2 || ksource[i+1] != '\n' || memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer))){
                            v_lin = 0;
                        }
                        else{
                            v_lin = 1;
                            len_v = i-strlen(v_linear_last_layer)-index_v;
                            index_vv = i+2;
                        }
                    }
                    else{
                        if(!a_hid){
                            if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_hidden_layers,temp,strlen(a_hidden_layers))){
                                a_hid = 0;
                            }
                            else{
                                a_hid = 1;
                                len_vv = i-strlen(a_hidden_layers)-index_vv;
                                index_a = i+2;
                            }
                        }
                        
                        else{
                            if(!a_lin){
                                if(i >= size-2 || ksource[i+1] != '\n' || memcmp(a_linear_last_layer,temp,strlen(a_linear_last_layer))){
                                    a_lin = 0;
                                }
                                else{
                                    a_lin = 1;
                                    len_a = i-strlen(a_linear_last_layer)-index_a;
                                    index_aa = i+2;
                                    len_aa = size-index_aa;
                                    break;
                                }
                            }
                            else{
                                if(!memcmp(shared_hidden_layers,temp,strlen(shared_hidden_layers)) || !memcmp(v_hidden_layers,temp,strlen(v_hidden_layers)) || !memcmp(v_linear_last_layer,temp,strlen(v_linear_last_layer)) || !memcmp(a_hidden_layers,temp,strlen(a_hidden_layers)) || !memcmp(a_linear_last_layer,temp,strlen(v_linear_last_layer))){
                                    fprintf(stderr,"Error: something went wrong during parsing!\n");
                                    free(ksource);
                                    free(temp);
                                    return NULL;
                                }
                            }
                        }
                    }
                }
            }
            counter = 0;
            lines++;
        }
        
        else if(ksource[i] == '\n'){
            counter = 0;
            lines++;
        }
    }
    
    free(temp);
    model* shared_hidden_layers_dqn = parse_model_without_arrays_str(ksource+index_shar,len_shar);
    model* v_hidden_layers_dqn = parse_model_without_arrays_str(ksource+index_v,len_v);
    model* v_linear_last_layer_dqn = parse_model_without_arrays_str(ksource+index_vv,len_vv);
    model* a_hidden_layers_dqn = parse_model_without_arrays_str(ksource+index_a,len_a);
    model* a_linear_last_layer_dqn = parse_model_without_arrays_str(ksource+index_aa,len_aa);
    if(a_linear_last_layer_dqn->output_dimension%v_linear_last_layer_dqn->output_dimension){
        free_model_without_arrays(shared_hidden_layers_dqn);
        free_model_without_arrays(v_hidden_layers_dqn);
        free_model_without_arrays(a_hidden_layers_dqn);
        free_model_without_arrays(v_linear_last_layer_dqn);
        free_model_without_arrays(a_linear_last_layer_dqn);
        return NULL;
    }
    dueling_categorical_dqn* dqn = dueling_categorical_dqn_init_without_arrays(get_input_layer_size(shared_hidden_layers_dqn),a_linear_last_layer_dqn->output_dimension/v_linear_last_layer_dqn->output_dimension,v_linear_last_layer_dqn->output_dimension,-10,10,shared_hidden_layers_dqn,v_hidden_layers_dqn,a_hidden_layers_dqn,v_linear_last_layer_dqn,a_linear_last_layer_dqn);
    if(dqn == NULL){
        free_model_without_arrays(shared_hidden_layers_dqn);
        free_model_without_arrays(v_hidden_layers_dqn);
        free_model_without_arrays(a_hidden_layers_dqn);
        free_model_without_arrays(v_linear_last_layer_dqn);
        free_model_without_arrays(a_linear_last_layer_dqn);
    }
    return dqn;
    
}
