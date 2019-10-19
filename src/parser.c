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
 *             @ float** output:= the float vector where will be stored the output from the files, dimension: n_filesxinput_size
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



