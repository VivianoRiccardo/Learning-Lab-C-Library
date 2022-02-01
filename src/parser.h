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

#ifndef __PARSER_H__
#define __PARSER_H__

int single_instance_single_csv_file_parser(float* input, float* output,char* filename,int input_size);
int single_instance_multiple_csv_file_parser(float** input, float** output,char** filename,int input_size, int n_files);
int multiple_instance_single_csv_file_parser(float** input, float** output,char* filename,int input_size);
int single_instance_single_file_parser(float* input, float* output,char* filename,int input_size);
int single_instance_multiple_file_parser(float** input, float** output,char** filename,int input_size, int n_files);
int multiple_instance_single_file_parser(float** input, float** output,char* filename,int input_size);
model* parse_model_file(char* filename);
float** get_inputs_from_multiple_instances_single_char_binary_file_with_single_softmax_output(char* filename,int input_dimension, int instances);
float** get_outputs_from_multiple_instances_single_char_binary_file_with_single_softmax_output(char* filename,int input_dimension,int output_dimension, int instances);
model* parse_model_without_learning_parameters_file(char* filename);
model* parse_model_without_arrays_file(char* filename);
model* parse_model_str(char* ksource, int size);
model* parse_model_without_learning_parameters_str(char* ksource, int size);
model* parse_model_without_arrays_str(char* ksource, int size);
dueling_categorical_dqn* parse_dueling_categorical_dqn_file(char* filename);
dueling_categorical_dqn* parse_dueling_categorical_dqn_str(char* str, int size);
dueling_categorical_dqn* parse_dueling_categorical_dqn_without_learning_parameters_str(char* str, int size);
dueling_categorical_dqn* parse_dueling_categorical_dqn_without_learning_parameters_file(char* filename);
dueling_categorical_dqn* parse_dueling_categorical_dqn_without_arrays_str(char* str, int size);
dueling_categorical_dqn* parse_dueling_categorical_dqn_without_arrays_file(char* filename);

#endif
