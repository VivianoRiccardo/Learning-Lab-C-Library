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

#ifndef __LLAB_DT_H__
#define __LLAB_DT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONDITION_A(x,y) (x > y)
#define CONDITION_B(x,y) (x >= y)
#define CONDITION_C(x,y) (x == y)
#define CONDITION_D(x,y) (x <= y)
#define CONDITION_E(x,y) (x < y)
#define CONDITION_F(x,y) (!strcmp(x,y))


typedef struct decision_tree {
    int number_instances;// Number of total instances
    int char_feature_number, char_labels_number; // feature_number = 0 no char features, labels_number = 0 no char labels
    int int_feature_number, int_labels_number; // feature_number = 0 no int features, labels_number = 0 no int labels
    int float_feature_number, float_labels_number; // feature_number = 0 no char features, labels_number = 0 no char labels
    int char_condition_flag;//if the son is created with a char condition on char features (indicates on which char feature it's the condition)
    int int_condition_flag;//if the son is created with a float condition on float features (indicates on which char feature it's the condition)
    int float_condition_flag;//if the son is created with an int condition on int features (indicates on which int feature it's the condition)
    int unwanted_char_size;
    int unwanted_float_size;
    int unwanted_int_size;
    int char_second_dimension_max_size;
    char** char_features;//(number_instances*different_features)*char_second_dimension_max_size
    int* int_features;//number_instances*different_features
    float* float_features;//number_instances*different_features
    char** char_labels;//(number_instances*different_labels)*char_second_dimension_max_size
    int* int_labels;//(number_instances*different_labels)
    float* float_labels;//(number_instances*different_labels)
    float impurity;
    float conditional_threshold;
    char* conditional_string;
    char** unwanted_conditional_char_list//unwanted_char_size*char_second_dimension_max_size
    int* unwanted_conditional_int_list//unwanted_int_size
    float* unwanted_conditional_float_list//unwanted_float_size
    decision_tree** sons;
} decision_tree;

#endif __LLAB_DT_H__
