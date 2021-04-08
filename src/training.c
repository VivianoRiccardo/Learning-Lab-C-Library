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


training* get_training(char** chars, int** ints, float** floats, model** m, rmodel** r,int epochs, int n_char_size, int n_float_size, int n_int_size, int instance, int n_m, int n_r, int n_char, int n_float, int n_int){
    training* t = (training*)malloc(sizeof(training));
    t->floats = floats;
    t->ints = ints;
    t->chars = chars;
    t->m = m;
    t->r = r;
    t->n_m = n_m;
    t->n_r = n_r;
    t->n_char = n_char;
    t->n_int = n_int;
    t->n_float = n_float;
    t->epochs = epochs;
    t->instance = instance;
    t->n_char_size = n_char_size;
    t->n_int_size = n_int_size;
    t->n_float_size = n_float_size;
    return t;
}


void save_training(training* t, int n){
    int i,j, count = n;
    
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* temp = ".bin";
    s = itoa(count,s);
    s = strcat(s,temp);
    
    fw = fopen(s,"w");
    
    i = fwrite(&t->n_m,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_r,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->epochs,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->instance,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_char,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_int,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_float,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_char_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_int_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_float_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    for(j = 0; j < t->n_char; j++){
        i = fwrite(t->chars[j],sizeof(char)*t->n_char_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the training structure\n");
            exit(1);
        }
    }
    
    for(j = 0; j < t->n_int; j++){
        i = fwrite(t->ints[j],sizeof(char)*t->n_int_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the training structure\n");
            exit(1);
        }
    }
    
    for(j = 0; j < t->n_float; j++){
        i = fwrite(t->floats[j],sizeof(char)*t->n_float_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the training structure\n");
            exit(1);
        }
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    free(s);
    
    count++;
    
    for(i = 0;i < t->n_m; i++,count++){
        save_model(t->m[i],count);
    }
    for(i = 0;i < t->n_r; i++,count++){
        heavy_save_rmodel(t->r[i],count);
    }
}


void standard_save_training(training* t, int n){
    int i,j, count = n;
    
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* temp = ".bin";
    s = itoa(count,s);
    s = strcat(s,temp);
    
    fw = fopen(s,"w");
    
    i = fwrite(&t->n_m,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_r,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->epochs,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->instance,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_char,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_int,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_float,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_char_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_int_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    
    i = fwrite(&t->n_float_size,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving the training structure\n");
        exit(1);
    }
    for(j = 0; j < t->n_char; j++){
        i = fwrite(t->chars[j],sizeof(char)*t->n_char_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the training structure\n");
            exit(1);
        }
    }
    
    for(j = 0; j < t->n_int; j++){
        i = fwrite(t->ints[j],sizeof(char)*t->n_int_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the training structure\n");
            exit(1);
        }
    }
    
    for(j = 0; j < t->n_float; j++){
        i = fwrite(t->floats[j],sizeof(char)*t->n_float_size,1,fw);
    
        if(i != 1){
            fprintf(stderr,"Error: an error occurred saving the training structure\n");
            exit(1);
        }
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    free(s);
    
    count++;
    
    for(i = 0;i < t->n_m; i++,count++){
        save_model(t->m[i],count);
    }
    for(i = 0;i < t->n_r; i++,count++){
        save_rmodel(t->r[i],count);
    }
}


