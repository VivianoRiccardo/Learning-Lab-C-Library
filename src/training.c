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
        heavy_save_model(t->m[i],count);
    }
    for(i = 0;i < t->n_r; i++,count++){
        heavy_save_rmodel(t->r[i],count);
    }
}

/* This function loads a network model from a .bin file with name file
 * 
 * Input:
 * 
 *             @ char* file:= the binary file from which the model will be loaded
 * 
 * */
training* load_training(int n, int n_files){
    
    int i,j;
    if(n_files < 0){
        fprintf(stderr,"Error: number of files < 0\n");
        exit(1);
    }
    
    char** file = get_files(n,n_files);
    FILE* fr = fopen(file[0],"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",file[0]);
        exit(1);
    }
    
    int epochs,instance,n_char_size,n_int_size,n_float_size,n_m, n_r, n_float, n_int, n_char;
    char** chars = NULL;
    int** ints = NULL;
    float** floats = NULL;
    i = fread(&n_m,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    
    i = fread(&n_r,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    
    i = fread(&epochs,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    
    i = fread(&instance,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    
    i = fread(&n_char,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    i = fread(&n_int,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    i = fread(&n_float,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    i = fread(&n_char_size,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    i = fread(&n_int_size,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    i = fread(&n_float_size,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading the training_structure\n");
        exit(1);
    }
    
    if(n_char!= 0)
        chars = (char**)malloc(sizeof(char*)*n_char);
    for(j = 0; j < n_char; j++){
        chars[j] = (char*)malloc(sizeof(char)*n_char_size);
        i = fread(chars[j],sizeof(char)*n_char_size,1,fr);
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading the training_structure\n");
            exit(1);
        }
    }
    
    if(n_int!= 0)
        ints = (int**)malloc(sizeof(int*)*n_int);
    
    for(j = 0; j < n_int; j++){
        ints[j] = (int*)malloc(sizeof(int)*n_int_size);
        i = fread(ints[j],sizeof(int)*n_int_size,1,fr);
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading the training_structure\n");
            exit(1);
        }
    }
    if(n_float!= 0)
        floats = (float**)malloc(sizeof(float*)*n_float);
    
    for(j = 0; j < n_float; j++){
        floats[j] = (float*)malloc(sizeof(float)*n_float_size);
        i = fread(floats[j],sizeof(float)*n_float_size,1,fr);
        if(i != 1){
            fprintf(stderr,"Error: an error occurred loading the training_structure\n");
            exit(1);
        }
    }
    
    
    i = fclose(fr);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",file[0]);
        exit(1);
    }
    
    model** m = NULL;
    rmodel** r = NULL;
    
    if(n_m != 0){
        m = (model**)malloc(sizeof(model*)*n_m);
    }
    
    if(n_r != 0){
        r = (rmodel**)malloc(sizeof(rmodel*)*n_r);
    }
    
    int count = 1;
    for(i = 0;i < n_m; i++, count++){
        m[i] = heavy_load_model(file[count]);
    }
    for(i = 0;i < n_r; i++,count++){
        r[i] = heavy_load_rmodel(file[count]);
    }
    
    for(i = 0; i < n_files; i++){
        free(file[i]);
    }
    free(file);
    
    return get_training(chars,ints,floats,m,r,epochs,n_char_size,n_float_size,n_int_size,instance,n_m,n_r,n_char,n_float,n_int);    
    
}
