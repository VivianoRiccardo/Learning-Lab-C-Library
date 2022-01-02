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


/* This function return the decoder layer of a transformer
 * 
 * 
 * Inputs:
 * 
 *                 @ int input_dimension:= the dimension of the input of the linear layer after the first attention mechanism aka output dimension of the attention mechanism
 *                 @ int n_head1:= how many heads in the first attention mechanism
 *                 @ int n_head2:= the number of heads in the second attention mechanism
 *                 @ int residual_flag1:= if there is a residual connection after the first attention mechanism
 *                 @ int normalization_flag1:= if there is a normalization after the first attention mechanism
 *                 @ int residual_flag2:= if there is a reisudal connection after the second attention mechanism
 *                 @ int normalization_flag2:= if there is normalization after the second attention mechanism
 *                 @ int residual_flag3:= if there is a reisudal connection after the feed forward layers (given by model)
 *                 @ int normalization_flag3:= you got the point...
 *                 @ int attention_flag1:= for the first attention mechanism (STANDARD_ATTENTION or MASKED_ATTENTION)
 *                 @ int attention_flag2:= for the second attention mechanism (STANDARD_ATTENTION or MASKED_ATTENTION)
 *                 @ int encoder_input_dimension:= the dimension of the input of the linear layer after the second attention, aka the output dimension of the second attention mechanism
 *                 @ model* m:= the model after the second attention mechanism (usually 1 feed forward with relu + 1 feedforward with no activation functions)
 *                 @ model* linear_after_activation1:= the linear layer after the first attention     
 *                 @ model* linear_after_activation2:= the linear layer after the second attention     
 *                 @ model** q:= the queries of the first and second attention mechanism dimension: (n_head1+n_head2)
 *                 @ model** k:= the keys of the first and second attention mechanism dimension: (n_head1+n_head2)
 *                 @ model** v:= the values of the first and second attention mechanism dimension: (n_head1+n_head2)
 *                 @ scaled_l2_norm** l2:= can be 3,2,1 according to the normalization flags
 *                 @ int decoder_k_embedding:= the embedding dimension of the decoder q and k
 *                 @ int decoder_v_embedding:= the embedding dimension of the decoder v
 *                 @ int encoder_k_embedding:= the embedding dimension of the encoder q and k
 *                 @ int encoder_v_embedding:= the embedding dimension of the encoder v
 * */
 
transformer_decoder* transformer_decoder_layer(int input_dimension, int left_dimension, int n_head1, int n_head2, int residual_flag1, int normalization_flag1, int residual_flag2, int normalization_flag2, int residual_flag3, int normalization_flag3, int attention_flag1, int attention_flag2, int encoder_input_dimension, model* m,model* linear_after_attention1,model* linear_after_attention2, model** q,model** k, model** v, scaled_l2_norm** l2, int decoder_k_embedding, int decoder_v_embedding, int encoder_k_embedding, int encoder_v_embedding){
    if(n_head1 <= 0 || input_dimension <= 0 || encoder_input_dimension <= 0 || n_head2 <= 0){
        fprintf(stderr,"Error: n_head1, input_dimension, encoder_input_dimensionm n_head2 must be > 0\n");
        exit(1);
    }
    
    if(input_dimension%n_head1){
        fprintf(stderr,"Error: n_head1 must divide perfectly input_dimension\n");
        exit(1);
    }
    
    if(m == NULL){
        fprintf(stderr,"Error: the model can't be set to NULL there must be something after the attention mechanism!\n");
        exit(1);
    }
    
    if(linear_after_attention1 == NULL){
        fprintf(stderr,"Error: can't have no linearity after the attention! Read the paper noob!\n");
        exit(1);
    }
    
    if(linear_after_attention2 == NULL){
        fprintf(stderr,"Error: can't have no linearity after the attention! Read the paper noob!\n");
        exit(1);
    }
    
    if(input_dimension%(n_head1*decoder_v_embedding)){
        fprintf(stderr,"Error: your input_dimension is not perfectly divisible by n_head1*decoder_v_embedding!\n");
        exit(1);
    }
    
    int count = 0;
    if(normalization_flag1 == SCALED_L2_NORMALIZATION)
        count=1;
    transformer_encoder* e = transformer_encoder_layer(q+n_head1, k+n_head1, v+n_head1,m,linear_after_attention2, &l2[count],encoder_input_dimension,n_head2,residual_flag2,normalization_flag2,residual_flag3,normalization_flag3,attention_flag2,encoder_k_embedding,encoder_v_embedding);
    transformer_decoder* d = (transformer_decoder*)malloc(sizeof(transformer_decoder));
    if(residual_flag1 != TRANSFORMER_RESIDUAL)
        residual_flag1 = TRANSFORMER_NO_RESIDUAL;
    if(normalization_flag1 != SCALED_L2_NORMALIZATION)
        normalization_flag1 = NO_NORMALIZATION;
    d->linear_after_attention = linear_after_attention1;
    d->input_dimension = input_dimension;
    d->n_head = n_head1;
    d->attention_flag = attention_flag1;
    d->residual_flag = residual_flag1;
    d->normalization_flag = normalization_flag1;
    d->dimension = input_dimension/(n_head1*decoder_v_embedding);
    d->encoder_input_dimension = encoder_input_dimension;
    d->left_dimension = left_dimension;
    d->n_l2 = 0;
    d->k_embedding = decoder_k_embedding;
    d->v_embedding = decoder_v_embedding;
    if(normalization_flag1 == SCALED_L2_NORMALIZATION)
        d->n_l2++;
    d->l2 = l2;
    d->e = e;
    d->q = q;
    d->k = k;
    d->v = v;
    d->q_error = (float*)calloc(input_dimension,sizeof(float));
    d->k_error = (float*)calloc(input_dimension,sizeof(float));
    d->v_error = (float*)calloc(input_dimension,sizeof(float));
    d->score_matrix = (float*)calloc(d->dimension*d->dimension,sizeof(float));
    d->score_matrix_softmax = (float*)calloc(d->dimension*d->dimension,sizeof(float));
    d->score_matrix_softmax_error = (float*)calloc(d->dimension*d->dimension,sizeof(float));
    d->score_matrix_error = (float*)calloc(d->dimension*d->dimension,sizeof(float));
    d->attention_output = (float*)calloc(input_dimension,sizeof(float));
    d->incoming_input = (float*)calloc(left_dimension,sizeof(float));
    d->incoming_input_error = (float*)calloc(left_dimension,sizeof(float));
    if(residual_flag1 == TRANSFORMER_RESIDUAL){
        d->residual1_output = (float*)calloc(d->linear_after_attention->output_dimension,sizeof(float));
        d->residual1_output_error = (float*)calloc(d->linear_after_attention->output_dimension,sizeof(float));
    }
    
    return d;
}

/* This function deallocates the space allocated by a transformer_decoder structure
 * 
 * 
 * Inputs:
 *             
 * 
 *                 @ transformer_decoder* d:= the decoder we must to deallocate
 * */
void free_transformer_decoder_layer(transformer_decoder* d){
    if(d == NULL)
        return;
    free_transformer_wrapped_encoder_layer(d->e);
    int i;
    for(i = 0; i < d->n_head; i++){
        free_model(d->q[i]);
        free_model(d->k[i]);
        free_model(d->v[i]);
    }
    free(d->q);
    free(d->k);
    free(d->v);
    free_model(d->linear_after_attention);
    if(d->normalization_flag == SCALED_L2_NORMALIZATION)
        free_scaled_l2_normalization_layer(d->l2[0]);
    free(d->l2);
    
    free(d->q_error);
    free(d->k_error);
    free(d->v_error);
    free(d->score_matrix);
    free(d->score_matrix_softmax);
    free(d->score_matrix_softmax_error);
    free(d->score_matrix_error);
    free(d->attention_output);
    free(d->incoming_input);
    free(d->incoming_input_error);
    if(d->residual_flag == TRANSFORMER_RESIDUAL){
        free(d->residual1_output);
        free(d->residual1_output_error);
    }
    
    free(d);
    
    return;
}
/* This function deallocates the space allocated by a transformer_decoder structure
 * 
 * 
 * Inputs:
 *             
 * 
 *                 @ transformer_decoder* d:= the decoder we must to deallocate
 * */
void free_transformer_decoder_layer_without_learning_parameters(transformer_decoder* d){
    if(d == NULL)
        return;
    free_transformer_wrapped_encoder_layer_without_learning_parameters(d->e);
    int i;
    for(i = 0; i < d->n_head*3; i++){
        free_model_without_learning_parameters(d->q[i]);
        free_model_without_learning_parameters(d->k[i]);
        free_model_without_learning_parameters(d->v[i]);
    }
    free(d->q);
    free(d->k);
    free(d->v);
    free_model_without_learning_parameters(d->linear_after_attention);
    if(d->normalization_flag == SCALED_L2_NORMALIZATION)
        free_scaled_l2_normalization_layer(d->l2[0]);
    free(d->l2);
    
    free(d->q_error);
    free(d->k_error);
    free(d->v_error);
    free(d->score_matrix);
    free(d->score_matrix_softmax);
    free(d->score_matrix_softmax_error);
    free(d->score_matrix_error);
    free(d->attention_output);
    free(d->incoming_input);
    free(d->incoming_input_error);
    if(d->residual_flag == TRANSFORMER_RESIDUAL){
        free(d->residual1_output);
        free(d->residual1_output_error);
    }
    
    free(d);
    
    return;
}


/* This function saves a transformer decoder strcture into a file
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer_decoder* t:= the transformer decoder that must be saved
 *                 @ int n:= the file name in integer, example n = 0 the filename will be 0.bin
 * 
 * */
void save_transformer_decoder(transformer_decoder* t, int n){
    if(t == NULL)
        return;
    int i;
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
    i = fwrite(&t->k_embedding,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    
    i = fwrite(&t->v_embedding,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    
    
    i = fwrite(&t->input_dimension,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->left_dimension,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->n_head,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->attention_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->residual_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->normalization_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->encoder_input_dimension,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    
    if(i != 0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    free(s);
    
    
    for(i = 0; i < t->n_l2; i++){
        save_scaled_l2_norm(t->l2[i],n);
    }
    
    for(i = 0; i < t->n_head; i++){
        save_model(t->q[i],n);
        save_model(t->k[i],n);
        save_model(t->v[i],n);
    }
    
    save_model(t->linear_after_attention,n);
    save_transformer_encoder(t->e,n);
    
    return;
    
}

/* This function loads a transformer decoder structure from a file fr
 * 
 * Inputs:
 *         
 *                 @ FILE* fr:= the file from which must be loaded
 * 
 * */
transformer_decoder* load_transformer_decoder(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    int input_dimension = 0,left_dimension = 0,n_head = 0,residual_flag = 0,normalization_flag = 0, attention_flag = 0, encoder_input_dimension = 0;
    int k_embedding,v_embedding;
    
    
    i = fread(&k_embedding,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&v_embedding,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&input_dimension,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&left_dimension,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&n_head,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&attention_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&residual_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&normalization_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&encoder_input_dimension,sizeof(int),1,fr);
    
    
    
    model** q = (model**)malloc(sizeof(model*)*n_head);
    model** k = (model**)malloc(sizeof(model*)*n_head);
    model** v = (model**)malloc(sizeof(model*)*n_head);
    
    scaled_l2_norm** l2 = NULL;
    int count = 0;
    if(normalization_flag == SCALED_L2_NORMALIZATION)
        count++;
   

    if(count)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*count);
    for(i = 0; i < count; i++){
        l2[i] = load_scaled_l2_norm(fr);
    }
    for(i = 0; i < n_head; i++){
        q[i] = load_model_with_file_already_opened(fr);
        k[i] = load_model_with_file_already_opened(fr);
        v[i] = load_model_with_file_already_opened(fr);
    }
    model* linear_after_attention = load_model_with_file_already_opened(fr);
    transformer_encoder* e = load_transformer_encoder(fr);
    model** total_q = (model**)malloc(sizeof(model*)*(n_head+e->n_head));
    model** total_k = (model**)malloc(sizeof(model*)*(n_head+e->n_head));
    model** total_v = (model**)malloc(sizeof(model*)*(n_head+e->n_head));
    model* m = copy_model(e->m);
    model* mm = copy_model(e->linear_after_attention);
    scaled_l2_norm** total_l2 = NULL;
    count = 0;
    if(normalization_flag == SCALED_L2_NORMALIZATION)
        count++;
    count+=e->n_l2;
    if(count)
        total_l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*count);
    for(i = 0; i < n_head; i++){
        total_q[i] = q[i];
        total_k[i] = k[i];
        total_v[i] = v[i];
    }
    for(i = n_head; i < (n_head+e->n_head); i++){
        total_q[i] = copy_model(e->q[i-n_head]);
        total_k[i] = copy_model(e->k[i-n_head]);
        total_v[i] = copy_model(e->v[i-n_head]);
    }
    
    if(normalization_flag == SCALED_L2_NORMALIZATION)
        count = 1;
    else
        count = 0;
    
    for(i = 0; i < count; i++){
        total_l2[i] = copy_scaled_l2_norm(l2[i]);
        free_scaled_l2_normalization_layer(l2[i]);
    }
    free(l2);
    for(i = count; i < count+e->n_l2; i++){
        total_l2[i] = copy_scaled_l2_norm(e->l2[i-count]);
    }
    
    transformer_decoder* t = transformer_decoder_layer(input_dimension,left_dimension, n_head,e->n_head, residual_flag,normalization_flag,e->residual_flag1,e->normalization_flag1,e->residual_flag2,e->normalization_flag2,attention_flag,e->attention_flag,encoder_input_dimension,m,linear_after_attention,mm,total_q,total_k,total_v,total_l2,k_embedding,v_embedding,e->k_embedding_dimension,e->v_embedding_dimension);
    free_transformer_encoder_layer(e);
    return t;
}

/* this function allocates the space for a new transformer decoder structure that is the exact copy of the 
 * transformer decoder given as input
 * 
 * 
 * Inputs:
 * 
 *             @transformer_decoder* t:= the structure that must be copied
 * 
 * */
transformer_decoder* copy_transformer_decoder(transformer_decoder* t){
    if( t == NULL)
        return NULL;
    model** q = (model**)malloc(sizeof(model*)*(t->n_head+t->e->n_head));
    model** k = (model**)malloc(sizeof(model*)*(t->n_head+t->e->n_head));
    model** v = (model**)malloc(sizeof(model*)*(t->n_head+t->e->n_head));
    scaled_l2_norm** l2 = NULL;
    int i;
    for(i = 0; i < (t->n_head+t->e->n_head); i++){
        q[i] = copy_model(t->q[i]);
        k[i] = copy_model(t->k[i]);
        v[i] = copy_model(t->v[i]);
    }
    
    if(t->n_l2 || t->e->n_l2)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*(t->n_l2+t->e->n_l2));
    for(i = 0; i < t->n_l2+t->e->n_l2; i++){
        l2[i] = copy_scaled_l2_norm(t->l2[i]);
    }
    model* m = copy_model(t->e->m);
    model* mm = copy_model(t->e->linear_after_attention);
    model* linear_after_attention = copy_model(t->linear_after_attention);
    return  transformer_decoder_layer(t->input_dimension,t->left_dimension, t->n_head,t->e->n_head, t->residual_flag,t->normalization_flag,t->e->residual_flag1,t->e->normalization_flag1,t->e->residual_flag2,t->e->normalization_flag2,t->attention_flag,t->e->attention_flag,t->encoder_input_dimension,m,linear_after_attention,mm,q,k,v,l2,t->k_embedding,t->v_embedding,t->e->k_embedding_dimension,t->e->v_embedding_dimension);
}
/* this function allocates the space for a new transformer decoder structure that is the exact copy of the 
 * transformer decoder given as input
 * 
 * 
 * Inputs:
 * 
 *             @transformer_decoder* t:= the structure that must be copied
 * 
 * */
transformer_decoder* copy_transformer_decoder_without_learning_parameters(transformer_decoder* t){
    if( t == NULL)
        return NULL;
    model** q = (model**)malloc(sizeof(model*)*(t->n_head+t->e->n_head));
    model** k = (model**)malloc(sizeof(model*)*(t->n_head+t->e->n_head));
    model** v = (model**)malloc(sizeof(model*)*(t->n_head+t->e->n_head));
    scaled_l2_norm** l2 = NULL;
    int i;
    for(i = 0; i < (t->n_head+t->e->n_head); i++){
        q[i] = copy_model_without_learning_parameters(t->q[i]);
        k[i] = copy_model_without_learning_parameters(t->k[i]);
        v[i] = copy_model_without_learning_parameters(t->v[i]);
    }
    
    if(t->n_l2 || t->e->n_l2)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*(t->n_l2+t->e->n_l2));
    for(i = 0; i < t->n_l2+t->e->n_l2; i++){
        l2[i] = copy_scaled_l2_norm(t->l2[i]);
    }
    model* m = copy_model_without_learning_parameters(t->e->m);
    model* mm = copy_model_without_learning_parameters(t->e->linear_after_attention);
    model* linear_after_attention = copy_model_without_learning_parameters(t->linear_after_attention);
    return  transformer_decoder_layer(t->input_dimension,t->left_dimension, t->n_head,t->e->n_head, t->residual_flag,t->normalization_flag,t->e->residual_flag1,t->e->normalization_flag1,t->e->residual_flag2,t->e->normalization_flag2,t->attention_flag,t->e->attention_flag,t->encoder_input_dimension,m,linear_after_attention,mm,q,k,v,l2,t->k_embedding,t->v_embedding,t->e->k_embedding_dimension,t->e->v_embedding_dimension);
}

/* This function resets the arrays used during the feed forward and backpropagation by the transformer
 * 
 * Inputs:
 * 
 * 
 *                 @transformer_decoder* t:= the transformer decoder structure that must be rests
 * */
void reset_transformer_decoder(transformer_decoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head; i++){
        reset_model(t->q[i]);
        reset_model(t->k[i]);
        reset_model(t->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    
    set_vector_with_value(0,t->incoming_input,t->left_dimension);
    set_vector_with_value(0,t->incoming_input_error,t->left_dimension);
    set_vector_with_value(0,t->q_error,t->input_dimension);
    set_vector_with_value(0,t->k_error,t->input_dimension);
    set_vector_with_value(0,t->v_error,t->input_dimension);
    set_vector_with_value(0,t->attention_output,t->input_dimension);
    set_vector_with_value(0,t->score_matrix,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_error,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax_error,t->dimension*t->dimension);
        

     
    if(t->residual_flag == TRANSFORMER_RESIDUAL){
        set_vector_with_value(0,t->residual1_output,t->linear_after_attention->output_dimension);
        set_vector_with_value(0,t->residual1_output_error,t->linear_after_attention->output_dimension);
    }
    reset_transformer_encoder(t->e);
    reset_model(t->linear_after_attention);
    return;
}
/* This function resets the arrays used during the feed forward and backpropagation by the transformer
 * 
 * Inputs:
 * 
 * 
 *                 @transformer_decoder* t:= the transformer decoder structure that must be rests
 * */
void reset_transformer_decoder_without_learning_parameters(transformer_decoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head; i++){
        reset_model_without_learning_parameters(t->q[i]);
        reset_model_without_learning_parameters(t->k[i]);
        reset_model_without_learning_parameters(t->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    
    set_vector_with_value(0,t->incoming_input,t->left_dimension);
    set_vector_with_value(0,t->incoming_input_error,t->left_dimension);
    set_vector_with_value(0,t->q_error,t->input_dimension);
    set_vector_with_value(0,t->k_error,t->input_dimension);
    set_vector_with_value(0,t->v_error,t->input_dimension);
    set_vector_with_value(0,t->attention_output,t->input_dimension);
    set_vector_with_value(0,t->score_matrix,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_error,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax_error,t->dimension*t->dimension);
        

     
    if(t->residual_flag == TRANSFORMER_RESIDUAL){
        set_vector_with_value(0,t->residual1_output,t->linear_after_attention->output_dimension);
        set_vector_with_value(0,t->residual1_output_error,t->linear_after_attention->output_dimension);
    }
    reset_transformer_encoder_without_learning_parameters(t->e);
    reset_model_without_learning_parameters(t->linear_after_attention);
    return;
}
/* This function resets the arrays used during the feed forward and backpropagation by the transformer
 * 
 * Inputs:
 * 
 * 
 *                 @transformer_decoder* t:= the transformer decoder structure that must be rests
 * */
void reset_transformer_decoder_except_partial_derivatives_and_left_input(transformer_decoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head; i++){
        reset_model_except_partial_derivatives(t->q[i]);
        reset_model_except_partial_derivatives(t->k[i]);
        reset_model_except_partial_derivatives(t->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    
    set_vector_with_value(0,t->incoming_input,t->left_dimension);
    set_vector_with_value(0,t->q_error,t->input_dimension);
    set_vector_with_value(0,t->k_error,t->input_dimension);
    set_vector_with_value(0,t->v_error,t->input_dimension);
    set_vector_with_value(0,t->attention_output,t->input_dimension);
    set_vector_with_value(0,t->score_matrix,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_error,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax_error,t->dimension*t->dimension);
        

     
    if(t->residual_flag == TRANSFORMER_RESIDUAL){
        set_vector_with_value(0,t->residual1_output,t->linear_after_attention->output_dimension);
        set_vector_with_value(0,t->residual1_output_error,t->linear_after_attention->output_dimension);
    }
    reset_transformer_encoder_except_partial_derivatives(t->e);
    reset_model_except_partial_derivatives(t->linear_after_attention);
    return;
}


/* This function does exactly what the function above does but for the fully connected leyers inside
 * the transformer the reset is for the edge popup
 * 
 * Inputs:
 * 
 * 
 *             @transformer_decoder* t:= the transformer decoder that must be reset
 * */
void reset_transformer_decoder_for_edge_popup(transformer_decoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head; i++){
        reset_model_for_edge_popup(t->q[i]);
        reset_model_for_edge_popup(t->k[i]);
        reset_model_for_edge_popup(t->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    
    set_vector_with_value(0,t->incoming_input,t->left_dimension);
    set_vector_with_value(0,t->incoming_input_error,t->left_dimension);
    set_vector_with_value(0,t->q_error,t->input_dimension);
    set_vector_with_value(0,t->k_error,t->input_dimension);
    set_vector_with_value(0,t->v_error,t->input_dimension);
    set_vector_with_value(0,t->attention_output,t->input_dimension);
    set_vector_with_value(0,t->score_matrix,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_error,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax,t->dimension*t->dimension);
    set_vector_with_value(0,t->score_matrix_softmax_error,t->dimension*t->dimension);
        

     
    if(t->residual_flag == TRANSFORMER_RESIDUAL){
        set_vector_with_value(0,t->residual1_output,t->linear_after_attention->output_dimension);
        set_vector_with_value(0,t->residual1_output_error,t->linear_after_attention->output_dimension);
    }
    reset_transformer_encoder_for_edge_popup(t->e);
    reset_model_for_edge_popup(t->linear_after_attention);
    return;
}

/* this function gives the number of bytes more or less occupied by this structure
 * 
 * Inputs:
 * 
 *                 @transformer_decoder* t:= the structure that must be sized
 * */
uint64_t size_of_transformer_decoder(transformer_decoder* t){
    uint64_t sum = 0;
    int i;
    for(i = 0; i < t->n_head; i++){
        sum+=size_of_model(t->q[i]);
        sum+=size_of_model(t->k[i]);
        sum+=size_of_model(t->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        sum+=size_of_scaled_l2_norm(t->l2[i]);
    }
    
    sum+= sizeof(float)*(t->input_dimension*4 + t->dimension*t->dimension*4 +t->left_dimension*2);
    if(t->residual_flag == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2*sizeof(float);
    sum+=size_of_transformer_encoder(t->e) + size_of_model(t->linear_after_attention);
    return sum;    
}
/* this function gives the number of bytes more or less occupied by this structure
 * 
 * Inputs:
 * 
 *                 @transformer_decoder* t:= the structure that must be sized
 * */
uint64_t size_of_transformer_decoder_without_learning_parameters(transformer_decoder* t){
    uint64_t sum = 0;
    int i;
    for(i = 0; i < t->n_head; i++){
        sum+=size_of_model_without_learning_parameters(t->q[i]);
        sum+=size_of_model_without_learning_parameters(t->k[i]);
        sum+=size_of_model_without_learning_parameters(t->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        sum+=size_of_scaled_l2_norm(t->l2[i]);
    }
    
    sum+= sizeof(float)*(t->input_dimension*4 + t->dimension*t->dimension*4 +t->left_dimension*2);
    if(t->residual_flag == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2*sizeof(float);
    sum+=size_of_transformer_encoder_without_learning_parameters(t->e) + size_of_model_without_learning_parameters(t->linear_after_attention);
    return sum;    
}

/* This function, given 2 structures with the same number of layers will copy
 * the main features of the first into the second one
 * 
 * Inputs:
 * 
 *             @transformer_decoder* t:= the transformer decoder that must be copied
 *             @transformer_decoder* copy:= the transformer decoder structure in which will be copied t
 * */
void paste_transformer_decoder(transformer_decoder* t, transformer_decoder* copy){
    int i;
    for(i = 0; i < t->n_head; i++){
        paste_model(t->q[i],copy->q[i]);
        paste_model(t->k[i],copy->k[i]);
        paste_model(t->v[i],copy->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        paste_scaled_l2_norm(t->l2[i],copy->l2[i]);
    }
    copy->attention_flag = t->attention_flag;
    paste_transformer_encoder(t->e,copy->e);
    paste_model(t->linear_after_attention,copy->linear_after_attention);
    copy->k_embedding = t->k_embedding;
    copy->v_embedding = t->v_embedding;
}
/* This function, given 2 structures with the same number of layers will copy
 * the main features of the first into the second one
 * 
 * Inputs:
 * 
 *             @transformer_decoder* t:= the transformer decoder that must be copied
 *             @transformer_decoder* copy:= the transformer decoder structure in which will be copied t
 * */
void paste_transformer_decoder_without_learning_parameters(transformer_decoder* t, transformer_decoder* copy){
    int i;
    for(i = 0; i < t->n_head; i++){
        paste_model_without_learning_parameters(t->q[i],copy->q[i]);
        paste_model_without_learning_parameters(t->k[i],copy->k[i]);
        paste_model_without_learning_parameters(t->v[i],copy->v[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        paste_scaled_l2_norm(t->l2[i],copy->l2[i]);
    }
    copy->attention_flag = t->attention_flag;
    paste_transformer_encoder_without_learning_parameters(t->e,copy->e);
    paste_model_without_learning_parameters(t->linear_after_attention,copy->linear_after_attention);
    copy->k_embedding = t->k_embedding;
    copy->v_embedding = t->v_embedding;
}
/* This function, given 2 structures with the same number of layers will copy
 * the main features of the first into the second one
 * 
 * Inputs:
 * 
 *             @transformer_decoder* t:= the transformer decoder that must be copied
 *             @transformer_decoder* copy:= the transformer decoder structure in which will be copied t
 * */
void slow_paste_transformer_decoder(transformer_decoder* t, transformer_decoder* copy, float tau){
    int i;
    for(i = 0; i < t->n_head; i++){
        slow_paste_model(t->q[i],copy->q[i],tau);
        slow_paste_model(t->k[i],copy->k[i],tau);
        slow_paste_model(t->v[i],copy->v[i],tau);
    }
    for(i = 0; i < t->n_l2; i++){
        slow_paste_scaled_l2_norm(t->l2[i],copy->l2[i],tau);
    }
    copy->attention_flag = t->attention_flag;
    slow_paste_transformer_encoder(t->e,copy->e,tau);
    slow_paste_model(t->linear_after_attention,copy->linear_after_attention,tau);
    copy->k_embedding = t->k_embedding;
    copy->v_embedding = t->v_embedding;
}



/* This function computes the ff of the decoder
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from below the decoder, dimension:input1_dimension
 *             @ float* inputs2:= the inputs coming from the encoder/encoders,dimension: input2_dimension
 *             @ transformer_decoder*t:= the decoder that computes the feedforward
 *             @ int input1_dimension:= the dimension of inputs1
 *             @ int input2_dimension:= the dimension of inputs2
 * */

void decoder_transformer_ff(float* inputs1, float* inputs2, transformer_decoder* t,int input1_dimension, int input2_dimension){
    int i;
    for(i = 0; i < t->n_head; i++){
        model_tensor_input_ff(t->q[i],1,1,input1_dimension,inputs1);
        model_tensor_input_ff(t->k[i],1,1,input1_dimension,inputs1);
        model_tensor_input_ff(t->v[i],1,1,input1_dimension,inputs1);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding,t->v_embedding);
    model_tensor_input_ff(t->linear_after_attention,1,1,t->input_dimension,t->attention_output);
    if(t->residual_flag == TRANSFORMER_RESIDUAL){
        sum1D(inputs1,t->linear_after_attention->output_layer,t->residual1_output,t->linear_after_attention->output_dimension);
        if(t->normalization_flag == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual1_output,t->l2[0]->output);
            wrapped_encoder_transformer_decoder_ff(t->l2[0]->output,inputs2,t->e,input2_dimension,t->l2[0]->input_dimension);
        }
        
        else
            wrapped_encoder_transformer_decoder_ff(t->residual1_output,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension);    
    }
    else{
        if(t->normalization_flag == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->linear_after_attention->output_layer,t->l2[0]->output);
            wrapped_encoder_transformer_decoder_ff(t->l2[0]->output,inputs2,t->e,input2_dimension,t->l2[0]->input_dimension);
        }
        
        else
            wrapped_encoder_transformer_decoder_ff(t->linear_after_attention->output_layer,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension);
            
    }    
}
/* This function computes the ff of the decoder
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from below the decoder, dimension:input1_dimension
 *             @ float* inputs2:= the inputs coming from the encoder/encoders,dimension: input2_dimension
 *             @ transformer_decoder*t:= the decoder that computes the feedforward
 *             @ int input1_dimension:= the dimension of inputs1
 *             @ int input2_dimension:= the dimension of inputs2
 * */

void decoder_transformer_ff_opt(float* inputs1, float* inputs2, transformer_decoder* t,int input1_dimension, int input2_dimension, transformer_decoder* t2){
    int i;
    for(i = 0; i < t->n_head; i++){
        model_tensor_input_ff_without_learning_parameters(t->q[i],t2->q[i],1,1,input1_dimension,inputs1);
        model_tensor_input_ff_without_learning_parameters(t->k[i],t2->k[i],1,1,input1_dimension,inputs1);
        model_tensor_input_ff_without_learning_parameters(t->v[i],t2->v[i],1,1,input1_dimension,inputs1);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding,t->v_embedding);
    model_tensor_input_ff_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output);
    if(t->residual_flag == TRANSFORMER_RESIDUAL){
        sum1D(inputs1,t->linear_after_attention->output_layer,t->residual1_output,t->linear_after_attention->output_dimension);
        if(t->normalization_flag == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual1_output,t->l2[0]->output);
            wrapped_encoder_transformer_decoder_ff_opt(t->l2[0]->output,inputs2,t->e,input2_dimension,t->l2[0]->input_dimension,t2->e);
        }
        
        else
            wrapped_encoder_transformer_decoder_ff_opt(t->residual1_output,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension,t2->e);    
    }
    else{
        if(t->normalization_flag == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->linear_after_attention->output_layer,t->l2[0]->output);
            wrapped_encoder_transformer_decoder_ff_opt(t->l2[0]->output,inputs2,t->e,input2_dimension,t->l2[0]->input_dimension,t2->e);
        }
        
        else
            wrapped_encoder_transformer_decoder_ff_opt(t->linear_after_attention->output_layer,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension,t2->e);
            
    }    
}

/* This function computes the ff of the decoder
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from below the decoder, dimension:input1_dimension
 *             @ float* inputs2:= the inputs coming from the encoder/encoders,dimension: input2_dimension
 *             @ transformer_decoder*t:= the decoder that computes the feedforward
 *             @ int input1_dimension:= the dimension of inputs1
 *             @ int input2_dimension:= the dimension of inputs2
 * */
float* decoder_transformer_bp(float* inputs1, float* inputs2, transformer_decoder* t, int input1_dimension, int input2_dimension, float* output_error, float* inputs2_error){
    int i;
    float* err, *err2, *temp, *tempp, *inputs = inputs1;
    uint64_t sumq=0,sumk=0,sumv=0;
    if(t->normalization_flag == SCALED_L2_NORMALIZATION)
    err  = wrapped_encoder_transformer_decoder_bp(t->l2[0]->output,inputs2,t->e,input2_dimension,t->l2[0]->input_dimension,output_error,inputs2_error);
    else if(t->residual_flag == TRANSFORMER_RESIDUAL)
    err  = wrapped_encoder_transformer_decoder_bp(t->residual1_output,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension,output_error,inputs2_error);
    else
    err  = wrapped_encoder_transformer_decoder_bp(t->linear_after_attention->output_layer,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension,output_error,inputs2_error);
    if(t->normalization_flag == SCALED_L2_NORMALIZATION){
        if(t->residual_flag == TRANSFORMER_RESIDUAL)
            back_propagation_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,err,t->l2[0]->output_error);
        else
            back_propagation_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,err,t->l2[0]->output_error);
        err2 = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
        multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,err2,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding,t->v_embedding);
    
    }
    
    else{
        err2 = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,err,t->linear_after_attention->output_dimension);
        multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,err2,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding,t->v_embedding);
    }
    
    for(i = 0; i < t->n_head; i++){
        if(!i)
            temp = model_tensor_input_bp(t->q[i],1,1,input1_dimension,inputs,&t->q_error[sumq],t->q[i]->output_dimension);
        else{
            tempp = model_tensor_input_bp(t->q[i],1,1,input1_dimension,inputs,&t->q_error[sumq],t->q[i]->output_dimension);
            sum1D(temp,tempp,temp,input1_dimension);
        }
        tempp = model_tensor_input_bp(t->k[i],1,1,input1_dimension,inputs,&t->k_error[sumk],t->k[i]->output_dimension);
        sum1D(temp,tempp,temp,input1_dimension);
        tempp = model_tensor_input_bp(t->v[i],1,1,input1_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
        sum1D(temp,tempp,temp,input1_dimension);
        sumq+=t->q[i]->output_dimension;
        sumk+=t->k[i]->output_dimension;
        sumv+=t->v[i]->output_dimension;
    }
    
    if (t->residual_flag == TRANSFORMER_RESIDUAL){
        if(t->normalization_flag == SCALED_L2_NORMALIZATION)
            sum1D(t->fcls[2]->temp2,t->l2[0]->output_error,t->fcls[2]->temp2,t->linear_after_attention->output_dimension);
        else
            sum1D(t->fcls[2]->temp2,err,t->fcls[2]->temp2,t->linear_after_attention->output_dimension);
    }
    
    return t->fcls[2]->temp2;
        
}

float* decoder_transformer_bp_opt(float* inputs1, float* inputs2, transformer_decoder* t, int input1_dimension, int input2_dimension, float* output_error, float* inputs2_error,transformer_decoder* t2){
    int i;
    float* err, *err2, *temp, *tempp, *inputs = inputs1;
    uint64_t sumq=0,sumk=0,sumv=0;
    if(t->normalization_flag == SCALED_L2_NORMALIZATION)
    err  = wrapped_encoder_transformer_decoder_bp_opt(t->l2[0]->output,inputs2,t->e,input2_dimension,t->l2[0]->input_dimension,output_error,inputs2_error,t2->e);
    else if(t->residual_flag == TRANSFORMER_RESIDUAL)
    err  = wrapped_encoder_transformer_decoder_bp_opt(t->residual1_output,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension,output_error,inputs2_error,t2->e);
    else
    err  = wrapped_encoder_transformer_decoder_bp_opt(t->linear_after_attention->output_layer,inputs2,t->e,input2_dimension,t->linear_after_attention->output_dimension,output_error,inputs2_error,t2->e);
    if(t->normalization_flag == SCALED_L2_NORMALIZATION){
        if(t->residual_flag == TRANSFORMER_RESIDUAL)
            back_propagation_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,err,t->l2[0]->output_error);
        else
            back_propagation_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,err,t->l2[0]->output_error);
        err2 = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
        multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,err2,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding,t->v_embedding);
    
    }
    
    else{
        err2 = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,err,t->linear_after_attention->output_dimension);
        multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,err2,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding,t->v_embedding);
    }
    
    for(i = 0; i < t->n_head; i++){
        if(!i)
            temp = model_tensor_input_bp_without_learning_parameters(t->q[i],t2->q[i],1,1,input1_dimension,inputs,&t->q_error[sumq],t->q[i]->output_dimension);
        else{
            tempp = model_tensor_input_bp_without_learning_parameters(t->q[i],t2->q[i],1,1,input1_dimension,inputs,&t->q_error[sumq],t->q[i]->output_dimension);
            sum1D(temp,tempp,temp,input1_dimension);
        }
        tempp = model_tensor_input_bp_without_learning_parameters(t->k[i],t2->k[i],1,1,input1_dimension,inputs,&t->k_error[sumk],t->k[i]->output_dimension);
        sum1D(temp,tempp,temp,input1_dimension);
        tempp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input1_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
        sum1D(temp,tempp,temp,input1_dimension);
        sumq+=t->q[i]->output_dimension;
        sumk+=t->k[i]->output_dimension;
        sumv+=t->v[i]->output_dimension;
    }
    
    if (t->residual_flag == TRANSFORMER_RESIDUAL){
        if(t->normalization_flag == SCALED_L2_NORMALIZATION)
            sum1D(t->fcls[2]->temp2,t->l2[0]->output_error,t->fcls[2]->temp2,t->linear_after_attention->output_dimension);
        else
            sum1D(t->fcls[2]->temp2,err,t->fcls[2]->temp2,t->linear_after_attention->output_dimension);
    }
    
    return t->fcls[2]->temp2;
        
}

/* This function computes the feed forward of the encoder inside the decoder
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ float* inputs1:= the inputs coming from the bottom of this encoder, dimension: input_dimension
 *             @ float* inputs2:= the inputs coming from another encoder /encoders or somenthing else for the queries, keys, dimension: input_dimension1
 *             @ transformer_encoder* t:= the transformer encoder that must computes the feed forward
 *             @ int input_dimension2:= the dimension of inputs2
 *             @ int input_dimension1:= the dimension of inputs1
 * */        
void wrapped_encoder_transformer_decoder_ff(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1){
    int i, input_dimension = input_dimension1;
    float* inputs = inputs1;
    for(i = 0; i < t->n_head; i++){
        if(input_dimension2 != t->q[i]->cls[0]->channels*t->q[i]->cls[0]->input_rows*t->q[i]->cls[0]->input_cols || input_dimension2 != t->k[i]->cls[0]->channels*t->k[i]->cls[0]->input_rows*t->k[i]->cls[0]->input_cols || input_dimension != t->v[i]->cls[0]->channels*t->v[i]->cls[0]->input_rows*t->v[i]->cls[0]->input_cols){
            fprintf(stderr,"Error: queries, keys and values don't match the inputs - inputs given: %d, %d, %d, inputs fcls: %d,%d,%d\n",input_dimension2,input_dimension2,input_dimension,t->q[i]->cls[0]->channels*t->q[i]->cls[0]->input_rows*t->q[i]->cls[0]->input_cols,t->k[i]->cls[0]->channels*t->k[i]->cls[0]->input_rows*t->k[i]->cls[0]->input_cols,t->v[i]->cls[0]->channels*t->v[i]->cls[0]->input_rows*t->v[i]->cls[0]->input_cols);
            exit(1);
        }
        model_tensor_input_ff(t->q[i],1,1,input_dimension2,inputs2);
        model_tensor_input_ff(t->k[i],1,1,input_dimension2,inputs2);
        model_tensor_input_ff(t->v[i],1,1,input_dimension,inputs1);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
    model_tensor_input_ff(t->linear_after_attention,1,1,t->input_dimension,t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        if(t->linear_after_attention->output_dimension != input_dimension){
            fprintf(stderr,"Error: the input dimension of the transformer does not match the multi headed attention output!\n");
            exit(1);
        }
        sum1D(inputs,t->linear_after_attention->output_layer,t->residual1_output,t->linear_after_attention->output_dimension);
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual1_output,t->l2[0]->output);
            model_tensor_input_ff(t->m,1,t->linear_after_attention->output_dimension,1,t->l2[0]->output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->l2[0]->output,t->m->output_layer,t->residual2_output,t->m->output_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
        
        else{
            model_tensor_input_ff(t->m,1,t->linear_after_attention->output_dimension,1,t->residual1_output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->residual1_output,t->m->output_layer,t->residual2_output,t->m->output_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual2_output,t->l2[0]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->m->output_layer,t->l2[0]->output);
                }
            }
        }
    }
    
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->linear_after_attention->output_layer,t->l2[0]->output);
            model_tensor_input_ff(t->m,1,t->linear_after_attention->output_dimension,1,t->l2[0]->output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->l2[0]->output,t->m->output_layer,t->residual2_output,t->l2[0]->input_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
        
        else{
            model_tensor_input_ff(t->m,1,t->linear_after_attention->output_dimension,1,t->linear_after_attention->output_layer);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->linear_after_attention->output_layer,t->m->output_layer,t->residual2_output,t->linear_after_attention->output_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual2_output,t->l2[0]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->m->output_layer,t->l2[0]->output);
                }
            }
        }
    }
}

/* This function computes the feed forward of the encoder inside the decoder
 * 
 * 
 * Inputs:
 * 
 * 
 *             @ float* inputs1:= the inputs coming from the bottom of this encoder, dimension: input_dimension
 *             @ float* inputs2:= the inputs coming from another encoder /encoders or somenthing else for the queries, keys, dimension: input_dimension1
 *             @ transformer_encoder* t:= the transformer encoder that must computes the feed forward
 *             @ int input_dimension2:= the dimension of inputs2
 *             @ int input_dimension1:= the dimension of inputs1
 * */        
void wrapped_encoder_transformer_decoder_ff_opt(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1, transformer_encoder* t2){
    int i, input_dimension = input_dimension1;
    float* inputs = inputs1;
    for(i = 0; i < t->n_head; i++){
        if(input_dimension2 != t->q[i]->cls[0]->channels*t->q[i]->cls[0]->input_rows*t->q[i]->cls[0]->input_cols || input_dimension2 != t->k[i]->cls[0]->channels*t->k[i]->cls[0]->input_rows*t->k[i]->cls[0]->input_cols || input_dimension != t->v[i]->cls[0]->channels*t->v[i]->cls[0]->input_rows*t->v[i]->cls[0]->input_cols){
            fprintf(stderr,"Error: queries, keys and values don't match the inputs - inputs given: %d, %d, %d, inputs fcls: %d,%d,%d\n",input_dimension2,input_dimension2,input_dimension,t->q[i]->cls[0]->channels*t->q[i]->cls[0]->input_rows*t->q[i]->cls[0]->input_cols,t->k[i]->cls[0]->channels*t->k[i]->cls[0]->input_rows*t->k[i]->cls[0]->input_cols,t->v[i]->cls[0]->channels*t->v[i]->cls[0]->input_rows*t->v[i]->cls[0]->input_cols);
            exit(1);
        }
        model_tensor_input_ff_without_learning_parameters(t->q[i],t2->q[i],1,1,input_dimension2,inputs2);
        model_tensor_input_ff_without_learning_parameters(t->k[i],t2->k[i],1,1,input_dimension2,inputs2);
        model_tensor_input_ff_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs1);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
    model_tensor_input_ff_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        if(t->linear_after_attention->output_dimension != input_dimension){
            fprintf(stderr,"Error: the input dimension of the transformer does not match the multi headed attention output!\n");
            exit(1);
        }
        sum1D(inputs,t->linear_after_attention->output_layer,t->residual1_output,t->linear_after_attention->output_dimension);
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual1_output,t->l2[0]->output);
            model_tensor_input_ff_without_learning_parameters(t->m,t2->m,1,t->linear_after_attention->output_dimension,1,t->l2[0]->output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->l2[0]->output,t->m->output_layer,t->residual2_output,t->m->output_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->linear_after_attention->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
        
        else{
            model_tensor_input_ff_without_learning_parameters(t->m,t2->m,1,t->linear_after_attention->output_dimension,1,t->residual1_output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->residual1_output,t->m->output_layer,t->residual2_output,t->m->output_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual2_output,t->l2[0]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->m->output_layer,t->l2[0]->output);
                }
            }
        }
    }
    
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            feed_forward_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->linear_after_attention->output_layer,t->l2[0]->output);
            model_tensor_input_ff_without_learning_parameters(t->m,t2->m,1,t->linear_after_attention->output_dimension,1,t->l2[0]->output);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->l2[0]->output,t->m->output_layer,t->residual2_output,t->l2[0]->input_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->residual2_output,t->l2[1]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[1]->learned_g,&t->l2[1]->norm,t->m->output_layer,t->l2[1]->output);
                }
            }
        }
        
        else{
            model_tensor_input_ff_without_learning_parameters(t->m,t2->m,1,t->linear_after_attention->output_dimension,1,t->linear_after_attention->output_layer);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                if(t->linear_after_attention->output_dimension != t->m->output_dimension){
                    fprintf(stderr,"Error: the dimension of the output coming from the multiheaded attention doesn't match the output of the feed forward layers!\n");
                    exit(1);
                }
                sum1D(t->linear_after_attention->output_layer,t->m->output_layer,t->residual2_output,t->linear_after_attention->output_dimension);
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->residual2_output,t->l2[0]->output);
                }
            }
            else{
                if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
                    feed_forward_scaled_l2_norm(t->m->output_dimension,t->l2[0]->learned_g,&t->l2[0]->norm,t->m->output_layer,t->l2[0]->output);
                }
            }
        }
    }
}



/* This function computes the back propagation of our encoder transformer network insde the decoder (see the description for the
 * ff wrapped endoer)
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ float* inputs2:= the inputs coming from the encoders, the input dimension1
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension1:= the dimension of the inputs1
 *             @ int input_dimension2:= the dimension of the inputs2
 *                @ float* output_error:= the error for the output
 *               @float* encoder_error:= where will be stored the error for the encoders that fed the decoder
 * it returns the error for the inputs
 * */
float* wrapped_encoder_transformer_decoder_bp(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1,float* output_error,float* encoder_error){
    int i, input_dimension = input_dimension1;
    float* inputs = inputs1;
    float* temp, *tempp, *temppp;
    uint64_t sumq = 0;
    uint64_t sumk = 0;
    uint64_t sumv = 0;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
        if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
            back_propagation_scaled_l2_norm(t->l2[t->n_l2-1]->input_dimension,t->l2[t->n_l2-1]->learned_g,&t->l2[t->n_l2-1]->d_learned_g,t->l2[t->n_l2-1]->norm,t->residual2_output,output_error,t->l2[t->n_l2-1]->output_error);
        else    
            back_propagation_scaled_l2_norm(t->l2[t->n_l2-1]->input_dimension,t->l2[t->n_l2-1]->learned_g,&t->l2[t->n_l2-1]->d_learned_g,t->l2[t->n_l2-1]->norm,t->m->output_layer,output_error,t->l2[t->n_l2-1]->output_error);    
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp(t->m,1,1,t->l2[0]->input_dimension,t->l2[0]->output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->l2[t->n_l2-1]->input_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,temp,t->l2[0]->output_error);
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    temp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(temp,temppp,temp,input_dimension);
                }
                
                tt = model_tensor_input_bp(t->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp(t->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[0]->output_error,temp,input_dimension);
            return temp;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->residual1_output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->linear_after_attention->output_dimension,t->linear_after_attention->output_layer,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->l2[t->n_l2-1]->input_dimension);
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    tempp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(tempp,temppp,tempp,input_dimension);
                }
                
                tt = model_tensor_input_bp(t->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp(t->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(tempp,temp,tempp,input_dimension);
            return tempp;
        }
        
    }
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp(t->m,1,1,t->linear_after_attention->output_dimension,t->l2[0]->output,output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,temp,t->l2[0]->output_error);
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    temp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(temp,temppp,temp,input_dimension);
                }
                
                tt = model_tensor_input_bp(t->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp(t->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[0]->output_error,temp,input_dimension);
            return temp;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->linear_after_attention->output_dimension,t->residual1_output,output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->linear_after_attention->output_dimension,t->linear_after_attention->output_layer,output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    tempp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp(t->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(tempp,temppp,tempp,input_dimension);
                }
                
                tt = model_tensor_input_bp(t->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp(t->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(tempp,temp,tempp,input_dimension);
            return tempp;
        }
        
    }
    
    
}
/* This function computes the back propagation of our encoder transformer network insde the decoder (see the description for the
 * ff wrapped endoer)
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs1:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ float* inputs2:= the inputs coming from the encoders, the input dimension1
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension1:= the dimension of the inputs1
 *             @ int input_dimension2:= the dimension of the inputs2
 *                @ float* output_error:= the error for the output
 *               @float* encoder_error:= where will be stored the error for the encoders that fed the decoder
 * it returns the error for the inputs
 * */
float* wrapped_encoder_transformer_decoder_bp_opt(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1,float* output_error,float* encoder_error, transformer_encoder* t2){
    int i, input_dimension = input_dimension1;
    float* inputs = inputs1;
    float* temp, *tempp, *temppp;
    uint64_t sumq = 0;
    uint64_t sumk = 0;
    uint64_t sumv = 0;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION){
        if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
            back_propagation_scaled_l2_norm(t->l2[t->n_l2-1]->input_dimension,t->l2[t->n_l2-1]->learned_g,&t->l2[t->n_l2-1]->d_learned_g,t->l2[t->n_l2-1]->norm,t->residual2_output,output_error,t->l2[t->n_l2-1]->output_error);
        else    
            back_propagation_scaled_l2_norm(t->l2[t->n_l2-1]->input_dimension,t->l2[t->n_l2-1]->learned_g,&t->l2[t->n_l2-1]->d_learned_g,t->l2[t->n_l2-1]->norm,t->m->output_layer,output_error,t->l2[t->n_l2-1]->output_error);    
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->l2[0]->input_dimension,t->l2[0]->output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->l2[t->n_l2-1]->input_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,temp,t->l2[0]->output_error);
            tempp = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    temp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(temp,temppp,temp,input_dimension);
                }
                
                tt = model_tensor_input_bp_without_learning_parameters(t->k[i],t2->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp_without_learning_parameters(t->q[i],t2->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[0]->output_error,temp,input_dimension);
            return temp;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->input_dimension,t->residual1_output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->linear_after_attention->output_dimension,t->linear_after_attention->output_layer,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->l2[t->n_l2-1]->input_dimension);
            tempp = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    tempp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(tempp,temppp,tempp,input_dimension);
                }
                
                tt = model_tensor_input_bp_without_learning_parameters(t->k[i],t2->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp_without_learning_parameters(t->q[i],t2->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(tempp,temp,tempp,input_dimension);
            return tempp;
        }
        
    }
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->linear_after_attention->output_dimension,t->l2[0]->output,output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,temp,t->l2[0]->output_error);
            tempp = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    temp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(temp,temppp,temp,input_dimension);
                }
                
                tt = model_tensor_input_bp_without_learning_parameters(t->k[i],t2->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp_without_learning_parameters(t->q[i],t2->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[0]->output_error,temp,input_dimension);
            return temp;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->linear_after_attention->output_dimension,t->residual1_output,output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->linear_after_attention->output_dimension,t->linear_after_attention->output_layer,output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            
            tempp = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag,t->k_embedding_dimension,t->v_embedding_dimension);
            for(i = 0; i < t->n_head; i++){
                float* tt;
                if(!i)
                    tempp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                else{
                    temppp = model_tensor_input_bp_without_learning_parameters(t->v[i],t2->v[i],1,1,input_dimension,inputs,&t->v_error[sumv],t->v[i]->output_dimension);
                    sum1D(tempp,temppp,tempp,input_dimension);
                }
                
                tt = model_tensor_input_bp_without_learning_parameters(t->k[i],t2->k[i],1,1,input_dimension2,inputs2,&t->k_error[sumk],t->k[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                tt = model_tensor_input_bp_without_learning_parameters(t->q[i],t2->q[i],1,1,input_dimension2,inputs2,&t->q_error[sumq],t->q[i]->output_dimension);
                sum1D(encoder_error,tt,encoder_error,input_dimension2);
                sumq+=t->q[i]->output_dimension;
                sumk+=t->k[i]->output_dimension;
                sumv+=t->v[i]->output_dimension;
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(tempp,temp,tempp,input_dimension);
            return tempp;
        }
        
    }
    
    
}



