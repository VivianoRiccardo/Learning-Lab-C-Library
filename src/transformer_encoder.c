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


/* This function allocates the space needed for a transformer encoder
 * 
 * 
 * Input:
 * 
 *             @ model* m:= the model after attention + residual + normalization (accrding to the transformer sould be 2 fully connected layers)
 *             @ fcl** fcls:= the fully connected layers, we should have n_head fully connected layers without any activation function for the queries, the keys, and the values at the
 *                            beginning of the transformer encoder layer, becase each query, value and key must pass through a linear matrix given by the fully connected layers weights
 *                            then the encoder needs 2 other fully connected layer after the self- attention, remember the layer before the last one must have an activation function
 *                            (Relu/Leaky Relu/ Elu suggested) and the last one should not have any activation function. Dimensions 3*n_head
 *            @ scaled_l2_norm** l2:= this layer is used as normalization layer instead of a layer normalization layer because of this paper: Transformers without Tears:Improving the Normalization of Self-Attention
 *                                     future implementation with fixed normalization too or simply cosine normalization with learnable parameter will be implemented (maybe), dimensions: 0,1 or 2
 *             @ int input_dimension:= the total dimension of the input of the attention layer
 *             @ int n_head:= number of head attention
 *             @ int residual_flag1:= TRANSFORMER_RESIDUAL or TRANSFORMER_NO_RESIDUAL
 *             @ int normalization_flag1:= SCALED_L2_NORMALIZATION or NO_NORMALIZATION
 *             @ int residual_flag2:= TRANSFORMER_RESIDUAL or TRANSFORMER_NO_RESIDUAL
 *             @ int normalization_flag2:= SCALED_L2_NORMALIZATION or NO_NORMALIZATION (FUTURE COSINE NORMALIZATION WITH LEARNABLE PARAMETER WILL BE ADDED[fix + scaled normalization])
 *             */
transformer_encoder* transformer_encoder_layer(model* m, model* linear_after_attention, fcl** fcls, scaled_l2_norm** l2, int input_dimension, int n_head,int residual_flag1,int normalization_flag1,int residual_flag2,int normalization_flag2, int attention_flag){
    if(fcls == NULL){
        fprintf(stderr,"Error: there must be 3*n_head fully connected layers\n");
        exit(1);
    }
    
    if(l2 == NULL && (normalization_flag1!=NO_NORMALIZATION || normalization_flag2 != NO_NORMALIZATION)){
        fprintf(stderr,"Error: l2 is a normalization layer in this case you must set either normalization flag1 or normalization flag2 or both!\n");
        exit(1);
    }
    
    if(l2 != NULL && normalization_flag1 != SCALED_L2_NORMALIZATION && normalization_flag2 != SCALED_L2_NORMALIZATION){
        fprintf(stderr,"Error: if you have scaled l2 normalization layers you must set the normalization flags accordingly!\n");
        exit(1);
    }
    
    if(input_dimension <= 0 || n_head <= 0){
        fprintf(stderr,"Error: input_dimension and n_head must be > 0\n");
        exit(1); 
    }
    
    if(input_dimension%n_head){
        fprintf(stderr,"Error: n_head must divide perfectly input_dimension\n");
        exit(1);
    }
    
    if(m->output_dimension != input_dimension){
        fprintf(stderr,"Error: the output dimension of your model must match the input dimension!\n");
        exit(1);
    }
    
    int count = 0;
    if(normalization_flag1 == SCALED_L2_NORMALIZATION){
        if(l2[count]->input_dimension != input_dimension){
            fprintf(stderr,"Error: you normalization dimension must match the transformer input dimension!\n");
            exit(1);
        }
        count++;
    }
    if(normalization_flag2 == SCALED_L2_NORMALIZATION){
        if(l2[count]->input_dimension != input_dimension){
            fprintf(stderr,"Error: you normalization dimension must match the transformer input dimension!\n");
            exit(1);
        }
        count++;
    }
    
        
    transformer_encoder* t = (transformer_encoder*)malloc(sizeof(transformer_encoder));
    t->n_l2 = 0;
    if(normalization_flag1 != SCALED_L2_NORMALIZATION)
        normalization_flag1 = NO_NORMALIZATION;
    else
        t->n_l2++;
    if(normalization_flag2 != SCALED_L2_NORMALIZATION)
        normalization_flag2 = NO_NORMALIZATION;
    else
        t->n_l2++;
    if(residual_flag1 != TRANSFORMER_RESIDUAL)
        residual_flag1 = TRANSFORMER_NO_RESIDUAL;
    if(residual_flag2 != TRANSFORMER_RESIDUAL)
        residual_flag2 = TRANSFORMER_NO_RESIDUAL;
    
    t->m = m;
    t->linear_after_attention = linear_after_attention;
    t->attention_flag = attention_flag;
    t->fcls = fcls;
    t->l2 = l2;
    t->input_dimension = input_dimension;
    t->n_head = n_head;
    t->residual_flag1 = residual_flag1;
    t->residual_flag2 = residual_flag2;
    t->normalization_flag1 = normalization_flag1;
    t->normalization_flag2 = normalization_flag2;
    t->dimension = input_dimension/n_head;
    t->encoder_output_error = (float*)calloc(m->output_dimension,sizeof(float));
    t->q = (float*)calloc(input_dimension,sizeof(float));
    t->q_error = (float*)calloc(input_dimension,sizeof(float));
    t->k = (float*)calloc(input_dimension,sizeof(float));
    t->k_error = (float*)calloc(input_dimension,sizeof(float));
    t->v = (float*)calloc(input_dimension,sizeof(float));
    t->v_error = (float*)calloc(input_dimension,sizeof(float));
    t->score_matrix = (float*)calloc(input_dimension*t->dimension,sizeof(float));
    t->score_matrix_softmax = (float*)calloc(input_dimension*t->dimension,sizeof(float));
    t->score_matrix_softmax_error = (float*)calloc(input_dimension*t->dimension,sizeof(float));
    t->score_matrix_error = (float*)calloc(input_dimension*t->dimension,sizeof(float));
    t->attention_output = (float*)calloc(input_dimension,sizeof(float));
    if(residual_flag1 == TRANSFORMER_RESIDUAL){
        t->residual1_output = (float*)calloc(linear_after_attention->output_dimension,sizeof(float));
        t->residual1_output_error = (float*)calloc(linear_after_attention->output_dimension,sizeof(float));
    }
    else{
        t->residual1_output = NULL;
        t->residual1_output_error = NULL;
    }
    if(residual_flag2 == TRANSFORMER_RESIDUAL){
        t->residual2_output = (float*)calloc(m->output_dimension,sizeof(float));
        t->residual2_output_error = (float*)calloc(m->output_dimension,sizeof(float));
    }
    else{
        t->residual2_output = NULL;
        t->residual2_output_error = NULL;
    }
    return t;
}

/* This function deallocates the space allocated by a transformer encoder
 * 
 * Input:
 *             @ transformer_encoder* t:= the transformer encoder
 * */
void free_transformer_encoder_layer(transformer_encoder* t){
    int i;
    if (t == NULL)
        return;
    for(i = 0; i < t->n_head*3; i++){
        free_fully_connected(t->fcls[i]);
    }
    free(t->fcls);
    free_model(t->m);
    free_model(t->linear_after_attention);
    int n_l2 = 0;
    if(t->normalization_flag1 == SCALED_L2_NORMALIZATION)
        n_l2++;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION)
        n_l2++;
    for(i = 0; i < n_l2; i++){
        free_scaled_l2_normalization_layer(t->l2[i]);
    }
    
    free(t->l2);
    free(t->encoder_output_error);
    free(t->q);
    free(t->q_error);
    free(t->k);
    free(t->v);
    free(t->v_error);
    free(t->k_error);
    free(t->score_matrix);
    free(t->score_matrix_error);
    free(t->score_matrix_softmax);
    free(t->score_matrix_softmax_error);
    free(t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        free(t->residual1_output);
        free(t->residual1_output_error);
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        free(t->residual2_output);
        free(t->residual2_output_error);
    }
    free(t);
    
    return;
}
/* This function deallocates the space allocated by a transformer encoder
 * 
 * Input:
 *             @ transformer_encoder* t:= the transformer encoder
 * */
void free_transformer_encoder_layer_without_learning_parameters(transformer_encoder* t){
    int i;
    if (t == NULL)
        return;
    for(i = 0; i < t->n_head*3; i++){
        free_fully_connected(t->fcls[i]);
    }
    free(t->fcls);
    free_model_without_learning_parameters(t->m);
    free_model_without_learning_parameters(t->linear_after_attention);
    int n_l2 = 0;
    if(t->normalization_flag1 == SCALED_L2_NORMALIZATION)
        n_l2++;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION)
        n_l2++;
    for(i = 0; i < n_l2; i++){
        free_scaled_l2_normalization_layer(t->l2[i]);
    }
    
    free(t->l2);
    free(t->encoder_output_error);
    free(t->q);
    free(t->q_error);
    free(t->k);
    free(t->v);
    free(t->v_error);
    free(t->k_error);
    free(t->score_matrix);
    free(t->score_matrix_error);
    free(t->score_matrix_softmax);
    free(t->score_matrix_softmax_error);
    free(t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        free(t->residual1_output);
        free(t->residual1_output_error);
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        free(t->residual2_output);
        free(t->residual2_output_error);
    }
    free(t);
    
    return;
}
/* This function deallocates the space allocated by a transformer encoder wrapped inside a decoder (is not deallocated l2 and fcls
 * 
 * Input:
 *             @ transformer_encoder* t:= the transformer encoder
 * */
void free_transformer_wrapped_encoder_layer(transformer_encoder* t){
    int i;
    if (t == NULL)
        return;
    for(i = 0; i < t->n_head*3; i++){
        free_fully_connected(t->fcls[i]);
    }
    free_model(t->m);
    free_model(t->linear_after_attention);
    int n_l2 = 0;
    if(t->normalization_flag1 == SCALED_L2_NORMALIZATION)
        n_l2++;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION)
        n_l2++;
    for(i = 0; i < n_l2; i++){
        free_scaled_l2_normalization_layer(t->l2[i]);
    }
    free(t->encoder_output_error);
    free(t->q);
    free(t->q_error);
    free(t->k);
    free(t->v);
    free(t->v_error);
    free(t->k_error);
    free(t->score_matrix);
    free(t->score_matrix_error);
    free(t->score_matrix_softmax);
    free(t->score_matrix_softmax_error);
    free(t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        free(t->residual1_output);
        free(t->residual1_output_error);
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        free(t->residual2_output);
        free(t->residual2_output_error);
    }
    free(t);
    
    return;
    
}
/* This function deallocates the space allocated by a transformer encoder wrapped inside a decoder (is not deallocated l2 and fcls
 * 
 * Input:
 *             @ transformer_encoder* t:= the transformer encoder
 * */
void free_transformer_wrapped_encoder_layer_without_learning_parameters(transformer_encoder* t){
    int i;
    if (t == NULL)
        return;
    for(i = 0; i < t->n_head*3; i++){
        free_fully_connected(t->fcls[i]);
    }
    free_model_without_learning_parameters(t->m);
    free_model_without_learning_parameters(t->linear_after_attention);
    int n_l2 = 0;
    if(t->normalization_flag1 == SCALED_L2_NORMALIZATION)
        n_l2++;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION)
        n_l2++;
    for(i = 0; i < n_l2; i++){
        free_scaled_l2_normalization_layer(t->l2[i]);
    }
    free(t->encoder_output_error);
    free(t->q);
    free(t->q_error);
    free(t->k);
    free(t->v);
    free(t->v_error);
    free(t->k_error);
    free(t->score_matrix);
    free(t->score_matrix_error);
    free(t->score_matrix_softmax);
    free(t->score_matrix_softmax_error);
    free(t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        free(t->residual1_output);
        free(t->residual1_output_error);
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        free(t->residual2_output);
        free(t->residual2_output_error);
    }
    free(t);
    
    return;
    
}


/* This function saves a transformer encoder strcture into a file
 * 
 * 
 * Inputs:
 * 
 * 
 *                 @ transformer_encoder* t:= the transformer encoder that must be saved
 *                 @ int n:= the file name in integer, example n = 0 the filename will be 0.bin
 * 
 * */
void save_transformer_encoder(transformer_encoder* t, int n){
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
    
    i = fwrite(&t->attention_flag,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->input_dimension,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->n_head,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->residual_flag1,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->residual_flag2,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->normalization_flag1,sizeof(int),1,fw);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a transformer layer\n");
        exit(1);
    }
    i = fwrite(&t->normalization_flag2,sizeof(int),1,fw);
    
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
    
    for(i = 0; i < t->n_head*3; i++){
        save_fcl(t->fcls[i],n);
    }

    for(i = 0; i < t->n_l2; i++){
        save_scaled_l2_norm(t->l2[i],n);
    }
    
    save_model(t->m,n);
    save_model(t->linear_after_attention,n);
}

/* This function loads a transformer encoder structure from a file fr
 * 
 * Inputs:
 *         
 *                 @ FILE* fr:= the file from which must be loaded
 * 
 * */
transformer_encoder* load_transformer_encoder(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    int input_dimension = 0,n_head = 0,residual_flag1 = 0,normalization_flag1 = 0,residual_flag2 = 0,normalization_flag2 = 0, attention_flag = 0;
    
    
    i = fread(&attention_flag,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&input_dimension,sizeof(int),1,fr);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&n_head,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&residual_flag1,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&residual_flag2,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&normalization_flag1,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    i = fread(&normalization_flag2,sizeof(int),1,fr);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a transformer layer\n");
        exit(1);
    }
    
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*n_head*3);

    scaled_l2_norm** l2 = NULL;
    int count = 0;
    if(normalization_flag1 == SCALED_L2_NORMALIZATION)
        count++;
    if(normalization_flag2 == SCALED_L2_NORMALIZATION)
        count++;
    for(i = 0; i < n_head*3; i++){
        fcls[i] = load_fcl(fr);
    }
    if(count)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*count);
    for(i = 0; i < count; i++){
        l2[i] = load_scaled_l2_norm(fr);
    }
    
    model* m = load_model_with_file_already_opened(fr);
    model* linear_after_attention = load_model_with_file_already_opened(fr);
    return transformer_encoder_layer(m,linear_after_attention,fcls,l2,input_dimension,n_head,residual_flag1,normalization_flag1,residual_flag2,normalization_flag2,attention_flag);
}

/* this function allocates the space for a new transformer encoder structure that is the exact copy of the 
 * transformer encoder given as input
 * 
 * 
 * Inputs:
 * 
 *             @transformer_encoder* t:= the structure that must be copied
 * 
 * */
transformer_encoder* copy_transformer_encoder(transformer_encoder* t){
    if( t == NULL)
        return NULL;
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*t->n_head*3);
    scaled_l2_norm** l2 = NULL;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        fcls[i] = copy_fcl(t->fcls[i]);
    }
    
    if(t->n_l2)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*t->n_l2);
    for(i = 0; i < t->n_l2; i++){
        l2[i] = copy_scaled_l2_norm(t->l2[i]);
    }
    model* m = copy_model(t->m);
    model* m2 = copy_model(t->linear_after_attention);
    return transformer_encoder_layer(m,m2,fcls,l2,t->input_dimension,t->n_head,t->residual_flag1,t->normalization_flag1,t->residual_flag2,t->normalization_flag2,t->attention_flag);
}
/* this function allocates the space for a new transformer encoder structure that is the exact copy of the 
 * transformer encoder given as input
 * 
 * 
 * Inputs:
 * 
 *             @transformer_encoder* t:= the structure that must be copied
 * 
 * */
transformer_encoder* copy_transformer_encoder_without_learning_parameters(transformer_encoder* t){
    if( t == NULL)
        return NULL;
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*t->n_head*3);
    scaled_l2_norm** l2 = NULL;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        fcls[i] = copy_fcl_without_learning_parameters(t->fcls[i]);
    }
    
    if(t->n_l2)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*t->n_l2);
    for(i = 0; i < t->n_l2; i++){
        l2[i] = copy_scaled_l2_norm(t->l2[i]);
    }
    model* m = copy_model_without_learning_parameters(t->m);
    model* m2 = copy_model_without_learning_parameters(t->linear_after_attention);
    return transformer_encoder_layer(m,m2,fcls,l2,t->input_dimension,t->n_head,t->residual_flag1,t->normalization_flag1,t->residual_flag2,t->normalization_flag2,t->attention_flag);
}

/* This function resets the arrays used during the feed forward and backpropagation by the transformer
 * 
 * Inputs:
 * 
 * 
 *                 @transformer_encoder* t:= the transformer encoder structure that must be rests
 * */
void reset_transformer_encoder(transformer_encoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        reset_fcl(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    for(i = 0; i < t->m->output_dimension; i++){
        t->encoder_output_error[i] = 0;
    }
    for(i = 0; i < t->input_dimension*t->dimension; i++){
        if(i < t->input_dimension){
            
            t->q[i] = 0;
            t->q_error[i] = 0;
            t->k[i] = 0;
            t->k_error[i] = 0;
            t->v[i] = 0;
            t->v_error[i] = 0;
            t->attention_output[i] = 0;
        }
        t->score_matrix[i] = 0;
        t->score_matrix_error[i] = 0;
        t->score_matrix_softmax[i] = 0;
        t->score_matrix_softmax_error[i] = 0;
    }
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->linear_after_attention->output_dimension; i++){
                t->residual1_output[i] = 0;
                t->residual1_output_error[i] = 0;
        }
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->m->output_dimension; i++){
                t->residual2_output[i] = 0;
                t->residual2_output_error[i] = 0;            
        }
    }
    reset_model(t->m);
    reset_model(t->linear_after_attention);
    return;
}
/* This function resets the arrays used during the feed forward and backpropagation by the transformer
 * 
 * Inputs:
 * 
 * 
 *                 @transformer_encoder* t:= the transformer encoder structure that must be rests
 * */
void reset_transformer_encoder_without_learning_parameters(transformer_encoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        reset_fcl_without_learning_parameters(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    for(i = 0; i < t->m->output_dimension; i++){
        t->encoder_output_error[i] = 0;
    }
    for(i = 0; i < t->input_dimension*t->dimension; i++){
        if(i < t->input_dimension){
            
            t->q[i] = 0;
            t->q_error[i] = 0;
            t->k[i] = 0;
            t->k_error[i] = 0;
            t->v[i] = 0;
            t->v_error[i] = 0;
            t->attention_output[i] = 0;
        }
        t->score_matrix[i] = 0;
        t->score_matrix_error[i] = 0;
        t->score_matrix_softmax[i] = 0;
        t->score_matrix_softmax_error[i] = 0;
    }
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->linear_after_attention->output_dimension; i++){
                t->residual1_output[i] = 0;
                t->residual1_output_error[i] = 0;
        }
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->m->output_dimension; i++){
                t->residual2_output[i] = 0;
                t->residual2_output_error[i] = 0;            
        }
    }
    reset_model_without_learning_parameters(t->m);
    reset_model_without_learning_parameters(t->linear_after_attention);
    return;
}
/* This function resets the arrays used during the feed forward and backpropagation by the transformer
 * 
 * Inputs:
 * 
 * 
 *                 @transformer_encoder* t:= the transformer encoder structure that must be rests
 * */
void reset_transformer_encoder_except_partial_derivatives(transformer_encoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        reset_fcl_except_partial_derivatives(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm_except_partial_derivatives(t->l2[i]);
    }
    for(i = 0; i < t->m->output_dimension; i++){
        t->encoder_output_error[i] = 0;
    }
    for(i = 0; i < t->input_dimension*t->dimension; i++){
        if(i < t->input_dimension){
            
            t->q[i] = 0;
            t->q_error[i] = 0;
            t->k[i] = 0;
            t->k_error[i] = 0;
            t->v[i] = 0;
            t->v_error[i] = 0;
            t->attention_output[i] = 0;
        }
        t->score_matrix[i] = 0;
        t->score_matrix_error[i] = 0;
        t->score_matrix_softmax[i] = 0;
        t->score_matrix_softmax_error[i] = 0;
    }
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->linear_after_attention->output_dimension; i++){
                t->residual1_output[i] = 0;
                t->residual1_output_error[i] = 0;
        }
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->m->output_dimension; i++){
                t->residual2_output[i] = 0;
                t->residual2_output_error[i] = 0;            
        }
    }
    reset_model_except_partial_derivatives(t->m);
    reset_model_except_partial_derivatives(t->linear_after_attention);
    return;
}

/* This function does exactly what the function above does but for the fully connected leyers inside
 * the transformer the reset is for the edge popup
 * 
 * Inputs:
 * 
 * 
 *             @transformer_encoder* t:= the transformer encoder that must be reset
 * */
void reset_transformer_encoder_for_edge_popup(transformer_encoder* t){
    if(t == NULL)
        return;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        reset_fcl_for_edge_popup(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    for(i = 0; i < t->m->output_dimension; i++){
        t->encoder_output_error[i] = 0;
    }
    for(i = 0; i < t->input_dimension*t->dimension; i++){
        if(i < t->input_dimension){
            
            t->q[i] = 0;
            t->q_error[i] = 0;
            t->k[i] = 0;
            t->k_error[i] = 0;
            t->v[i] = 0;
            t->v_error[i] = 0;
            t->attention_output[i] = 0;
        }
        t->score_matrix[i] = 0;
        t->score_matrix_error[i] = 0;
        t->score_matrix_softmax[i] = 0;
        t->score_matrix_softmax_error[i] = 0;
    }
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->linear_after_attention->output_dimension; i++){
                t->residual1_output[i] = 0;
                t->residual1_output_error[i] = 0;
        }
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        for(i = 0; i < t->m->output_dimension; i++){
                t->residual2_output[i] = 0;
                t->residual2_output_error[i] = 0;            
        }
    }
    reset_model_for_edge_popup(t->m);
    reset_model_for_edge_popup(t->linear_after_attention);
    return;
}

/* this function gives the number of bytes more or less occupied by this structure
 * 
 * Inputs:
 * 
 *                 @transformer_encoder* t:= the structure that must be sized
 * */
uint64_t size_of_transformer_encoder(transformer_encoder* t){
    uint64_t sum = 0;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        sum+=size_of_fcls(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        sum+=size_of_scaled_l2_norm(t->l2[i]);
    }
    
    sum+= t->input_dimension*8 + t->input_dimension*t->dimension*4;
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2;
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2;
    sum+=size_of_model(t->m)+size_of_model(t->linear_after_attention);
    return sum;    
}
/* this function gives the number of bytes more or less occupied by this structure
 * 
 * Inputs:
 * 
 *                 @transformer_encoder* t:= the structure that must be sized
 * */
uint64_t size_of_transformer_encoder_without_learning_parameters(transformer_encoder* t){
    uint64_t sum = 0;
    int i;
    for(i = 0; i < t->n_head*3; i++){
        sum+=size_of_fcls_without_learning_parameters(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        sum+=size_of_scaled_l2_norm(t->l2[i]);
    }
    
    sum+= t->input_dimension*8 + t->input_dimension*t->dimension*4;
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2;
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2;
    sum+=size_of_model_without_learning_parameters(t->m)+size_of_model_without_learning_parameters(t->linear_after_attention);
    return sum;    
}

/* This function, given 2 structures with the same number of layers will copy
 * the main features of the first into the second one
 * 
 * Inputs:
 * 
 *             @transformer_encoder* t:= the transformer encoder that must be copied
 *             @transformer_encoder* copy:= the transformer encoder structure in which will be copied t
 * */
void paste_transformer_encoder(transformer_encoder* t, transformer_encoder* copy){
    int i;
    for(i = 0; i < t->n_head*3; i++){
        paste_fcl(t->fcls[i],copy->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        paste_scaled_l2_norm(t->l2[i],copy->l2[i]);
    }
    paste_model(t->m,copy->m);
    paste_model(t->linear_after_attention,copy->linear_after_attention);
    copy->attention_flag = t->attention_flag;
}
/* This function, given 2 structures with the same number of layers will copy
 * the main features of the first into the second one
 * 
 * Inputs:
 * 
 *             @transformer_encoder* t:= the transformer encoder that must be copied
 *             @transformer_encoder* copy:= the transformer encoder structure in which will be copied t
 * */
void slow_paste_transformer_encoder(transformer_encoder* t, transformer_encoder* copy, float tau){
    int i;
    for(i = 0; i < t->n_head*3; i++){
        slow_paste_fcl(t->fcls[i],copy->fcls[i],tau);
    }
    for(i = 0; i < t->n_l2; i++){
        slow_paste_scaled_l2_norm(t->l2[i],copy->l2[i],tau);
    }
    slow_paste_model(t->m,copy->m,tau);
    slow_paste_model(t->linear_after_attention,copy->linear_after_attention,tau);
    copy->attention_flag = t->attention_flag;
}


/* This function computes the feed forward of our encoder transformer network
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension:= the dimension of the inputs
 * */
void encoder_transformer_ff(float* inputs, transformer_encoder* t, int input_dimension){
    int i;
    for(i = 0; i < t->n_head; i++){
        if(input_dimension != t->fcls[i*3]->input || input_dimension != t->fcls[i*3+1]->input || input_dimension != t->fcls[i*3+2]->input){
            fprintf(stderr,"Error: your fully connected layers don't match the inputs - inputs given: %d, %d, %d, inputs fcls: %d,%d,%d\n",input_dimension,input_dimension,input_dimension,t->fcls[i*3]->input,t->fcls[i*3+1]->input,t->fcls[i*3+2]->input);
            exit(0);
        }
        if(t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
            fully_connected_feed_forward(inputs,&t->q[i*t->dimension],t->fcls[i*3]->weights,NULL,input_dimension,t->dimension);
        else if(t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
            fully_connected_feed_forward_edge_popup(inputs,&t->q[i*t->dimension],t->fcls[i*3]->weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->indices,t->fcls[i*3]->k_percentage*t->fcls[i*3]->input*t->fcls[i*3]->output);
        if(t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
            fully_connected_feed_forward(inputs,&t->k[i*t->dimension],t->fcls[i*3+1]->weights,NULL,input_dimension,t->dimension);
        else if(t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
            fully_connected_feed_forward_edge_popup(inputs,&t->k[i*t->dimension],t->fcls[i*3+1]->weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
        if(t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
            fully_connected_feed_forward(inputs,&t->v[i*t->dimension],t->fcls[i*3+2]->weights,NULL,input_dimension,t->dimension);
        else if(t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
            fully_connected_feed_forward_edge_popup(inputs,&t->v[i*t->dimension],t->fcls[i*3+2]->weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
    model_tensor_input_ff(t->linear_after_attention,1,1,t->input_dimension,t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        if(t->linear_after_attention->output_dimension != input_dimension){
            fprintf(stderr,"Error: the inputs of the transformer don't match the multi headed attention output!\n");
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
                if(t->input_dimension != t->m->output_dimension){
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
                if(t->input_dimension != t->m->output_dimension){
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
/* This function computes the feed forward of our encoder transformer network
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension:= the dimension of the inputs
 * */
void encoder_transformer_ff_opt(float* inputs, transformer_encoder* t, int input_dimension, transformer_encoder* t2){
    int i;
    for(i = 0; i < t->n_head; i++){
        if(input_dimension != t->fcls[i*3]->input || input_dimension != t->fcls[i*3+1]->input || input_dimension != t->fcls[i*3+2]->input){
            fprintf(stderr,"Error: your fully connected layers don't match the inputs - inputs given: %d, %d, %d, inputs fcls: %d,%d,%d\n",input_dimension,input_dimension,input_dimension,t->fcls[i*3]->input,t->fcls[i*3+1]->input,t->fcls[i*3+2]->input);
            exit(0);
        }
        if(t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
            fully_connected_feed_forward(inputs,&t->q[i*t->dimension],t2->fcls[i*3]->weights,NULL,input_dimension,t->dimension);
        else if(t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
            fully_connected_feed_forward_edge_popup(inputs,&t->q[i*t->dimension],t2->fcls[i*3]->weights,NULL,input_dimension,t->dimension,t2->fcls[i*3]->indices,t2->fcls[i*3]->k_percentage*t2->fcls[i*3]->input*t2->fcls[i*3]->output);
        if(t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
            fully_connected_feed_forward(inputs,&t->k[i*t->dimension],t2->fcls[i*3+1]->weights,NULL,input_dimension,t->dimension);
        else if(t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
            fully_connected_feed_forward_edge_popup(inputs,&t->k[i*t->dimension],t2->fcls[i*3+1]->weights,NULL,input_dimension,t->dimension,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
        if(t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
            fully_connected_feed_forward(inputs,&t->v[i*t->dimension],t2->fcls[i*3+2]->weights,NULL,input_dimension,t->dimension);
        else if(t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
            fully_connected_feed_forward_edge_popup(inputs,&t->v[i*t->dimension],t2->fcls[i*3+2]->weights,NULL,input_dimension,t->dimension,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);
    }
    multi_head_attention_ff(t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,t->attention_output,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
    model_tensor_input_ff_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        if(t->linear_after_attention->output_dimension != input_dimension){
            fprintf(stderr,"Error: the inputs of the transformer don't match the multi headed attention output!\n");
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
                if(t->input_dimension != t->m->output_dimension){
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
                if(t->input_dimension != t->m->output_dimension){
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

/* This function computes the back propagation of our encoder transformer network
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension:= the dimension of the inputs
 *                @ float* output_error:= the error for the output
 * it returns the error for the inputs
 * */
float* encoder_transformer_bp(float* inputs, transformer_encoder* t, int input_dimension,float* output_error){
    int i;
    float* temp, *tempp;
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
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,t->l2[0]->output_error,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->residual1_output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->linear_after_attention->output_layer,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->l2[t->n_l2-1]->input_dimension);
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,temp,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
        }
        
    }
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->l2[0]->output,output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,temp,t->l2[0]->output_error);
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,t->l2[0]->output_error,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->linear_after_attention->output_dimension,t->residual1_output,output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->linear_after_attention->output_dimension,t->linear_after_attention->output_layer,output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t->fcls[i*3]->indices,t->fcls[i*3]->input*t->fcls[i*3]->output*t->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t->fcls[i*3+1]->indices,t->fcls[i*3+1]->input*t->fcls[i*3+1]->output*t->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t->fcls[i*3+2]->indices,t->fcls[i*3+2]->input*t->fcls[i*3+2]->output*t->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,temp,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
        }
        
    }
    
    
}
/* This function computes the back propagation of our encoder transformer network
 * 
 * 
 * Inputs:
 * 
 *             @ float* inputs:= the inputs coming from the bottom of the transformer, the input dimension
 *             @ transformer_encoder* t:= our encoder transformer
 *             @ int input_dimension:= the dimension of the inputs
 *                @ float* output_error:= the error for the output
 * it returns the error for the inputs
 * */
float* encoder_transformer_bp_opt(float* inputs, transformer_encoder* t, int input_dimension,float* output_error, transformer_encoder* t2){
    int i;
    float* temp, *tempp;
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
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,t->l2[0]->output_error,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->residual1_output,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp(t->m,1,1,t->input_dimension,t->linear_after_attention->output_layer,t->l2[t->n_l2-1]->output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,t->l2[t->n_l2-1]->output_error,temp,t->l2[t->n_l2-1]->input_dimension);
            tempp = model_tensor_input_bp(t->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,temp,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
        }
        
    }
    else{
        if(t->normalization_flag1 == SCALED_L2_NORMALIZATION){
            temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->input_dimension,t->l2[0]->output,output_error,t->m->output_dimension);
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->residual1_output,temp,t->l2[0]->output_error);
            }
            else
                back_propagation_scaled_l2_norm(t->l2[0]->input_dimension,t->l2[0]->learned_g,&t->l2[0]->d_learned_g,t->l2[0]->norm,t->linear_after_attention->output_layer,temp,t->l2[0]->output_error);
            tempp = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,t->l2[0]->output_error,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,t->l2[0]->output_error,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
            
        }
        else{
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
                temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->linear_after_attention->output_dimension,t->residual1_output,output_error,t->m->output_dimension);
            else    
                temp = model_tensor_input_bp_without_learning_parameters(t->m,t2->m,1,1,t->linear_after_attention->output_dimension,t->linear_after_attention->output_layer,output_error,t->m->output_dimension);
            
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
                sum1D(temp,output_error,temp,t->linear_after_attention->output_dimension);
            
            tempp = model_tensor_input_bp_without_learning_parameters(t->linear_after_attention,t2->linear_after_attention,1,1,t->input_dimension,t->attention_output,temp,t->linear_after_attention->output_dimension);
            multi_head_attention_bp(t->q_error,t->k_error,t->v_error,t->score_matrix_error,t->score_matrix_softmax_error,t->q,t->k,t->v,t->score_matrix,t->score_matrix_softmax,tempp,t->dimension,t->n_head,t->input_dimension,t->attention_flag);
            for(i = 0; i < t->n_head; i++){
                if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3]->training_mode != EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                else if(t->fcls[i*3]->training_mode == EDGE_POPUP && t->fcls[i*3]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->q_error[i*t->dimension],t2->fcls[i*3]->weights,t->fcls[2]->temp2,t->fcls[i*3]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3]->d_scores,t2->fcls[i*3]->indices,t2->fcls[i*3]->input*t2->fcls[i*3]->output*t2->fcls[i*3]->k_percentage);
                
                if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension);
                else if(t->fcls[i*3+1]->training_mode != EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                else if(t->fcls[i*3+1]->training_mode == EDGE_POPUP && t->fcls[i*3+1]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->k_error[i*t->dimension],t2->fcls[i*3+1]->weights,t->fcls[2]->temp2,t->fcls[i*3+1]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+1]->d_scores,t2->fcls[i*3+1]->indices,t2->fcls[i*3+1]->input*t2->fcls[i*3+1]->output*t2->fcls[i*3+1]->k_percentage);
                
                if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == FULLY_FEED_FORWARD)
                    fully_connected_back_prop(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension);   
                else if(t->fcls[i*3+2]->training_mode != EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup_ff_gd_bp(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
                else if(t->fcls[i*3+2]->training_mode == EDGE_POPUP && t->fcls[i*3+2]->feed_forward_flag == EDGE_POPUP)
                    fully_connected_back_prop_edge_popup(inputs,&t->v_error[i*t->dimension],t2->fcls[i*3+2]->weights,t->fcls[2]->temp2,t->fcls[i*3+2]->d_weights,NULL,input_dimension,t->dimension,t->fcls[i*3+2]->d_scores,t2->fcls[i*3+2]->indices,t2->fcls[i*3+2]->input*t2->fcls[i*3+2]->output*t2->fcls[i*3+2]->k_percentage);   
            }
            
            if (t->residual_flag1 == TRANSFORMER_RESIDUAL)
                sum1D(t->fcls[2]->temp2,temp,t->fcls[2]->temp2,input_dimension);
            return t->fcls[2]->temp2;
        }
        
    }
    
    
}





