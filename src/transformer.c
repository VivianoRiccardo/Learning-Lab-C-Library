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
 * 
 *             @ fcl** fcls:= the fully connected layers, we should have n_head fully connected layers without any activation function for the queries, the keys, and the values at the
 *                            beginning of the transformer encoder layer, becase each query, value and key must pass through a linear matrix given by the fully connected layers weights
 *                            then the encoder needs 2 other fully connected layer after the self- attention, remember the layer before the last one must have an activation function
 *                            (Relu/Leaky Relu/ Elu / BLEU suggested) and the last one should not have any activation function. Dimensions 3*n_head+2
 *            @ scaled_l2_norm** l2:= this layer is used as normalization layer instead of a layer normalization layer because of this paper: Transformers without Tears:Improving the Normalization of Self-Attention
 *                                     future implementation with fixed normalization too or simply cosine normalization with learnable parameter will be implemented (maybe), dimensions: 0,1 or 2
 *             @ int input_dimension:= the total dimension of the input completly flatten
 *             @ int n_head:= number of head attention
 *             @ int residual_flag1:= TRANSFORMER_RESIDUAL or TRANSFORMER_NO_RESIDUAL
 *             @ int normalization_flag1:= SCALED_L2_NORMALIZATION or NO_NORMALIZATION
 *             @ int residual_flag2:= TRANSFORMER_RESIDUAL or TRANSFORMER_NO_RESIDUAL
 *             @ int normalization_flag2:= SCALED_L2_NORMALIZATION or NO_NORMALIZATION
 *             */
transformer_encoder* transformer_encoder_layer(fcl** fcls, scaled_l2_norm** l2, int input_dimension, int n_head,int residual_flag1,int normalization_flag1,int residual_flag2,int normalization_flag2){
    if(fcls == NULL){
        fprintf(stderr,"Error: there must be 3*n_head + 2 fully connected layers\n");
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
    
    
    t->fcls = fcls;
    t->l2 = l2;
    t->input_dimension = input_dimension;
    t->n_head;
    t->residual_flag1 = residual_flag1;
    t->residual_flag2 = residual_flag2;
    t->normalization_flag1 = normalization_flag1;
    t->normalization_flag2 = normalization_flag2;
    t->dimension = input_dimension/n_head;
    t->incoming_input = (float*)calloc(input_dimension,sizeof(float));
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
    t->attention_output_error = (float*)calloc(input_dimension,sizeof(float));
    if(residual_flag1 == TRANSFORMER_RESIDUAL){
        t->residual1_output = (float*)calloc(input_dimension,sizeof(float));
        t->residual1_output_error = (float*)calloc(input_dimension,sizeof(float));
    }
    if(residual_flag2 == TRANSFORMER_RESIDUAL){
        t->residual2_output = (float*)calloc(input_dimension,sizeof(float));
        t->residual2_output_error = (float*)calloc(input_dimension,sizeof(float));
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
    for(i = 0; i < t->n_head*3 + 2; i++){
        free_fully_connected(t->fcls[i]);
    }
    free(t->fcls);
    int n_l2 = 0;
    if(t->normalization_flag1 == SCALED_L2_NORMALIZATION)
        n_l2++;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION)
        n_l2++;
    for(i = 0; i < n_l2; i++){
        free_scaled_l2_normalization_layer(t->l2[i]);
    }
    
    free(t->l2);
    free(t->incoming_input);
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
    free(t->attention_output_error);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        free(t->residual1_output);
        free(t->residual1_output_error);
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        free(t->residual2_output);
        free(t->residual2_output_error);
    }
    
    return;
}
/* This function deallocates the space allocated by useless arrays for the fully connected layers inside the transformer
 * during the edge popup training and test
 * 
 * Input:
 *             @ transformer_encoder* t:= the transformer encoder
 * */
void free_transformer_encoder_layer_for_edge_popup(transformer_encoder* t){
    int i;
    if (t == NULL)
        return;
    for(i = 0; i < t->n_head*3 + 2; i++){
        free_fully_connected_for_edge_popup(t->fcls[i]);
    }
    
    return;
}


/* This function deallocates the space allocated by a transformer encoder and the arrays used for edge popup for the fully
 * connected layers insider the encoder. (this + free_transformer_ancoder_layer_for_edge_popup = free_transformer_encoder_layer)
 * 
 * Input:
 *             @ transformer_encoder* t:= the transformer encoder
 * */
void free_transformer_encoder_layer_complementary_edge_popup(transformer_encoder* t){
    int i;
    if (t == NULL)
        return;
    for(i = 0; i < t->n_head*3 + 2; i++){
        free_fully_connected_complementary_edge_popup(t->fcls[i]);
    }
    free(t->fcls);
    int n_l2 = 0;
    if(t->normalization_flag1 == SCALED_L2_NORMALIZATION)
        n_l2++;
    if(t->normalization_flag2 == SCALED_L2_NORMALIZATION)
        n_l2++;
    for(i = 0; i < n_l2; i++){
        free_scaled_l2_normalization_layer(t->l2[i]);
    }
    
    free(t->l2);
    free(t->incoming_input);
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
    free(t->attention_output_error);
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
        free(t->residual1_output);
        free(t->residual1_output_error);
    }
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
        free(t->residual2_output);
        free(t->residual2_output_error);
    }
    
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
    
    for(i = 0; i < t->n_head*3 + 2; i++){
        save_fcl(t->fcls[i],n);
    }

    for(i = 0; i < t->n_l2; i++){
        save_scaled_l2_norm(t->l2[i],n);
    }
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
    int input_dimension = 0,n_head = 0,residual_flag1 = 0,normalization_flag1 = 0,residual_flag2 = 0,normalization_flag2 = 0;
    
    
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
    
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*n_head*3+2);
    scaled_l2_norm** l2 = NULL;
    int count = 0;
    if(normalization_flag1 == SCALED_L2_NORMALIZATION)
        count++;
    if(normalization_flag2 == SCALED_L2_NORMALIZATION)
        count++;
    for(i = 0; i < n_head*3 + 2; i++){
        fcls[i] = load_fcl(fr);
    }
    if(count)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*count);
    for(i = 0; i < count; i++){
        l2[i] = load_scaled_l2_norm(fr);
    }
    
    return transformer_encoder_layer(fcls,l2,input_dimension,n_head,residual_flag1,normalization_flag1,residual_flag2,normalization_flag2);
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
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*t->n_head*3+2);
    scaled_l2_norm** l2 = NULL;
    int i;
    for(i = 0; i < t->n_head*3+2; i++){
        fcls[i] = copy_fcl(t->fcls[i]);
    }
    
    if(t->n_l2)
        l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*t->n_l2);
    for(i = 0; i < t->n_l2; i++){
        l2[i] = copy_scaled_l2_norm(t->l2[i]);
    }
    
    return transformer_encoder_layer(fcls,l2,t->input_dimension,t->n_head,t->residual_flag1,t->normalization_flag1,t->residual_flag2,t->normalization_flag2);
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
    for(i = 0; i < t->n_head; i++){
        reset_fcl(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    
    for(i = 0; i < t->input_dimension*t->dimension; i++){
        if(i < t->input_dimension){
            t->incoming_input[i] = 0;
            t->q[i] = 0;
            t->q_error[i] = 0;
            t->k[i] = 0;
            t->k_error[i] = 0;
            t->v[i] = 0;
            t->v_error[i] = 0;
            t->attention_output[i] = 0;
            t->attention_output_error[i] = 0;
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                t->residual1_output[i] = 0;
                t->residual1_output_error[i] = 0;
            }
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                t->residual2_output[i] = 0;
                t->residual2_output_error[i] = 0;
            }
        }
        t->score_matrix[i] = 0;
        t->score_matrix_error[i] = 0;
        t->score_matrix_softmax[i] = 0;
        t->score_matrix_softmax_error[i] = 0;
    }
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
    for(i = 0; i < t->n_head; i++){
        reset_fcl_for_edge_popup(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        reset_scaled_l2_norm(t->l2[i]);
    }
    
    for(i = 0; i < t->input_dimension*t->dimension; i++){
        if(i < t->input_dimension){
            t->incoming_input[i] = 0;
            t->q[i] = 0;
            t->q_error[i] = 0;
            t->k[i] = 0;
            t->k_error[i] = 0;
            t->v[i] = 0;
            t->v_error[i] = 0;
            t->attention_output[i] = 0;
            t->attention_output_error[i] = 0;
            if(t->residual_flag1 == TRANSFORMER_RESIDUAL){
                t->residual1_output[i] = 0;
                t->residual1_output_error[i] = 0;
            }
            if(t->residual_flag2 == TRANSFORMER_RESIDUAL){
                t->residual2_output[i] = 0;
                t->residual2_output_error[i] = 0;
            }
        }
        t->score_matrix[i] = 0;
        t->score_matrix_error[i] = 0;
        t->score_matrix_softmax[i] = 0;
        t->score_matrix_softmax_error[i] = 0;
    }
    return;
}

/* this function gives the number of bytes more or less occupied by this structure
 * 
 * Inputs:
 * 
 *                 @transformer_encoder* t:= the structure that must be sized
 * */
unsigned long long int size_of_transformer_encoder(transformer_encoder* t){
    long long unsigned int sum = 0;
    int i;
    for(i = 0; i < t->n_head*3+2; i++){
        sum+=size_of_fcls(t->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        sum+=size_of_scaled_l2_norm(t->l2[i]);
    }
    
    sum+= t->input_dimension*9 + t->input_dimension*t->dimension*4;
    if(t->residual_flag1 == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2;
    if(t->residual_flag2 == TRANSFORMER_RESIDUAL)
        sum+=t->input_dimension*2;
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
    for(i = 0; i < t->n_head*3+2; i++){
        paste_fcl(t->fcls[i],copy->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        paste_scaled_l2_norm(t->l2[i],copy->l2[i]);
    }
}

/* This function does exactly what the function above does but for the fully connected layers inside
 * the structure will be copied the main features for the edge popup training
 * 
 * Inputs:
 *                 
 *             @transformer_encoder* t:= the transformer encoder structure that must be copied
 *             @transformer_encoder* copy:= the transformer encoder structure in which will be copied t
 * */
void paste_transformer_encoder_for_edge_popup(transformer_encoder* t, transformer_encoder* copy){
    int i;
    for(i = 0; i < t->n_head*3+2; i++){
        paste_fcl_for_edge_popup(t->fcls[i],copy->fcls[i]);
    }
    for(i = 0; i < t->n_l2; i++){
        paste_scaled_l2_norm(t->l2[i],copy->l2[i]);
    }
}

