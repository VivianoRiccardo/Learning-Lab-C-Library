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
 * This function returns a vector that must be summed up with the embeddings of the input for the new input in a transformer.
 * Is based on the positional encoding described in the papre attention is all you need
 * 
 * Inputs:
 * 
 *             @ int embedding_dimension:= the dimension of the embedding of each token of your sequence, 
 *                                         for example if the sequence is a sentence and each token is a word,
 *                                         then if each token is rapresented by a k-dimensional array, k is the embedding dimension
 *             @ int sequence_length:= the name of this value is self-explanatory
 * 
 * */
float* sin_cos_positional_encoding_vector(int embedding_dimension, int sequence_length){
    if (embedding_dimension%2){
        fprintf(stderr,"Error, embedding_dimension must be an even number!\n");
        exit(1);
    }
    float* pos_vect = (float*)calloc(sequence_length*embedding_dimension,sizeof(float));
    int i,j;
    for(i = 0; i < sequence_length; i++){
        for(j = 0; j < embedding_dimension; j+=2){
            pos_vect[i*embedding_dimension + j] = (float)(sin((double)i / (double)((pow((double)(10000),(double)2*j) / embedding_dimension))));
            pos_vect[i*embedding_dimension + j + 1] = (float)(cos((double)i / (double)((pow((double)(10000),(double)2*(j+1)) / embedding_dimension))));
        }
    }
    
    return pos_vect;
}
