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


/* This function create 7 activation nodes with every node being a different activation (or no activation)
 * output:= aco_node** a, size: 7 
 * */
aco_node** aco_create_activations_node(){
    activation* a1 = init_activation(NO_ACTIVATION);
    activation* a2 = init_activation(SIGMOID);
    activation* a3 = init_activation(RELU);
    //activation* a4 = init_activation(SOFTMAX);
    activation* a5 = init_activation(TANH);
    activation* a6 = init_activation(LEAKY_RELU);
    activation* a7 = init_activation(ELU);
    
    aco_node* an1 = init_aco_node(NULL,NULL,a1,NULL);
    aco_node* an2 = init_aco_node(NULL,NULL,a2,NULL);
    aco_node* an3 = init_aco_node(NULL,NULL,a3,NULL);
    //aco_node* an4 = init_aco_node(NULL,NULL,a4,NULL);
    aco_node* an5 = init_aco_node(NULL,NULL,a5,NULL);
    aco_node* an6 = init_aco_node(NULL,NULL,a6,NULL);
    aco_node* an7 = init_aco_node(NULL,NULL,a7,NULL);
    aco_node** a = (aco_node**)malloc(sizeof(aco_node*)*7);
    a[0] = an1;
    a[1] = an2;
    a[2] = an3;
    //a[3] = an4;
    a[3] = an7;
    a[4] = an5;
    a[5] = an6;
    return a;
}

aco_node* aco_create_activation_node(int activation_flag){
    activation* a1 = init_activation(activation_flag);
    aco_node* an1 = init_aco_node(NULL,NULL,a1,NULL);
    return an1;
}

/* This function attach a fully connected node to 7 activation nodes
 * 
 * Inputs:
 * 
 *             aco_node* f:= the fully connected node
 *             aco_node** a:= the 7 activations, dimension: 7
 * */
void aco_attach_activations_to_fcl(aco_node* f, aco_node** a){
    
    int i;
    aco_edge* e = init_aco_edge(f,a[0],ACO_OPERATION_EXECUTE_NODE);
    add_aco_edge(f,e,0);
    add_aco_edge(a[0],e,1);
    e = init_aco_edge(f,a[1],ACO_OPERATION_EXECUTE_NODE);
    add_aco_edge(f,e,0);
    add_aco_edge(a[1],e,1);
    e = init_aco_edge(f,a[2],ACO_OPERATION_EXECUTE_NODE);
    add_aco_edge(f,e,0);
    add_aco_edge(a[2],e,1);
    e = init_aco_edge(f,a[3],ACO_OPERATION_EXECUTE_NODE);
    add_aco_edge(f,e,0);
    add_aco_edge(a[3],e,1);
    e = init_aco_edge(f,a[4],ACO_OPERATION_EXECUTE_NODE);
    add_aco_edge(f,e,0);
    add_aco_edge(a[4],e,1);
    e = init_aco_edge(f,a[5],ACO_OPERATION_EXECUTE_NODE);
    add_aco_edge(f,e,0);
    add_aco_edge(a[5],e,1);
    //e = init_aco_edge(f,a[6],ACO_OPERATION_EXECUTE_NODE);
    //add_aco_edge(f,e,0);
    //add_aco_edge(a[6],e,1);
}


/* This function attach a fcl node to some aco_nodes
 * 
 * Inputs:
 * 
 *                 @ aco_node** n:= the inputs node that will be attached to a fully connected node, dimension: size
 *                 @ int size:= the number of input nodes
 *                 @ int input_size:= the input size of the fully connected node that will be created
 *                 @ int output_size:= the output size of the fully connected node that will be created
 * 
 * Returns:
 * 
 *                 @ aco_node* fcl_node:= the fully connected node created
 * */ 
aco_node* aco_attach_fcl_to_params(aco_node** n, int size, int input_size, int output_size){
    fcl_func* f = init_fcl_func(input_size,output_size);
    aco_node* fcl_node = init_aco_node(NULL,NULL,NULL,f);
    aco_edge* e = NULL;
    int i;
    for(i = 0; i < size; i++){
        e = init_aco_edge(n[i],fcl_node,ACO_OPERATION_EXECUTE_NODE);
        add_aco_edge(n[i],e,0);
        add_aco_edge(fcl_node,e,1);
    }
    return fcl_node;
}

// number of nodes returned: 4 * width * depth
/* This function initializes some parameter nodes
 * 
 * Inputs:
 * 
 *                 @ int input_size:= is a multiplier factor for the size of the params (input_size*output_size)
 *                 @ int output_size:= is a multiplier factor for the size of the params (input_size*output_size)
 *                 @ int width:= is the number of same parameters in the same sub layer
 *                 @ int depth:= is the number of sublayers
 * 
 * returns:
 * 
 *                 @ aco_node** n:= is the number of nodes created (depth*width*4 - width*3) depth*width weights params input_size*output_size
 *                                                                                 depth*width biases params output_size
 *                                                                                 depth*width weights params 1
 *                                                                                 depth*width biases params 1
 * */
aco_node** aco_build_layer_param_nodes(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1){
    if(!width || depth < 3 || (input_size*output_size)%sub_matrix_dimension1){
        return NULL;
    }
    int i,j;
    aco_node** n = (aco_node**)malloc(sizeof(aco_node*)*4*width*depth - width*3);
    
    for(i = 0; i < 4*width*depth - width*3; i++){
        if((((int)(i/width)))%4 != 0 && (((int)(i/width)))%4 != 1){
            params* p1 = NULL;// single for weights
            params* p2 = NULL;// single for biases
            if( ((((int)(i/width)))%4) == 2)
                p1 = init_params(1,input_size,0,0,0);
            else
                p2 = init_params(1,input_size,0,0,0);
            n[i] = init_aco_node(p1,p2,NULL,NULL);
        }
        else{
            params* p1 = NULL;// multiple for weights
            params* p2 = NULL;// multiple for biases
            if( ((((int)(i/width)))%4) == 0){
                if((((int)(i/width))) == 0)
                    p1 = init_params(input_size*sub_matrix_dimension1, input_size, input_size, sub_matrix_dimension1, 1);
                else if((((int)(i/(width*4)))) == depth-1)
                    p1 = init_params(sub_matrix_dimension1*output_size, input_size, sub_matrix_dimension1, output_size, 1);
                else
                    p1 = init_params(sub_matrix_dimension1*sub_matrix_dimension1, input_size, sub_matrix_dimension1, sub_matrix_dimension1, 1);
            }
            else
                p2 = init_params(output_size, input_size,1,1,1);
            n[i] = init_aco_node(p1,p2,NULL,NULL);
        }
    }
    for(i = 0; i < 4*width*depth - width*3; i++){
        for(j = i+(width-(i%width)); j < 4*width*depth - width*3; j++){
            aco_edge* e1;
            aco_edge* e2;
            aco_edge* e3;
            
            if((int)(i/width) == 0){
                if((int)(j/width) == 1){
                    e3 = init_aco_edge(n[i],n[j],ACO_OPERATION_COPY);
                    add_aco_edge(n[i],e3,0);
                    add_aco_edge(n[j],e3,1);
                }
                if((int)(j/width) > 1)
                    break;
            }
            else{
                if(node_state(n[j]) == ACO_IS_WEIGHT && n[j]->weights->size != 1){
                    e1 = init_aco_edge(n[i],n[j],ACO_OPERATION_MATRIX_MUL);
                    add_aco_edge(n[i],e1,0);
                    add_aco_edge(n[j],e1,1);
                }
                
                else if(node_state(n[j]) == ACO_IS_BIAS && n[j]->biases->size != 1){
                    e1 = init_aco_edge(n[i],n[j],ACO_OPERATION_MUL);
                    e2 = init_aco_edge(n[i],n[j],ACO_OPERATION_SUM);
                    e3 = init_aco_edge(n[i],n[j],ACO_OPERATION_COPY);
                    add_aco_edge(n[i],e1,0);
                    add_aco_edge(n[i],e2,0);
                    add_aco_edge(n[i],e3,0);
                    add_aco_edge(n[j],e1,1);
                    add_aco_edge(n[j],e2,1);
                    add_aco_edge(n[j],e3,1);
                }
                
                else{
                    e1 = init_aco_edge(n[i],n[j],ACO_OPERATION_MUL);
                    e2 = init_aco_edge(n[i],n[j],ACO_OPERATION_SUM);
                    add_aco_edge(n[i],e1,0);
                    add_aco_edge(n[i],e2,0);
                    add_aco_edge(n[j],e1,1);
                    add_aco_edge(n[j],e2,1);
                }
            }
        }
    }
    
    return n;
} 

aco_node** aco_build_layer_param_nodes_complete(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1){
    if(!width){
        return NULL;
    }
    int i,j;
    depth = input_size*output_size;
    aco_node** n = (aco_node**)malloc(sizeof(aco_node*)*width*depth);
    
    for(i = 0; i < width*depth; i++){
        params* w = init_params(1, input_size, input_size, output_size, (int)i/width);
        if(!(i%width))
            w->single_p = 0;
        n[i] = init_aco_node(w,NULL,NULL,NULL);
    }
    for(i = 0; i < width*(depth-1); i++){
        for(j = i+(width-(i%width)); j < i+(width-(i%width))+width; j++){
            aco_edge* e = init_aco_edge(n[i],n[j],ACO_OPERATION_TAKE);
            add_aco_edge(n[i],e,0);
            add_aco_edge(n[j],e,1);
        }
    }
    return n;
} 

aco_node** aco_build_layer_param_nodes_complete2(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1){
    if(!width){
        return NULL;
    }
    int i,j;
    depth = input_size*output_size;
    aco_node** n = (aco_node**)malloc(sizeof(aco_node*)*width*input_size*output_size);
    
    for(i = 0; i < width*input_size*output_size; i++){
        params* w = init_params(1, input_size, input_size, output_size, (int)i/width);
        if(!(i%width))
            w->single_p = 0;
        n[i] = init_aco_node(w,NULL,NULL,NULL);
    }
    for(i = 0; i < width*(input_size*output_size-1); i++){
        for(j = i+(width-(i%width)); j < i+(width-(i%width))+width; j++){
            aco_edge* e = init_aco_edge(n[i],n[j],ACO_OPERATION_INDEX_COPY);
            add_aco_edge(n[i],e,0);
            add_aco_edge(n[j],e,1);
        }
    }
    return n;
} 


/* This function creates a layer of sublayers according to the inputs and returns the
 * nodes needed to be attached to the root node
 * 
 * Inputs:
 * 
 *             int* input_sizes:= the sizes possible input sizes of the final fcl nodes of the layer, size: sizes
 *             int* output_sizes:= the sizes of possible output sizes of the final fcl nodes of the layer, size: sizes
 *             int width:= the width for the params
 *             int depth:= the depth for the params
 *             int size:= the sizes of input_sizes and output_sizes
 * 
 * Returns:
 * 
 *             aco_node** ret:= the nodes to attach to the root, size: 4
 * PAY ATTENTION input_sizes[i]*output_sizes[i] = input_size[j]*output_sizes[j] for each i,j
 * */
aco_node** get_nodes_to_attach(int input_size, int output_size, int width, int depth, int sub_matrix_dimension1, int activations){
    int i, counter;
    //aco_node** n = aco_build_layer_param_nodes(input_size, output_size, width, depth, sub_matrix_dimension1);// create parameters
    //aco_node** n = aco_build_layer_param_nodes_complete(input_size, output_size, width, depth, sub_matrix_dimension1);// create parameters
    aco_node** n = aco_build_layer_param_nodes_complete2(input_size, output_size, width, depth, sub_matrix_dimension1);// create parameters
    //aco_node* f = aco_attach_fcl_to_params(&n[4*width*(depth-1)], width, input_size, output_size);// attach params to fcl computations
    //aco_node* f = aco_attach_fcl_to_params(&n[width*(input_size*output_size-1)], width, input_size, output_size);// attach params to fcl computations
    aco_node* f = aco_attach_fcl_to_params(&n[width*(input_size*output_size-1)], width, input_size, output_size);// attach params to fcl computations
    if(activations == ALL_ACTIVATIONS){
        aco_node** a = aco_create_activations_node();// create activations
        aco_attach_activations_to_fcl(f, a);// attach activations to fcl computations
        free(a);
    }
    else{
        aco_node* a = aco_create_activation_node(activations);
        aco_add_output_nodes_to_node(f, &a, 1, ACO_OPERATION_EXECUTE_NODE);
    }
    return n;
}

/* This function just add some nodes as output for a single input node
 * 
 * Inputs:
 * 
 *             aco_node* n:= theinput node
 *             aco_node** ns:= the output nodes, size: size
 *             int size:= the number of output nodes
 *             int operation_flag:= the operation flag of the edge that will be created for the attachement
 * */
void aco_add_output_nodes_to_node(aco_node* n, aco_node** ns, int size, int operation_flag){
    int i;
    for(i = 0; i < size; i++){
        aco_edge* e = init_aco_edge(n,ns[i],operation_flag);
        add_aco_edge(n,e,0);
        add_aco_edge(ns[i],e,1);
    }
}


/* This function will attach nodes to the extreme components of the root node,
 * this means that will be reached all the leaf nodes of the graph where root is the root node and will be attached the nodes
 * passed as inputs
 * 
 * Inputs:
 * 
 *                 aco_node* root:= the root node
 *                 aco_node** ns:= the nodes that will be attached, size:size
 *                 int size:= the number of nodes that will be attached
 * */
 /*
int attach_params_to_root(aco_node* root, aco_node** ns, int size){
    if(root == NULL)
        return 1;
    if(root->flag)
        return 0;
    int i;
    root->flag = 1;
    if(!root->n_outputs){
        aco_add_output_nodes_to_node(root,ns,size,ACO_OPERATION_COPY);
        return 0;
    } 
    for(i = 0; i < root->n_outputs; i++){
        if(attach_params_to_root(root->outputs[i]->output, ns, size)){
            aco_add_output_nodes_to_node(root->outputs[i]->output,ns,size,ACO_OPERATION_COPY);
        }
    }
    return 0;
}*/
int attach_params_to_root(aco_node* root, aco_node** ns, int size){
    if(root == NULL)
        return 1;
    if(root->flag)
        return 0;
    if(!root->n_outputs){
        aco_add_output_nodes_to_node(root,ns,size,ACO_OPERATION_COPY);
        return 1;
    }
    int i;
    aco_node* n;
    for(n = root; !(node_state(n) == ACO_IS_FCL && !n->outputs[0]->output->n_outputs); n = n->outputs[0]->output);
    
    for(i = 0; i < n->n_outputs; i++){
        aco_add_output_nodes_to_node(n->outputs[i]->output,ns,size,ACO_OPERATION_COPY);
    }
    return 1;
}

// sizes is the size of input and output
/* this function just create a new layer from scratch and attachs it to a root node
 * as the function before explained
 * 
 * Inputs:
 * 
 *                 aco_node* root:= the root node
 *                 int* input_sizes:= see get_nodes_to_attach function, size: sizes
 *                 int* output_sizes:= see get_nodes_to_attach function, size: sizes
 *                 int width:= see get_nodes_to_attach function
 *                 int depth:= see get_nodes_to_attach function
 *                 int sizes:= see get_nodes_to_attach function
 *                 
 * */
aco_node** attach_layer(aco_node* root, int input_size, int output_size, int width, int depth, int sub_matrix_dimension1, int activations){
    aco_node** n = get_nodes_to_attach(input_size,output_size,width,depth, sub_matrix_dimension1, activations);
    attach_params_to_root(root,n, width);
    return n;
}
