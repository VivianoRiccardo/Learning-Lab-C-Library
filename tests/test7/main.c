#include <llab.h>

int main(){
    // Variational autoencoder 
    // encoder:
    // 1 CONVOLUTIONAL LAYER:
    /* input 1° cl = 1*32*32, activation = RELU, PADDING = 1, MAX POOLING
     */
    // 1 RESIDUAL LAYER:
    /* input 1° cl = 20*14*14 convoltion, group norm channel division = 10, padding
     * input 2° cl = 40*14*14 convolution group norm channel division = 5, padding
     * */
    // 2 RESIDUAL LAYER:
    /* input 1° cl = 20*14*14 convoltion, group norm channel division = 10, padding
     * input 2° cl = 40*14*14 convolution group norm channel division = 5, padding
     * */
    // 2 FULLY-CONNECTED LAYERS:
    /* input 1° fcl = 784, output = 100, activation = sigmoid,  no dropout
     * input 2° fcl = 100, output = 10, activation = softmax, no dropout
     **/
     // decoder:
     /* encoder reversed
      * 
      * 
     * mini batch = 10
     * radam algorithm with default b1 and b2
     * learning rate = 0.0003
     * no regularization
     * epochs = 4 (after 4 epochs reaches the best accuracy among all the previous models)
     * */
    srand(time(NULL));
    // Initializing Training resources
    int i,j,k,z,training_instances = 50000,input_dimension = 784,output_dimension = 10, middle_neurons = 100;
    int n_layers = 7;
    int batch_size = 5,threads = 4, mini_batch_size = 5;
    int epochs = 5;
    unsigned long long int t = 1;
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "../data/train.bin";
    int size = 0;
    char temp[2];
    float b1 = BETA1_ADAM;
    float b2 = BETA2_ADAM;
    
    temp[1] = '\0';
    // Encoder Architecture
    cl** cls = (cl**)malloc(sizeof(cl*));
    cl** cls2 = (cl**)malloc(sizeof(cl*)*2);
    cl** cls3 = (cl**)malloc(sizeof(cl*)*2);
    rl** rls = (rl**)malloc(sizeof(rl*)*2);
    cls[0] = convolutional(1,28,28,3,3,20,1,1,1,1,2,2,0,0,2,2,NO_NORMALIZATION,RELU,MAX_POOLING,0,CONVOLUTION,0);
    cls2[0] = convolutional(20,14,14,3,3,40,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,10,CONVOLUTION,1);
    cls3[0] = convolutional(20,14,14,3,3,40,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,10,CONVOLUTION,3);
    cls2[1] = convolutional(40,14,14,3,3,20,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,5,CONVOLUTION,2);
    cls3[1] = convolutional(40,14,14,3,3,20,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,5,CONVOLUTION,4);
    rls[0] = residual(cls[0]->n_kernels,cls[0]->rows2,cls[0]->cols2,2,cls2);
    rls[1] = residual(cls[0]->n_kernels,cls[0]->rows2,cls[0]->cols2,2,cls3);
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*2);
    fcls[0] = fully_connected(rls[0]->channels*rls[0]->input_rows*rls[0]->input_cols,middle_neurons,5,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION);
    fcls[1] = fully_connected(middle_neurons,2*output_dimension,6,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION);
    model* encoder = network(n_layers,2,1,2,rls,cls,fcls);// encoder
    
    // Decoder Architecture
    cl** dcls2 = (cl**)malloc(sizeof(cl*)*2);
    cl** dcls3 = (cl**)malloc(sizeof(cl*)*2);
    rl** drls = (rl**)malloc(sizeof(rl*)*2);
    dcls2[0] = convolutional(20,56,56,3,3,40,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,10,CONVOLUTION,2);
    dcls3[0] = convolutional(20,56,56,3,3,40,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,10,CONVOLUTION,4);
    dcls2[1] = convolutional(40,56,56,3,3,20,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,5,CONVOLUTION,3);
    dcls3[1] = convolutional(40,56,56,3,3,20,1,1,1,1,2,2,0,0,2,2,GROUP_NORMALIZATION,RELU,NO_POOLING,5,CONVOLUTION,5);
    drls[0] = residual(20,56,56,2,dcls2);
    drls[1] = residual(20,56,56,2,dcls3);
    fcl** dfcls = (fcl**)malloc(sizeof(fcl*)*3);
    dfcls[0] = fully_connected(output_dimension,middle_neurons,0,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION);
    dfcls[1] = fully_connected(middle_neurons,drls[0]->channels*drls[0]->input_rows*drls[0]->input_cols,1,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION);
    dfcls[2] = fully_connected(56*20*56,1*28*28,6,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION);
    
    model* decoder = network(n_layers,2,0,3,drls,NULL,dfcls);// decoder
    
    vaemodel* vae = variational_auto_encoder_model(encoder,decoder,output_dimension);
    
    vaemodel** batch_m = (vaemodel**)malloc(sizeof(vaemodel*)*batch_size);
    float** ret_err = (float**)malloc(sizeof(float*)*batch_size);
    for(i = 0; i < batch_size; i++){
        batch_m[i] = copy_vae_model(vae);
    }
    int ws = count_weights_vae_model(vae);
    float lr = 0.0003, momentum = 0.6, lambda = 0.0001;
    // Reading the data in a char** vector
    read_file_in_char_vector(ksource,filename,&size);
    float** inputs = (float**)malloc(sizeof(float*)*training_instances);
    float** outputs = (float**)malloc(sizeof(float*)*training_instances);
    // Putting the data in float** vectors
    for(i = 0; i < training_instances; i++){
        inputs[i] = (float*)malloc(sizeof(float)*input_dimension);
        outputs[i] = (float*)calloc(output_dimension,sizeof(float));
        for(j = 0; j < input_dimension+1; j++){
            temp[0] = ksource[0][i*(input_dimension+1)+j];
            if(j == input_dimension)
                outputs[i][atoi(temp)] = 1;
            else
                inputs[i][j] = atof(temp);
        }
    }
    
    float** errors = (float**)malloc(sizeof(float*)*mini_batch_size);
    
    for(i = 0; i < mini_batch_size; i++){
        errors[i] = (float*)malloc(sizeof(float)*input_dimension);
    }
    
    printf("Training phase!\n");
    // Training
    save_model(vae->decoder,0);
    for(k = 0; k < epochs; k++){
        if(k == 10)
            lr = 0.0001;
        else if(k == 15)
            lr = 0.00005;
        printf("Starting epoch %d/%d\n",k+1,epochs);
        // Shuffling before each epoch
        shuffle_float_matrices(inputs,outputs,training_instances);
        for(i = 0; i < training_instances/batch_size; i++){
            //printf("Mini batch number: %d\n",i+1);
            // Feed forward and backpropagation
            
            vae_model_tensor_input_ff_multicore(batch_m,input_dimension,1,1,&inputs[i*batch_size],batch_size,threads);
            
            
            for(j = 0; j < mini_batch_size; j++){
                derivative_mse_array(batch_m[j]->decoder->fcls[2]->post_activation,inputs[i*batch_size],errors[j],input_dimension);
            }
            
            vae_model_tensor_input_bp_multicore(batch_m,input_dimension,1,1,&inputs[i*batch_size],batch_size,threads,errors,input_dimension,ret_err);
            // sum the partial derivatives in m obtained from backpropagation
            for(j = 0; j < batch_size; j++){
                sum_vae_model_partial_derivatives(batch_m[j],vae,vae);
            }
            
            clipping_gradient_vae_model(vae,2);
            
            // update m, reset m and copy the new weights in each instance of m of the batch
            update_vae_model(vae,lr,momentum,batch_size,NESTEROV,&b1,&b2,L2_REGULARIZATION,ws,lambda,&t);
            reset_vae_model(vae);
            for(j = 0; j < batch_size; j++){
                paste_vae_model(vae,batch_m[j]);
                reset_vae_model(batch_m[j]);
            }
            
        }
        // Saving the model
        save_model(vae->decoder,k+1);
    }
    
    // Deallocating Training resources
    free(ksource[0]);
    free(ksource);
    free_vae_model(vae);
    free_model(encoder);
    free_model(decoder);
    for(i = 0; i < batch_size; i++){
        free_vae_model(batch_m[i]);
    }
    free(batch_m);
    free(ret_err);
    for(i = 0; i < training_instances; i++){
        free(inputs[i]);
        free(outputs[i]);
        free(errors[j]);
    }
    free(errors);
    free(inputs);
    free(outputs);
    
    // Initializing Testing resources
    model* test_m;
    float* normal_vector = (float*)malloc(sizeof(float)*output_dimension);
    char** ksource2 = (char**)malloc(sizeof(char*));
    char* filename2 = "../data/test.bin";
    int size2 = 0;
    int testing_instances = 10000;
    char temp2[256];
    read_file_in_char_vector(ksource2,filename2,&size);
    float** inputs_test = (float**)malloc(sizeof(float*)*testing_instances);
    float** outputs_test = (float**)malloc(sizeof(float*)*testing_instances);
    // Putting the data in float** vectors
    for(i = 0; i < testing_instances; i++){
        inputs_test[i] = (float*)malloc(sizeof(float)*input_dimension);
        outputs_test[i] = (float*)calloc(output_dimension,sizeof(float));
        for(j = 0; j < input_dimension+1; j++){
            temp[0] = ksource2[0][i*(input_dimension+1)+j];
            if(j == input_dimension)
                outputs_test[i][atoi(temp)] = 1;
            else
                inputs_test[i][j] = atof(temp);
        }
    }
    
    
    printf("Testing phase!\n");
    double error = 0;
    // Testing
    for(k = 0; k < epochs+1; k++){
        printf("Model N. %d/%d\n",k+1,epochs);
        // Loading the model
        char temp3[5];
        temp3[0] = '.';
        temp3[1] = 'b';
        temp3[2] = 'i';
        temp3[3] = 'n';
        temp3[4] = '\0';
        itoa(k,temp2);
        strcat(temp2,temp3);
        test_m = load_model(temp2);
        //for(i = 0; i < testing_instances; i++){
        for(i = 0; i < 3; i++){
            // Feed forward
            for(j = 0; j < output_dimension; j++){
                normal_vector[j] = random_normal();
            }
            model_tensor_input_ff(test_m,input_dimension,1,1,normal_vector);
            for(j = 0; j < 28; j++){
                for(z = 0; z < 28; z++){
                    if(test_m->fcls[2]->post_activation[j*28+z] >= 0.5)
                        printf("1");
                    else
                        printf("0");
                }
                printf("\n");
            }
            printf("\n");
            reset_model(test_m);  
        }
        printf("Error: %lf\n",error);
        error = 0;
        free_model(test_m);
    }
    // Deallocating testing resources
    free(ksource2[0]);
    free(ksource2);
    for(i = 0; i < testing_instances; i++){
        free(inputs_test[i]);
        free(outputs_test[i]);
    }
    free(inputs_test);
    free(outputs_test);
    free(normal_vector);
}
