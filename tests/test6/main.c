#include <llab.h>
#include <math.h>

int main(){
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
     * mini batch = 10
     * radam algorithm with default b1 and b2
     * learning rate = 0.0003
     * l2 regularization with lambda = 0.001
     * epochs = 4 (after 4 epochs reaches the best accuracy among all the previous models)
     * */
    srand(time(NULL));
    // Initializing Training resources
    int i,j,k,z,training_instances = 50000,input_dimension = 784,output_dimension = 10, middle_neurons = 100;
    int n_layers = 7;
    int batch_size = 10,threads = 8;
    int epochs = 5;
    unsigned long long int t = 1;
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "../data/train.bin";
    int size = 0;
    char temp[2];
    float b1 = BETA1_ADAM;
    float b2 = BETA2_ADAM;
    temp[1] = '\0';
    float** errors = (float**)malloc(sizeof(float*)*batch_size);
    
    for(i = 0; i < batch_size; i++){
        errors[i] = (float*)calloc(output_dimension,sizeof(float));
    }
    // Model Architecture
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
    fcls[0] = fully_connected(rls[0]->channels*rls[0]->input_rows*rls[0]->input_cols,middle_neurons,5,NO_DROPOUT,SIGMOID,0);
    fcls[1] = fully_connected(middle_neurons,output_dimension,6,NO_DROPOUT,SOFTMAX,0);
    model* m = network(n_layers,2,1,2,rls,cls,fcls);
    model** batch_m = (model**)malloc(sizeof(model*)*batch_size);
    float** ret_err = (float**)malloc(sizeof(float*)*batch_size);
    for(i = 0; i < batch_size; i++){
        batch_m[i] = copy_model(m);
    }
    int ws = count_weights(m);
    float lr = 0.0003, momentum = 0.9, lambda = 0.0001;
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
    
    printf("Training phase!\n");
    save_model(m,0);
    // Training
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
            model_tensor_input_ff_multicore(batch_m,input_dimension,1,1,&inputs[i*batch_size],batch_size,threads);
            for(j = 0; j < batch_size; j++){
                derivative_cross_entropy_array(batch_m[j]->fcls[1]->post_activation,outputs[i*batch_size+j],errors[j],output_dimension);
            }
            model_tensor_input_bp_multicore(batch_m,input_dimension,1,1,&inputs[i*batch_size],batch_size,threads,errors,output_dimension,ret_err);
            // sum the partial derivatives in m obtained from backpropagation
            for(j = 0; j < batch_size; j++){
                sum_model_partial_derivatives(batch_m[j],m,m);
            }
            // update m, reset m and copy the new weights in each instance of m of the batch
            update_model(m,lr,momentum,batch_size,RADAM,&b1,&b2,L2_REGULARIZATION,ws,lambda,&t);
            reset_model(m);
            for(j = 0; j < batch_size; j++){
                paste_model(m,batch_m[j]);
                reset_model(batch_m[j]);
            }
            
        }
        // Saving the model
        save_model(m,k+1);
    }
    
    // Deallocating Training resources
    free(ksource[0]);
    free(ksource);
    free_model(m);
    for(i = 0; i < batch_size; i++){
        free_model(batch_m[i]);
        free(errors[i]);
    }
    free(errors);
    free(batch_m);
    free(ret_err);
    for(i = 0; i < training_instances; i++){
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    
    // Initializing Testing resources
    model* test_m;
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
    
    
    long long unsigned int** cm;
    
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
        for(i = 0; i < testing_instances; i++){
            // Feed forward
            
            model_tensor_input_ff(test_m,input_dimension,1,1,inputs_test[i]);
            for(j = 0; j < output_dimension; j++){
                error+=cross_entropy(test_m->fcls[1]->post_activation[j],outputs_test[i][j]);
            }
              
            if(!i)
                cm = confusion_matrix(test_m->fcls[1]->post_activation, outputs_test[i],NULL, 10,0.5);
            else
                cm = confusion_matrix(test_m->fcls[1]->post_activation, outputs_test[i],cm, 10,0.5);
            reset_model(test_m);
        }
        printf("Error: %lf\n",error);
        printf("Accuracy, Precision, Sensitivity, Specificity:\n");
        print_accuracy(cm,output_dimension);
        print_precision(cm,output_dimension);
        print_sensitivity(cm,output_dimension);
        print_specificity(cm,output_dimension);
        for(i = 0; i < output_dimension*2; i++){
            free(cm[i]);
        }
        free(cm);
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
}
