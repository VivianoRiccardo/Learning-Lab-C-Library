#include <llab.h>

int main(){
    // 1 CONVOLUTIONAL LAYER:
    /* input 1° cl = 1*32*32, activation = RELU, PADDING = 1, MAX POOLING
     */
    // 1 RESIDUAL LAYER:
    /* input 1° cl = 20*14*14 convoltion, no normalization, padding
     * input 2° cl = 40*14*14 convolution no normalization, padding
     * */
    // 2 RESIDUAL LAYER:
    /* input 1° cl = 20*14*14 convoltion, no normalization, padding
     * input 2° cl = 40*14*14 convolution no normalization, padding
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
    int batch_size = 8,threads = batch_size;
    int epochs = 5;
    unsigned long long int t = 1;
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "../data/train.bin";
    int size = 0;
    char temp[2];
    float b1 = BETA1_ADAM;
    float b2 = BETA2_ADAM;
    temp[1] = '\0';

    
    // Model Architecture
    
    cl** cls = (cl**)malloc(sizeof(cl*));
    cl** cls2 = (cl**)malloc(sizeof(cl*)*2);
    cl** cls3 = (cl**)malloc(sizeof(cl*)*2);
    rl** rls = (rl**)malloc(sizeof(rl*)*2);
    cls[0] = convolutional(1,28,28,3,3,1000,1,1,1,1,2,2,0,0,2,2,LOCAL_RESPONSE_NORMALIZATION,LEAKY_RELU,MAX_POOLING,0,CONVOLUTION,EDGE_POPUP,EDGE_POPUP,0);
    cls2[0] = convolutional(1000,14,14,3,3,1500,1,1,1,1,2,2,0,0,2,2,LOCAL_RESPONSE_NORMALIZATION,LEAKY_RELU,NO_POOLING,10,CONVOLUTION,EDGE_POPUP,EDGE_POPUP,1);
    cls3[0] = convolutional(1000,14,14,3,3,1500,1,1,1,1,2,2,0,0,2,2,LOCAL_RESPONSE_NORMALIZATION,LEAKY_RELU,NO_POOLING,10,CONVOLUTION,EDGE_POPUP,EDGE_POPUP,3);
    cls2[1] = convolutional(1500,14,14,3,3,1000,1,1,1,1,2,2,0,0,2,2,LOCAL_RESPONSE_NORMALIZATION,LEAKY_RELU,NO_POOLING,5,CONVOLUTION,EDGE_POPUP,EDGE_POPUP,2);
    cls3[1] = convolutional(1500,14,14,3,3,1000,1,1,1,1,2,2,0,0,2,2,LOCAL_RESPONSE_NORMALIZATION,LEAKY_RELU,NO_POOLING,5,CONVOLUTION,EDGE_POPUP,EDGE_POPUP,4);
    rls[0] = residual(cls[0]->n_kernels,cls[0]->rows2,cls[0]->cols2,2,cls2);
    rls[1] = residual(cls[0]->n_kernels,cls[0]->rows2,cls[0]->cols2,2,cls3);
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*2);
    fcls[0] = fully_connected(rls[0]->channels*rls[0]->input_rows*rls[0]->input_cols,middle_neurons,5,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION,FREEZE_TRAINING,FULLY_FEED_FORWARD, STANDARD);
    fcls[1] = fully_connected(middle_neurons,output_dimension,6,NO_DROPOUT,SOFTMAX,0,0,NO_NORMALIZATION,FREEZE_TRAINING,FULLY_FEED_FORWARD, STANDARD);
    model* m = network(n_layers,2,1,2,rls,cls,fcls);
    set_model_training_edge_popup(m,0.5);
    m->fcls[1]->k_percentage = 0.8;
    
    set_model_error(m,FOCAL_LOSS,0,0,2,NULL,10);
    reset_model(m);
    
    
    /*
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*5);
    fcls[0] = fully_connected(input_dimension,200,0,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION,EDGE_POPUP,EDGE_POPUP);
    for(i = 1; i <4; i++){
        fcls[i] = fully_connected(200,200,i,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION,EDGE_POPUP,EDGE_POPUP);
    }
    fcls[i] = fully_connected(200,output_dimension,i,NO_DROPOUT,SOFTMAX,0,0,NO_NORMALIZATION,EDGE_POPUP,EDGE_POPUP);
    model* m = network(5,0,0,5,NULL,NULL,fcls);
    set_model_training_edge_popup(m,0.5);
    m->fcls[m->n_fcl-1]->k_percentage = 0.8;
    set_model_error(m,FOCAL_LOSS,0,0,2,NULL,10);
    reset_model(m);
    */
    //reinitialize_weights_according_to_scores_model(m,1,99999);
    model** batch_m = (model**)malloc(sizeof(model*)*batch_size);
    for(i = 0; i < batch_size; i++){
        batch_m[i] = copy_model_without_learning_parameters(m);
    }
    int ws = count_weights(m);
    float lr = 0.01, momentum = 0.9, lambda = 0.0001;
    // Reading the data in a char** vector
    read_file_in_char_vector(ksource,filename,&size);
    float** inputs = (float**)malloc(sizeof(float*)*training_instances);
    float** outputs = (float**)malloc(sizeof(float*)*training_instances);
    
    float* computed_error = (float*)calloc(sizeof(float),output_dimension);
    
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
    double epoch_error = 0;
    for(k = 0; k < epochs; k++){
        printf("Starting epoch %d/%d\n",k+1,epochs);
        // Shuffling before each epoch
        shuffle_float_matrices(inputs,outputs,training_instances);
        for(i = 0; i < training_instances/batch_size; i++){
            //if(!(i%(8*40)))
            printf("Mini batch: %d/%d\n",i+1,training_instances/batch_size);
            // Feed forward and backpropagation
            ff_error_bp_model_multicore_opt(batch_m,m,1,28,28,&inputs[i*batch_size],batch_size,batch_size,&outputs[i*batch_size],NULL);
            
            sum_models_partial_derivatives(m,batch_m,batch_size);
            // update m, reset m and copy the new weights in each instance of m of the batch
            update_model(m,lr,momentum,batch_size,NESTEROV,&b1,&b2,NO_REGULARIZATION,ws,lambda,&t);
            reset_model(m);
            for(j = 0; j < batch_size; j++){
                focal_loss_array(batch_m[j]->fcls[m->n_fcl-1]->post_activation,outputs[i*batch_size+j],computed_error,2,output_dimension);
                epoch_error += sum_over_input(computed_error,output_dimension);
                free(computed_error);
                computed_error = (float*)calloc(sizeof(float),output_dimension);
                reset_model_without_learning_parameters(batch_m[j]);
            }
            
            update_training_parameters(&b1,&b2,&t,m->beta1_adam,m->beta2_adam);
        }
        printf("epoch error: %lf\n",epoch_error);
        epoch_error = 0;
        // Saving the model
        save_model(m,k+1);
    }
    
    // Deallocating Training resources
    free(computed_error);
    free(ksource[0]);
    free(ksource);
    free_model(m);
    for(i = 0; i < batch_size; i++){
        free_model_without_learning_parameters(batch_m[i]);
    }
    free(batch_m);
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
    int cose = 0;
    printf("Testing phase!\n");
    double error = 0;
    // Testing
    for(k = 1; k < epochs+1; k++){
        int cose = 0;
        printf("Model N. %d/%d\n",k+1,epochs);
        // Loading the model
        char temp3[5];
        temp3[0] = '.';
        temp3[1] = 'b';
        temp3[2] = 'i';
        temp3[3] = 'n';
        temp3[4] = '\0';
        itoa_n(k,temp2);
        strcat(temp2,temp3);
        test_m = load_model(temp2);
        set_model_training_edge_popup(test_m,0.5);
        test_m->fcls[test_m->n_fcl-1]->k_percentage = 1;
        test_m->fcls[test_m->n_fcl-2]->k_percentage = 1;
        reset_model(test_m);
        for(i = 0; i < testing_instances; i++){
            // Feed forward
            
            model_tensor_input_ff(test_m,input_dimension,1,1,inputs_test[i]);
            
            for(j = 0; j < output_dimension; j++){
                error+=focal_loss(test_m->fcls[test_m->n_fcl-1]->post_activation[j],outputs_test[i][j],2);
                if(outputs_test[i][j] == 1)
                    if(test_m->fcls[test_m->n_fcl-1]->post_activation[j]>= 0.5)
                        cose++;
            }
            
              
            if(!i)
                cm = confusion_matrix(test_m->fcls[test_m->n_fcl-1]->post_activation, outputs_test[i],NULL, 10,0.5);
            else
                cm = confusion_matrix(test_m->fcls[test_m->n_fcl-1]->post_activation, outputs_test[i],cm, 10,0.5);
            reset_model_for_edge_popup(test_m);
        }
        printf("Error: %lf\n",error);
        printf("Accuracy, Precision, Sensitivity, Specificity:\n");
        printf("%d\n",cose);
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
