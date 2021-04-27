#include "../../src/llab.h"

int main(){
    //lstms
    srand(time(NULL));
    // Initializing Training resources
    int i,j,k,z,training_instances = 50000,input_dimension = 784,output_dimension = 10, window = 28, rmodel_size = 28;
    int n_layers = 3;
    int batch_size = 10,threads = batch_size;
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
    lstm** l = (lstm**)malloc(sizeof(lstm*)*n_layers);
    
    for(i = 0; i < n_layers-1; i++){
        l[i] = recurrent_lstm(rmodel_size,NO_DROPOUT,0.1,NO_DROPOUT,0.1,i,window,LSTM_RESIDUAL,GROUP_NORMALIZATION,4,GRADIENT_DESCENT,FULLY_FEED_FORWARD);
    }

    l[i] = recurrent_lstm(rmodel_size,NO_DROPOUT,0.1,NO_DROPOUT,0.1,i,window,LSTM_RESIDUAL,NO_NORMALIZATION,0,GRADIENT_DESCENT,FULLY_FEED_FORWARD);
    rmodel* r = recurrent_network(n_layers,n_layers,l,28,STATEFUL);
    
    rmodel** batch_r = (rmodel**)malloc(sizeof(rmodel*)*batch_size);
    
    
    fcl** fcls = (fcl**)malloc(sizeof(fcl*));
    fcls[0] = fully_connected(rmodel_size,output_dimension,0,NO_DROPOUT,SOFTMAX,0,0,NO_NORMALIZATION,GRADIENT_DESCENT,FULLY_FEED_FORWARD);
    model* m = network(1,0,0,1,NULL,NULL,fcls);
    set_model_error(m,FOCAL_LOSS,0,0,2,NULL,output_dimension);
    
    model** batch_m = (model**)malloc(sizeof(model*)*batch_size);
    for(i = 0; i < batch_size; i++){
        batch_m[i] = copy_model(m);
        batch_r[i] = copy_rmodel(r);
    }
    
    int ws = count_weights(m);
    float lr = 0.001, momentum = 0.9, lambda = 0.0001;
    
    
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
    
    float*** inputs_rmodel = (float***)malloc(sizeof(float**)*training_instances);
    float** outputs_rmodel = (float**)malloc(sizeof(float*)*batch_size);
    float*** h = (float***)malloc(sizeof(float**)*batch_size);
    float*** c = (float***)malloc(sizeof(float**)*batch_size);
    for(i = 0; i < batch_size; i++){
        outputs_rmodel[i] = (float*)calloc(rmodel_size,sizeof(float));
        h[i] = (float**)malloc(sizeof(float*)*n_layers); 
        c[i] = (float**)malloc(sizeof(float*)*n_layers);
        for(j = 0; j < n_layers; j++){
            h[i][j] = (float*)calloc(rmodel_size,sizeof(float));
            c[i][j] = (float*)calloc(rmodel_size,sizeof(float));
        }
    }
    
    for(i = 0; i < training_instances; i++){
        inputs_rmodel[i] = (float**)malloc(sizeof(float*)*window);
        for(j = 0; j < window; j++){
            inputs_rmodel[i][j] = (float*)calloc(rmodel_size,sizeof(float));
            for(z = 0; z < rmodel_size; z++){
                inputs_rmodel[i][j][z] = inputs[i][j*rmodel_size+z];
            }
        }
    }
    
    float** ret_model_error = (float**)malloc(sizeof(float*)*batch_size);
    float*** error_lstms = (float***)malloc(sizeof(float**)*batch_size);
    for(i = 0; i < batch_size; i++){
        error_lstms[i] = (float**)malloc(sizeof(float*)*window);
        for(j = 0; j < window; j++){
            error_lstms[i][j] = (float*)calloc(rmodel_size,sizeof(float));
        }
    } 
    
    float**** dfioc = (float****)malloc(sizeof(float***)*batch_size);
    float* computed_error = (float*)calloc(sizeof(float),output_dimension);
    double epoch_error = 0;
    printf("Training phase!\n");
    
    save_model(m,0);
    save_rmodel(r,1);
    // Training
    
    for(k = 0; k < epochs; k++){
        printf("Starting epoch %d/%d\n",k+1,epochs);
        // Shuffling before each epoch
        shuffle_float_matrix_float_tensor(outputs,inputs_rmodel,training_instances);
        for(i = 0; i < training_instances/batch_size; i++){
            //printf("batch: %d/%d\n",i,training_instances/batch_size);
            // Feed forward and backpropagation
            ff_rmodel_lstm_multicore(h,c,&inputs_rmodel[i*batch_size],batch_r,batch_size,threads);
            for(j = 0; j < batch_size; j++){
                copy_array(batch_r[j]->lstms[n_layers-1]->out_up[window-1],outputs_rmodel[j],rmodel_size);
            }
            ff_error_bp_model_multicore(batch_m,rmodel_size,1,1,outputs_rmodel,batch_size,threads,&outputs[i*batch_size],ret_model_error);
            for(j = 0; j < batch_size; j++){
                copy_array(ret_model_error[j],error_lstms[j][window-1],rmodel_size);
            }
            bp_rmodel_lstm_multicore(h,c,&inputs_rmodel[i*batch_size],batch_r,error_lstms,batch_size,threads,dfioc, NULL);
            for(j = 0; j < batch_size; j++){
                free_tensor(dfioc[j],n_layers,4);
            }

            sum_rmodels_partial_derivatives(r,batch_r,batch_size);
            sum_models_partial_derivatives(m,batch_m,batch_size);
            adaptive_gradient_clipping_model(m,0.01,1e-3);
            adaptive_gradient_clipping_rmodel(r,0.01,1e-3);
            update_rmodel(r,lr,momentum,batch_size,ADAM,&b1,&b2,NO_REGULARIZATION,ws,lambda,&t);
            update_model(m,lr,momentum,batch_size,ADAM,&b1,&b2,NO_REGULARIZATION,ws,lambda,&t);
            reset_rmodel(r);
            for(j = 0; j < batch_size; j++){
                focal_loss_array(batch_m[j]->fcls[m->n_fcl-1]->post_activation,outputs[i*batch_size+j],computed_error,2,output_dimension);
                epoch_error += sum_over_input(computed_error,output_dimension);
                free(computed_error);
                computed_error = (float*)calloc(sizeof(float),output_dimension);
                paste_model(m,batch_m[j]);
                reset_model(batch_m[j]);
                paste_rmodel(r,batch_r[j]);
                reset_rmodel(batch_r[j]);
                
            }
            
            update_training_parameters(&b1,&b2,&t,m->beta1_adam,m->beta2_adam);
            
        }
        printf("epoch error: %lf\n",epoch_error);
        epoch_error = 0;
        // Saving the model
        save_model(m,2*(k+1));
        save_rmodel(r,2*(k+1) + 1);
    }
    
    // Deallocating Training resources
    free(ksource[0]);
    free(ksource);
    free_model(m);
    free_rmodel(r);
    for(i = 0; i < batch_size; i++){
        free_model(batch_m[i]);
        free_rmodel(batch_r[i]);
        free(errors[i]);
    }
    free(batch_r);
    free(errors);
    free(batch_m);
    for(i = 0; i < training_instances; i++){
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    free_tensor(h,batch_size,n_layers);
    free_tensor(c,batch_size,n_layers);
    free_tensor(inputs_rmodel,training_instances,window);
    free_tensor(error_lstms,batch_size,window);
    free_matrix(outputs_rmodel,batch_size);
    free(ret_model_error);
    free(dfioc);
    free(computed_error);
    
    // Initializing Testing resources
    model* test_m;
    rmodel* test_r;
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
    
    
    
    
    inputs_rmodel = (float***)malloc(sizeof(float**)*testing_instances);
    float* outputs_rmodel2 = (float*)calloc(rmodel_size, sizeof(float));
    float** h2 = (float**)malloc(sizeof(float*)*n_layers);
    float** c2 = (float**)malloc(sizeof(float*)*n_layers);
    for(i = 0; i < n_layers; i++){
        h2[i] = (float*)calloc(rmodel_size,sizeof(float)); 
        c2[i] = (float*)calloc(rmodel_size,sizeof(float));
    }
    
    for(i = 0; i < testing_instances; i++){
        inputs_rmodel[i] = (float**)malloc(sizeof(float*)*window);
        for(j = 0; j < window; j++){
            inputs_rmodel[i][j] = (float*)calloc(rmodel_size,sizeof(float));
            for(z = 0; z < rmodel_size; z++){
                inputs_rmodel[i][j][z] = inputs_test[i][j*rmodel_size+z];
            }
        }
    }
    
    
    
    
    long long unsigned int** cm;
    
    printf("Testing phase!\n");
    double error = 0;
    // Testing
    for(k = 0; k < (epochs+1)*2; k+=2){
        printf("Model N. %d/%d\n",k/2+1,epochs);
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
        itoa(k+1,temp2);
        strcat(temp2,temp3);
        test_r = load_rmodel(temp2);
        for(i = 0; i < testing_instances; i++){
            // Feed forward
            ff_rmodel(h2,c2,inputs_rmodel[i],test_r);
            copy_array(test_r->lstms[n_layers-1]->out_up[window-1],outputs_rmodel2,rmodel_size);
            
            model_tensor_input_ff(test_m,rmodel_size,1,1,outputs_rmodel2);
            for(j = 0; j < output_dimension; j++){
                error+=focal_loss(test_m->fcls[0]->post_activation[j],outputs_test[i][j],2);
            }
            if(!i)
                cm = confusion_matrix(test_m->fcls[1]->post_activation, outputs_test[i],NULL, 10,0.5);
            else
                cm = confusion_matrix(test_m->fcls[1]->post_activation, outputs_test[i],cm, 10,0.5);
            reset_model(test_m);
            reset_rmodel(test_r);
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
        free_rmodel(test_r);
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
    free_tensor(inputs_rmodel,testing_instances,window);
    free(outputs_rmodel2),
    free_matrix(h2,n_layers);
    free_matrix(c2,n_layers);
}

