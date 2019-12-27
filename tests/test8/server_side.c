#include <llab.h>
#include <unistd.h>
#include <signal.h>

#define PORT 9000
#define THREAD_PER_CLIENT 2
#define INPUTS_PER_CLIENT 28*28*1*THREAD_PER_CLIENT
#define OUTPUT_PER_CLIENT 10*THREAD_PER_CLIENT

volatile sig_atomic_t child_pid = 0;

void term(int signum){
    kill(child_pid,SIGKILL);
}

int main(){
    
    char* file = "0.bin";
    model* m = load_model(file);
    
    int i,ret,pid;
    
    int number_connections = 5;
    
    int** writers = (int**)malloc(sizeof(int*)*number_connections);//writers pipes
    int** readers = (int**)malloc(sizeof(int*)*number_connections);// readers pipes
    
    // create pipes
    for(i = 0; i < number_connections; i++){
        writers[i] = (int*)malloc(sizeof(int)*2);
        readers[i] = (int*)malloc(sizeof(int)*2);
        ret = pipe(writers[i]);
        
        if(ret == -1){
            fprintf(stderr,"Error: not able to create the pipe\n");
            exit(1);
        }
        ret = pipe(readers[i]);
        
        if(ret == -1){
            fprintf(stderr,"Error: not able to create the pipe\n");
            exit(1);
        }
    }
    
    pid = fork();// split father-son process
    
    if(pid == -1){
        fprintf(stderr,"Error: not able to create a son process\n");
        exit(1);
    }
    
    else if(pid == 0){
        // Son
        int* w = (int*)malloc(sizeof(int)*number_connections);
        int* r = (int*)malloc(sizeof(int)*number_connections);
        for(i = 0; i < number_connections; i++){
            close(writers[i][0]);
            w[i] = writers[i][1];
            close(readers[i][0]);
            r[i] = readers[i][1];
        }
        
        int buffer_size = get_array_size_params_model(m);
        free_model(m);
        
        run_server(PORT,number_connections,r,w,buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT, "127.0.0.1");
        
        for(i = 0; i < number_connections; i++){
            close(writers[i][1]);
            close(readers[i][1]);
        }
        
        free(writers);
        free(readers);
        free(w);
        free(r);
        return 0;
    }
    
    else{
        child_pid = pid;
        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = term;
        sigaction(SIGTERM, &action, NULL);
        
        // Father
        int* w = (int*)malloc(sizeof(int)*number_connections);
        int* r = (int*)malloc(sizeof(int)*number_connections);
        for(i = 0; i < number_connections; i++){
            close(writers[i][1]);
            w[i] = readers[i][0];
            close(readers[i][1]);
            r[i] = writers[i][0];
        }
        
        srand(time(NULL));
        // Initializing Training resources
        int j,k,z,training_instances = 50000,input_dimension = 784,output_dimension = 10, middle_neurons = 100;
        int n_layers = 7;
        int batch_size = number_connections*THREAD_PER_CLIENT,threads = 4;
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
        float** buff = (float**)malloc(sizeof(float*)*number_connections);
        for(i = 0; i < number_connections; i++){
            buff[i] = (float*)calloc((get_array_size_params_model(m)+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT),sizeof(float));
        }
        
        int buffer_size = get_array_size_params_model(m);
        
        for(i = 0; i < batch_size; i++){
            errors[i] = (float*)calloc(output_dimension,sizeof(float));
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
        model* sum_m = copy_model(m);
        
        // Training
        for(k = 0; k < epochs; k++){
            if(k == 10)
                lr = 0.0001;
            else if(k == 15)
                lr = 0.00005;
            printf("Starting epoch %d/%d\n",k+1,epochs);
            // Shuffling before each epoch
            shuffle_float_matrices(inputs,outputs,training_instances);
            for(j = 0; j < number_connections; j++){
                memcopy_params_to_vector_model(m,buff[j]);
            }
            for(i = 0; i < training_instances/batch_size; i++){
                printf("Mini batch: %d\n", i+1);
                int ret;
                for(j = 0, z = 0; j < number_connections; j++, z+=2){
                    memcpy(&buff[j][buffer_size],inputs[i*batch_size+z],(INPUTS_PER_CLIENT/THREAD_PER_CLIENT)*sizeof(float));
                    memcpy(&buff[j][buffer_size+(INPUTS_PER_CLIENT/THREAD_PER_CLIENT)],outputs[i*batch_size+z],(OUTPUT_PER_CLIENT/THREAD_PER_CLIENT)*sizeof(float));
                    memcpy(&buff[j][buffer_size+(OUTPUT_PER_CLIENT/THREAD_PER_CLIENT)+(INPUTS_PER_CLIENT/THREAD_PER_CLIENT)],inputs[i*batch_size+z+1],(INPUTS_PER_CLIENT/THREAD_PER_CLIENT)*sizeof(float));
                    memcpy(&buff[j][buffer_size+INPUTS_PER_CLIENT+(OUTPUT_PER_CLIENT/THREAD_PER_CLIENT)],outputs[i*batch_size+z+1],(OUTPUT_PER_CLIENT/THREAD_PER_CLIENT)*sizeof(float));
                    ret = write(w[j],buff[j],sizeof(float)*(buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT));// writing to sons
                }
                for(j = 0, z = 0; j < number_connections; j++, z+=2){
                    while(read(r[j], buff[j], sizeof(float)*(buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT)) == 0);// waiting for sons
                    memcopy_vector_to_derivative_params_model(m,buff[j]);
                    sum_model_partial_derivatives(m,sum_m,sum_m);
                }
                
                // update m, reset m and copy the new weights in each instance of m of the batch
                update_model(sum_m,lr,momentum,batch_size,RADAM,&b1,&b2,L2_REGULARIZATION,ws,lambda,&t);
                reset_model(sum_m);
                reset_model(m);
                for(j = 0; j < number_connections; j++){
                    memcopy_params_to_vector_model(sum_m,buff[j]);
                }
            }
            // Saving the model
            save_model(sum_m,k+1);
        }
        
        // Deallocating Training resources
        free(ksource[0]);
        free(ksource);
        free_model(m);
        free_model(sum_m);
        for(i = 0; i < batch_size; i++){
            free(errors[i]);
        }
        free(errors);
        for(i = 0; i < training_instances; i++){
            free(inputs[i]);
            free(outputs[i]);
        }
        free(inputs);
        free(outputs);
        
        for(i = 0; i < number_connections; i++){
            close(writers[i][1]);
            close(readers[i][1]);
            free(buff[i]);
        }
        
        free(writers);
        free(readers);
        free(w);
        free(r);
        free(buff);
        kill(pid,SIGKILL);
        
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
            for(i = 0; i < 1; i++){
                // Feed forward
                model_tensor_input_ff(test_m,input_dimension,1,1,inputs_test[i]);
                for(j = 0; j < output_dimension; j++){
                    error+=cross_entropy(test_m->fcls[1]->post_activation[j],outputs_test[i][j]);
                }
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
        
        return 0;
    }
}
