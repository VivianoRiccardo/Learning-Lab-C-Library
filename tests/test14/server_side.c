#include <llab.h>
#include <unistd.h>
#include <signal.h>

#define PORT 9000
#define THREAD_PER_CLIENT 1
#define TRAINING_INSTANCES_PER_CLIENT 5000
#define INPUTS_PER_CLIENT 28*28*1*THREAD_PER_CLIENT*TRAINING_INSTANCES_PER_CLIENT

volatile sig_atomic_t child_pid = 0;

void term(int signum){
    kill(child_pid,SIGKILL);
}

int main(int argc, char** argv){
    
    if(argc < 2){
        fprintf(stderr,"Error: 2 params as input: ./exec number_of_clients/10\n");
        exit(1);
    }
    
    int numb = atoi(argv[1]);
    if(numb < 1){
        fprintf(stderr,"Error: second parameter must be >= 1\n");
        exit(1);
    }
    
    char* file = "0.bin";
    model* m = load_model(file);// loading a pre-saved model
    
    int i,ret,pid;
    
    int number_connections = 10*numb;// number of clients
    
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
            close(readers[i][1]);
            r[i] = readers[i][0];
        }
        
        int buffer_size = get_array_size_weights_model(m);
        free_model(m);
        
        run_server(PORT,number_connections,r,w,buffer_size*2+1, "127.0.0.1");// run the server with (scores + 1 )*sizeof(float) communication sockets with clients
        
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
        // ctrl+c just ends all the threads of the process
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
            w[i] = readers[i][1];
            close(readers[i][0]);
            r[i] = writers[i][0];
        }
        
        srand(time(NULL));
        // Initializing Training resources
        int j,k,z,training_instances = 50000,input_dimension = 784,output_dimension = 10;
        int iterations = 10;
        unsigned long long int t = 1;
        char** ksource = (char**)malloc(sizeof(char*));
        char* filename = "../../data/train.bin";
        int size = 0;
        char temp[2];
        float b1 = BETA1_ADAM;
        float b2 = BETA2_ADAM;
        temp[1] = '\0';
        float* buff = (float*)calloc((get_array_size_weights_model(m)*2+1),sizeof(float));// buffer used for communications
        
        int buffer_size = get_array_size_weights_model(m)*2;// get the buffer size number of weights
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
        set_model_training_edge_popup(m,0.5);
        m->fcls[m->n_fcl-1]->k_percentage = 1;
        model* sum_m = copy_model(m);// where we sum all the scores
        model* sum_m2 = copy_model(m);// where we sum all the scores
        set_model_training_edge_popup(sum_m,0.5);
        set_model_training_edge_popup(sum_m2,0.5);
        sum_m->fcls[sum_m->n_fcl-1]->k_percentage = 1;
        sum_m2->fcls[sum_m2->n_fcl-1]->k_percentage = 1;
        memcopy_weights_to_vector_model(m,buff);//filling the buffer
        memcopy_scores_to_vector_model(m,&buff[buffer_size/2]);//filling the buffer
        
        
        // Initializing Testing resources
        model* test_m;
        char** ksource2 = (char**)malloc(sizeof(char*));
        char* filename2 = "../../data/test.bin";
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
        
        // Training
        
        for(k = 0; k < iterations; k++){
            printf("Starting iteration %d/%d\n",k+1,iterations);
            if (!k){
                for(j = 0; j < numb; j++){
                    // Shuffling the instances
                    shuffle_float_matrices(inputs,outputs,training_instances);
                    
                    // creating the data for the clients storing it in files. these files are gonna be read by clients
                    for(i = 0; i < training_instances; i+=TRAINING_INSTANCES_PER_CLIENT){
                        char temp2[256];
                        char temp3[5];
                        temp3[0] = '.';
                        temp3[1] = 'b';
                        temp3[2] = 'i';
                        temp3[3] = 'n';
                        temp3[4] = '\0';
                        itoa(100+j*training_instances/TRAINING_INSTANCES_PER_CLIENT+i/TRAINING_INSTANCES_PER_CLIENT,temp2);
                        strcat(temp2,temp3);
                        printf("writing file %s\n",temp2);
                        FILE* fil = fopen(temp2,"w");
                        for(z = 0; z < TRAINING_INSTANCES_PER_CLIENT; z++){
                            fwrite(inputs[i+z],input_dimension*sizeof(float),1,fil);
                            fwrite(outputs[i+z],output_dimension*sizeof(float),1,fil);
                        }
                        fclose(fil);
                        
                    }
                }
            }
            int ret = -1;
            for(j = 0 ; j < number_connections; j++){
                ret = 0;
                buff[buffer_size] = (float)(j);// each buffer gets a final flag to tell the client: ehy you are the client number j, read from this file....
                ret = write(w[j],buff,sizeof(float)*(buffer_size+1));// writing to sons
                printf("%d Bytes written to client %d\n",ret,j);
            }
            ret = 0;
            for(j = 0; j < number_connections; j++){
                ret = 0;
                while(ret == 0){
                    ret = read(r[j], buff, sizeof(float)*(buffer_size+1));
                    if(ret == -1)
                        printf("Error description is : %s\n",strerror(errno));
                }// waiting for sons
                printf("%d Bytes read from client %d\n",ret,j);
                memcopy_vector_to_scores_model(m,buff);// copying the scores in m
                if(j == 0){
                    reset_score_model(sum_m);
                    sum_score_model(m,sum_m,sum_m);// summing the scores in sum_m
                }
                else
                    compare_score_model(m,sum_m,sum_m);
                sum_score_model(m,sum_m2,sum_m2);// compare the scores in sum_m
                set_model_training_edge_popup(sum_m,0.5);
                sum_m->fcls[sum_m->n_fcl-1]->k_percentage = 1;    
                reset_model(sum_m);
                int zi,kk;
                // test time
                double error = 0;
                for(zi = 0; zi < 10*TRAINING_INSTANCES_PER_CLIENT; zi++){
                    // Feed forward
                    set_model_training_edge_popup(sum_m,0.5);
                    sum_m->fcls[m->n_fcl-1]->k_percentage = 1;
                    model_tensor_input_ff(sum_m,input_dimension,1,1,inputs[zi]);
                    for(kk = 0; kk < output_dimension; kk++){
                        error+=focal_loss(sum_m->fcls[sum_m->n_fcl-1]->post_activation[kk],outputs[zi][kk],2);
                    }
                    set_model_training_gd(sum_m);
                    reset_model(sum_m);  
                }
                printf("Error mixed model until client n %d on training set: %lf\n",j+1,error);
                if(j == number_connections-1){
                    FILE* fi = fopen("train.txt","a+");
                    fprintf(fi,"%lf\n",error);
                    fclose(fi);
                }
                set_model_training_edge_popup(sum_m2,0.5);
                sum_m2->fcls[sum_m2->n_fcl-1]->k_percentage = 1;    
                reset_model(sum_m2);
                zi;
                // test time
                error = 0;
                for(zi = 0; zi < 10*TRAINING_INSTANCES_PER_CLIENT; zi++){
                    // Feed forward
                    set_model_training_edge_popup(sum_m2,0.5);
                    sum_m2->fcls[m->n_fcl-1]->k_percentage = 1;
                    model_tensor_input_ff(sum_m2,input_dimension,1,1,inputs[zi]);
                    for(kk = 0; kk < output_dimension; kk++){
                        error+=focal_loss(sum_m2->fcls[sum_m2->n_fcl-1]->post_activation[kk],outputs[zi][kk],2);
                    }
                    set_model_training_gd(sum_m2);
                    reset_model(sum_m2);  
                }
                printf("Error not mixed model from client number %d on training set: %lf\n",j+1,error);
                // test time
                error = 0;
                for(zi = 0; zi < testing_instances; zi++){
                    // Feed forward
                    set_model_training_edge_popup(sum_m,0.5);
                    sum_m->fcls[m->n_fcl-1]->k_percentage = 1;
                    model_tensor_input_ff(sum_m,input_dimension,1,1,inputs_test[zi]);
                    for(kk = 0; kk < output_dimension; kk++){
                        error+=focal_loss(sum_m->fcls[sum_m->n_fcl-1]->post_activation[kk],outputs_test[zi][kk],2);
                    }
                    set_model_training_gd(sum_m);
                    reset_model(sum_m);  
                }
                printf("Error mixed model until client number %d on test set: %lf\n",j+1,error);
                if(j == number_connections-1){
                    FILE* fi = fopen("test.txt","a+");
                    fprintf(fi,"%lf\n",error);
                    fclose(fi);
                }
                // test time
                error = 0;
                for(zi = 0; zi < testing_instances; zi++){
                    // Feed forward
                    set_model_training_edge_popup(sum_m2,0.5);
                    sum_m2->fcls[m->n_fcl-1]->k_percentage = 1;
                    model_tensor_input_ff(sum_m2,input_dimension,1,1,inputs_test[zi]);
                    for(kk = 0; kk < output_dimension; kk++){
                        error+=focal_loss(sum_m2->fcls[sum_m2->n_fcl-1]->post_activation[kk],outputs_test[zi][kk],2);
                    }
                    set_model_training_gd(sum_m2);
                    reset_model(sum_m2);  
                }
                printf("Error not mixed model from client number %d on test set: %lf\n",j+1,error);
                reset_score_model(sum_m2);
            }
            
            // update m, reset m and copy the new weights
            reset_model(sum_m);// 
            free_model(m);
            m = copy_model(sum_m);
            memcopy_weights_to_vector_model(sum_m,buff);
            memcopy_scores_to_vector_model(sum_m,&buff[buffer_size/2]);
            
            // Saving the model
            save_model(sum_m,k+1);
        }
        
        // Deallocating Training resources
        free(ksource[0]);
        free(ksource);
        free_model(m);
        free_model(sum_m);

        for(i = 0; i < training_instances; i++){
            free(inputs[i]);
            free(outputs[i]);
        }
        free(inputs);
        free(outputs);
        
        for(i = 0; i < number_connections; i++){
            close(writers[i][0]);
            close(readers[i][1]);
        }
        
        free(writers);
        free(readers);
        free(w);
        free(r);
        free(buff);
        kill(pid,SIGKILL);

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
