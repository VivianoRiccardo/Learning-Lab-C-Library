#include <llab.h>
#include <unistd.h>
#include <signal.h>

#define PORT 9000
#define THREAD_PER_CLIENT 4
#define TRAINING_INSTANCES_PER_CLIENT 5000
#define INPUTS_PER_CLIENT 28*28*1*THREAD_PER_CLIENT
#define OUTPUT_PER_CLIENT 5*THREAD_PER_CLIENT

volatile sig_atomic_t child_pid = 0;
 
void term(int signum){
    kill(child_pid,SIGKILL);
}


int main(){
    srand(time(NULL));
    char* file = "0.bin";
    model* m = load_model(file);
    set_model_error(m,FOCAL_LOSS,0,0,2,NULL,10);
    
    int i,ret,pid,j,k;
    
    int number_connections = 5;
    
    int fd1[2];
    int fd2[2];
    
    
    ret = pipe(fd1);
    
    if(ret == -1){
        fprintf(stderr,"Error: not able to create the pipe\n");
        exit(1);
    }
    
    ret = pipe(fd2);
    
    if(ret == -1){
        fprintf(stderr,"Error: not able to create the pipe\n");
        exit(1);
    }
    
    pid = fork();// split father-son process
    
    if(pid == -1){
        fprintf(stderr,"Error: not able to create a son process\n");
        exit(1);
    }
    
    if(pid == 0){
        
        close(fd1[1]);//reader
        close(fd2[0]);//writer
        
        int buffer_size = 2*get_array_size_weights_model(m);
        free_model(m);
        
        run_client(PORT,"127.0.0.1",buffer_size+1,fd1[0],fd2[1]);
        
        close(fd1[0]);
        close(fd2[1]);
        return 0;
    }
    
    else{
        child_pid = pid;
        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = term;
        sigaction(SIGTERM, &action, NULL);
        
        
        int j,batch_size = THREAD_PER_CLIENT;
        close(fd1[0]);//writer
        close(fd2[1]);//reader
        int output_dimension = 10, input_dimension = 28*28, epoch = 2;
        int buffer_size = 2*get_array_size_weights_model(m);
        model** batch_m = (model**)malloc(sizeof(model*)*THREAD_PER_CLIENT);
        
        for(i = 0; i < THREAD_PER_CLIENT; i++){
            batch_m[i] = copy_model(m);
        }
        
        float* buff = (float*)calloc((buffer_size+1),sizeof(float));
        
        float** inputs = (float**)malloc(sizeof(float*)*TRAINING_INSTANCES_PER_CLIENT);
        float** outputs = (float**)malloc(sizeof(float*)*TRAINING_INSTANCES_PER_CLIENT);
        for(i = 0; i < TRAINING_INSTANCES_PER_CLIENT; i++){
            inputs[i] = (float*)calloc(input_dimension,sizeof(float));
            outputs[i] = (float*)calloc(output_dimension,sizeof(float));
        }
        while(1){
            float b1 = BETA1_ADAM;
            float b2 = BETA2_ADAM;
            long long unsigned int t1 = 1;
            clock_t t; 
            t = clock();
            int flag = 0;
            int ret = 0;
            while(ret == 0){// reading from server
                ret = 0;
                ret = read(fd2[0], buff, sizeof(float)*(buffer_size+1));
                printf("%d Bytes read from server\n",ret);
                t+=clock();
                if((t)/CLOCKS_PER_SEC > 7200)
                    flag = 1;
                    
                break;
            }
            if(flag)
                break;
            //copying in model and batch m the model from server
            memcopy_vector_to_weights_model(m,buff);
            memcopy_vector_to_scores_model(m,&buff[buffer_size/2]);
            set_model_training_edge_popup(m,0.5);
            m->fcls[m->n_fcl-1]->k_percentage = 1;
            reset_model(m);
            //reset_score_model(m);
            for(i = 0; i < THREAD_PER_CLIENT; i++){
                paste_model(m,batch_m[i]);
            }
            int z;
            // getting the data
            int file_number = round(buff[buffer_size]);
            char temp2[256];
            char temp3[5];
            temp3[0] = '.';
            temp3[1] = 'b';
            temp3[2] = 'i';
            temp3[3] = 'n';
            temp3[4] = '\0';
            itoa(100+file_number,temp2);
            strcat(temp2,temp3);
            printf("reading file %s\n",temp2);
            FILE* fil = fopen(temp2,"r");
            for(i = 0; i < TRAINING_INSTANCES_PER_CLIENT; i++){
                j = fread(inputs[i],input_dimension*sizeof(float),1,fil);
                j = fread(outputs[i],output_dimension*sizeof(float),1,fil);
            }
            fclose(fil);
            
            t = clock();
            // training
            for(i = 0; i < epoch+1; i++){
                // test time
                double error = 0;
                for(j = 0; j < TRAINING_INSTANCES_PER_CLIENT; j++){
                    set_model_training_edge_popup(m,0.5);
                    m->fcls[m->n_fcl-1]->k_percentage = 1;
                    // Feed forward
                    model_tensor_input_ff(m,input_dimension,1,1,inputs[j]);
                    for(k = 0; k < output_dimension; k++){
                        error+=focal_loss(m->fcls[m->n_fcl-1]->post_activation[k],outputs[j][k],2);
                    }
                    set_model_training_gd(m);
                    reset_model(m);  
                }
                printf("Error: %lf\n",error);
                printf("Epoch: %d/%d\n",i+1,epoch);
                if(i == epoch)
                    break;
                // training time
                shuffle_float_matrices(inputs,outputs,TRAINING_INSTANCES_PER_CLIENT);//shuffling
                set_model_training_edge_popup(m,0.5);
                m->fcls[m->n_fcl-1]->k_percentage = 1;
                for(j = 0; j < TRAINING_INSTANCES_PER_CLIENT; j+=batch_size){
                    //printf("batch_size: %d/%d\n",j/batch_size,TRAINING_INSTANCES_PER_CLIENT/batch_size);
                    ff_error_bp_model_multicore(batch_m,1,28,28,&inputs[j],batch_size,batch_size,&outputs[j],NULL);
                    sum_models_partial_derivatives(m,batch_m,batch_size);
                    update_model(m,0.01,0.9,batch_size,NESTEROV,&b1,&b2,NO_REGULARIZATION,0,0,&t1);
                    reset_model(m);
                    for(k = 0; k < batch_size; k++){
                        paste_model(m,batch_m[k]);
                        set_model_training_gd(batch_m[k]);
                        reset_model(batch_m[k]);
                        set_model_training_edge_popup(batch_m[k],0.5);
                        batch_m[k]->fcls[batch_m[k]->n_fcl-1]->k_percentage = 1;
                    } 
                }    
            }
            memcopy_scores_to_vector_model(m,buff);
            t+=clock();
            printf("%ld\n",(t)/CLOCKS_PER_SEC);
            ret = write(fd1[1],buff,sizeof(float)*(buffer_size+1));// writing to sons
            printf("%d Bytes written to server\n",ret);
            free_model(m);
            m = load_model(file);
            set_model_error(m,FOCAL_LOSS,0,0,2,NULL,10);
            set_model_training_edge_popup(m,0.5);
            m->fcls[m->n_fcl-1]->k_percentage = 1;
            
        }
        
        kill(pid,SIGKILL);
        free_model(m);
        for(i = 0; i < THREAD_PER_CLIENT; i++){
            free_model(batch_m[i]);
        }
        for(i = 0; i < TRAINING_INSTANCES_PER_CLIENT; i++){
            free(inputs[i]);
            free(outputs[i]);
        }
        free_model(m);
        free(batch_m);
        free(inputs);
        free(outputs);
        free(buff);
    }
}
