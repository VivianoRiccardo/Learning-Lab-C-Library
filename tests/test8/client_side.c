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
        
        close(fd1[0]);//reader
        close(fd2[0]);//writer
        
        int buffer_size = get_array_size_params_model(m);
        free_model(m);
        
        run_client(PORT,"127.0.0.1",buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT,fd1[1],fd2[1]);
        
        close(fd1[1]);
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
        close(fd1[1]);//writer
        close(fd2[1]);//reader
        int buffer_size = get_array_size_params_model(m),output_dimension = 10;
        model** batch_m = (model**)malloc(sizeof(model*)*THREAD_PER_CLIENT);
        
        for(i = 0; i < THREAD_PER_CLIENT; i++){
            batch_m[i] = copy_model(m);
        }
        
        float* buff = (float*)calloc((buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT),sizeof(float));
        
        float** inputs = (float**)malloc(sizeof(float*)*THREAD_PER_CLIENT);
        float** outputs = (float**)malloc(sizeof(float*)*THREAD_PER_CLIENT);
        float** errors = (float**)malloc(sizeof(float*)*THREAD_PER_CLIENT);
        float** ret_err = (float**)malloc(sizeof(float*)*THREAD_PER_CLIENT);
        for(i = 0; i < THREAD_PER_CLIENT; i++){
            inputs[i] = (float*)calloc(INPUTS_PER_CLIENT/THREAD_PER_CLIENT,sizeof(float));
            outputs[i] = (float*)calloc(OUTPUT_PER_CLIENT/THREAD_PER_CLIENT,sizeof(float));
            errors[i] = (float*)calloc(OUTPUT_PER_CLIENT/THREAD_PER_CLIENT,sizeof(float));
        }
        
        while(1){
            clock_t t; 
            t = clock();
            int flag = 0; 
            while(read(fd2[0], buff, sizeof(float)*(buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT)) == 0){
                t+=clock();
                if((t)/CLOCKS_PER_SEC > 10)
                    flag = 1;
                    
                break;
            }
            if(flag)
                break;
            
            for(i = 0; i < THREAD_PER_CLIENT; i++){
                memcopy_vector_to_params_model(batch_m[i],buff);
            }
            
            memcpy(&buff[buffer_size],inputs[0],INPUTS_PER_CLIENT/THREAD_PER_CLIENT);
            memcpy(&buff[buffer_size+INPUTS_PER_CLIENT/THREAD_PER_CLIENT],outputs[0],OUTPUT_PER_CLIENT/THREAD_PER_CLIENT);
            memcpy(&buff[buffer_size+INPUTS_PER_CLIENT/THREAD_PER_CLIENT+OUTPUT_PER_CLIENT/THREAD_PER_CLIENT],inputs[1],INPUTS_PER_CLIENT/THREAD_PER_CLIENT);
            memcpy(&buff[buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT/THREAD_PER_CLIENT],outputs[1],OUTPUT_PER_CLIENT/THREAD_PER_CLIENT);
            
            model_tensor_input_ff_multicore(batch_m,1,28,28,inputs,THREAD_PER_CLIENT,THREAD_PER_CLIENT);
            
            for(j = 0; j < THREAD_PER_CLIENT; j++){
                derivative_cross_entropy_array(batch_m[j]->fcls[1]->post_activation,outputs[j],errors[j],output_dimension);
            }
            
            model_tensor_input_bp_multicore(batch_m,1,28,28,inputs,THREAD_PER_CLIENT,THREAD_PER_CLIENT,errors,output_dimension,ret_err);
            
            sum_model_partial_derivatives(batch_m[0],batch_m[1],batch_m[0]);
            memcopy_derivative_params_to_vector_model(batch_m[0],buff);
            ret = write(fd1[0],buff,sizeof(float)*(buffer_size+INPUTS_PER_CLIENT+OUTPUT_PER_CLIENT));// writing to sons
            
            for(j = 0; j < THREAD_PER_CLIENT; j++){
                reset_model(batch_m[j]);
            }
        }
        
        kill(pid,SIGKILL);
        free_model(m);
        for(i = 0; i < THREAD_PER_CLIENT; i++){
            free_model(batch_m[i]);
            free(inputs[i]);
            free(outputs[i]);
            free(errors[i]);
        }
        free(batch_m);
        free(ret_err);
        free(inputs[i]);
        free(outputs);
        free(errors);
        free(buff);
    }
}
