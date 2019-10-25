#include <llab.h>
#include <unistd.h>
#define THREADS 8 // can be changed
#define BUFFER 10000000 //10 milion, can be changed

void game_run(int fd1, int fd2,int fd3);

int main(){
    
    //Resources for the communication father-son, father = model training, son = game
    int games = 1000000; // 1 milion of games, can be changed
    int fd1[2];
    int fd2[2];
    int fd3[2];
    int ret;
    int pid;
    
    // Inputs frame dimension (are based on your rl problem), can be changed:
    int channels = 2,input_rows = 20, input_cols = 10;
    
    // Creating the channels to comunicate, should not be changed
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
    
    ret = pipe(fd3);
    
    if(ret == -1){
        fprintf(stderr,"Error: not able to create the pipe\n");
        exit(1);
    }
    
    // Creating the son who is gonna play the game, should not be changed
    
    pid = fork();
    
    if(pid == -1){
        fprintf(stderr,"Error: not able to create a son process\n");
        exit(1);
    }
    
    if(pid == 0){
        // Son
        srand(time(NULL));
        close(fd1[1]);
        close(fd2[0]);
        close(fd3[0]);
        int i;
        for(i = 0; i < games; i++){
            game_run(fd1[0],fd2[1],fd3[1]);// game_run is your game running and receivng the pipes to communicate
        }
        
        close(fd1[0]);
        close(fd2[1]);
        close(fd3[1]);
        
        return 0;
        
        //fd1 receives messages
        //fd2 sends messages
        //fd3 sends messages
    }
    
    else{
    
        srand(time(NULL));
        // Father
        close(fd1[0]);
        close(fd2[1]);
        close(fd3[1]);
        
        //fd1 sends messages
        //fd2 receives messages
        //fd3 receives messages
        
        int i,j,k,action,size = channels*input_rows*input_cols,count,game_goes_on = 1,max;// iterators, should not be changed, size instead is the input size
        int mini_batch_size = THREADS, mini_batch_counter = 0,mini_batch_limit = 1;// based on your choice, mini_batch_limit is the param that says: after tot number of frames, update my teta_ network
        unsigned long long int t = 1;
        
        float* array = (float*)malloc(sizeof(float)*size);//first array to handle the frame from the game, should not be changed, the frames are given in float format
        float* temp_array = (float*)malloc(sizeof(float)*size);//second array to handle the frame from the game, should not be changed, the frame are given in float format
        
        float** all_arrays_positive = (float**)malloc(sizeof(float*)*BUFFER);//The buffer with positive rewards, should not be changed
        float** all_arrays = (float**)malloc(sizeof(float*)*BUFFER);//The buffer with no positive rewards, should not be changed
        float** all_arrays2 = (float**)malloc(sizeof(float*)*BUFFER);//The buffer with no positive rewards, should not be changed
        float** all_arrays2_positive = (float**)malloc(sizeof(float*)*BUFFER);//The buffer with positive rewards, should not be changed
        float* rewards = (float*)malloc(sizeof(float)*BUFFER);// all the no positive rewards, should not be changed
        float* rewards2 = (float*)malloc(sizeof(float)*BUFFER);// all positive rewards, should not be changed
        int* actions = (int*)malloc(sizeof(int)*BUFFER);// all the actions with no positive rewards, should not be changed
        int* actions2 = (int*)malloc(sizeof(int)*BUFFER);// all the actions with positive rewards, should not be changed
        int buffer_counter = 0,buffer_counter2 = 0;// number of frames with no positive rewards, and frames with positive rewards, should not be changed
        
        float epsilon_greedy = 0.5;//the epsilon param is used for the epsilon greedy algorithm, can be changed
        float lambda = 0.5;
        
        float** input; float** output; float** ret_err; //input and output for the training, should not be changed
        
        float reward;// reward: where to store the reward for each timestep, should not be changed
        
        float lr = 0.01,momentum = 0.9,b1 = 0,b2 = 0,l2_value = 0.003; // can be changed
        
        int* replayed_times1 = (int*)malloc(sizeof(int)*BUFFER);// Used for the prioritized sampling of no positive transitions, should not be changed
        int* replayed_times2 = (int*)malloc(sizeof(int)*BUFFER);// Used for the prioritized sampling of positive transitions, should not be changed
        
        // These params can be changed
        double alpha = 0.6; //param used for prioritizing
        float fi = 0.5;
        float tau = 0.05;
        
        // Network sizes can be decided by you, only end with a fullyconnected layer
        int kernel_rows = 3,kernel_cols = 3;
        int stride1 = 1, padding1 = 0;
        int stride2 = 1, padding2 = 2;
        int pool_rows = 3, pool_cols = 3;
        int n_kernels = 60;
        int n_kernels2 = 40;
        int n_kernels3 = 20;
        int n_neurons1 = 256;
        int n_neurons2 = 64;
        int n_neurons3 = 4;
        
        // Initialization of the Network
        cl** c = (cl**)malloc(sizeof(cl*)*3);
        fcl** f = (fcl**)malloc(sizeof(fcl*)*3);
        
        c[0] = convolutional(channels,input_rows,input_cols,kernel_rows,kernel_cols,n_kernels,stride1,stride1,padding1,padding1,stride2,stride2,padding2,padding2,pool_rows,pool_cols,LOCAL_RESPONSE_NORMALIZATION,RELU,MAX_POOLING,0,CONVOLUTION,0);
        c[1] = convolutional(n_kernels,input_rows,input_cols,kernel_rows,kernel_cols,n_kernels2,stride1,stride1,padding1,padding1,stride2,stride2,padding2,padding2,pool_rows,pool_cols,LOCAL_RESPONSE_NORMALIZATION,RELU,MAX_POOLING,0,CONVOLUTION,1);
        c[2] = convolutional(n_kernels2,input_rows,input_cols,kernel_rows,kernel_cols,n_kernels3,stride1,stride1,padding1,padding1,stride2,stride2,padding2,padding2,pool_rows,pool_cols,LOCAL_RESPONSE_NORMALIZATION,RELU,NO_POOLING,0,CONVOLUTION,2);
        
        f[0] = fully_connected(n_kernels3*c[2]->rows1*c[2]->cols1,n_neurons1,3,NO_DROPOUT,RELU,0);
        f[1] = fully_connected(n_neurons1,n_neurons2,4,NO_DROPOUT,RELU,0);
        f[2] = fully_connected(n_neurons2,n_neurons3,5,NO_DROPOUT,TANH,0);
        

        
        model* m = network(6,0,3,3,NULL,c,f);
        
        int n_weights = count_weights(m);// we need number of total weights for the L2 regularization
        
        model* actual_m = copy_model(m);
        model* sum_m = copy_model(m);
        model** batch_m = (model**)malloc(sizeof(model*)*mini_batch_size);
        for(i = 0; i < mini_batch_size; i++){
            batch_m[i] = copy_model(m);
        }
        
        // Allocating space for input and output for training, can be changed
        output = (float**)malloc(sizeof(float*)*mini_batch_size);
        input = (float**)malloc(sizeof(float*)*mini_batch_size);
        ret_err = (float**)malloc(sizeof(float*)*mini_batch_size);
        for(i = 0; i < mini_batch_size; i++){
            output[i] = (float*)malloc(sizeof(float)*n_neurons3);
            input[i] = (float*)malloc(sizeof(float)*size);
        }
        
        // prioritized DQ algorithm

        for(count = 0; count < games && buffer_counter+buffer_counter2 <= BUFFER; count++){

            // First array when the game starts
            while(read(fd2[0],array,sizeof(float)*size)== 0); // read the frame at the beginning of the game (all 0s(?))
            while(read(fd3[0],&reward,sizeof(float)) == 0);// read the reward at the beginning of the game (0)

            game_goes_on = 1; // start the game
            
            while(game_goes_on && buffer_counter+buffer_counter2 <= BUFFER){ // while game goes on and until we don't exceed the buffer size we keep playing
                
                // Chosing the action
                
                printf("buffers: %d\n", buffer_counter+buffer_counter2);// just print how many frames we have seen
                // If the fate decides to take a random action, we take a random action
                if(r2() < epsilon_greedy)
                    action = rand()%actual_m->fcls[actual_m->n_fcl-1]->output;
                
                // Else we perform the feed forward across actual_m and we chose the best action
                else{
                    
                    // Performing the feed forward and taking the best action
                    model_tensor_input_ff(actual_m,channels,input_rows,input_cols,array);
                    
                    max = -2;
                    action = -1;
                    for(i = 0; i < actual_m->fcls[actual_m->n_fcl-1]->output; i++){
                        if(actual_m->fcls[actual_m->n_fcl-1]->post_activation[i] > max){
                            action = i;
                            max = actual_m->fcls[actual_m->n_fcl-1]->post_activation[i];
                        }
                    }
                    reset_model(actual_m);
                    
                }
                
                // End choice of action
                
                // Getting the new frame
                
                // Sending the action chosen
                i = write(fd1[1],&action,sizeof(int));
                
                // Waiting for next state and reward
                while(read(fd2[0],temp_array,sizeof(float)*size) == 0);
                while(read(fd3[0],&reward,sizeof(float)) == 0);
                
                printf("Reward: %f\n",reward);// just print the reward
                
                // if reward = -1 then is the end of this game and game_goes_on = 0
                if(reward == -1)
                    game_goes_on = 0;
               
                // Storing the transitions in the buffers
                    
                if(epsilon_greedy > 0.1)// we decrease epsilon greedy until 0.1, can be changed
                    epsilon_greedy -= 0.00001;//can be changed
                
                // Allocating space on all_arrays(sequence t) and all_arrays2 (Sequence t+1)
                
                if(reward <= 0){ // If we got a no positive reward then we store in the first buffers
                    all_arrays[buffer_counter] = (float*)malloc(sizeof(float)*size);
                    all_arrays2[buffer_counter] = (float*)malloc(sizeof(float)*size);
                    copy_array(array,all_arrays[buffer_counter],size);
                    copy_array(temp_array,all_arrays2[buffer_counter],size);
                    rewards[buffer_counter] = reward;
                    actions[buffer_counter] = action;
                    replayed_times1[buffer_counter] = 0;
                    buffer_counter++;
                    
                }
                
                else{
                    all_arrays_positive[buffer_counter2] = (float*)malloc(sizeof(float)*size);
                    all_arrays2_positive[buffer_counter2] = (float*)malloc(sizeof(float)*size);
                    copy_array(array,all_arrays_positive[buffer_counter2],size);
                    copy_array(temp_array,all_arrays2_positive[buffer_counter2],size);
                    rewards2[buffer_counter2] = reward;
                    actions2[buffer_counter2] = action;
                    replayed_times2[buffer_counter2] = 0;
                    buffer_counter2++;
                }
                mini_batch_counter++;
                
                
                // For the next stage sequence t (array) is gonna be sequence t+1 (temp_array)
                copy_array(temp_array,array,size);
                
                
                
                // If the mini batch counter reaches the mini batch limit value then will be performed the gradient descent on actual m
                if(mini_batch_counter == mini_batch_limit && buffer_counter2 >= mini_batch_size && buffer_counter >= mini_batch_size){
                    
                    shuffle_float_matrices_float_int_vectors(all_arrays,all_arrays2,rewards,actions,buffer_counter);
                    shuffle_float_matrices_float_int_vectors(all_arrays_positive,all_arrays2_positive,rewards2,actions2,buffer_counter2);
                    
                    mini_batch_counter = 0;//should not be changed
                    if(fi > 0.25)//the fi param decrease until 0.25, can be changed
                        fi -= 0.0000001;//can be changed
                        
                    // Chosing the batch using prioritized algorithm
                    int* last_index1 = (int*)malloc(sizeof(int)*mini_batch_size);
                    int* last_index2 = (int*)malloc(sizeof(int)*mini_batch_size);
                    int current_index = -1;
                    float temp_value;
                    int batch_flag;
                    for(i = 0; i < mini_batch_size; i++){
                        last_index1[i] = -1;
                        last_index2[i] = -1;
                    }
                    for(i = 0; i < mini_batch_size; i++){
                        if(r2() < fi){ // if random < f1 then we select positive instance
                            max = -2;
                            for(j = 0; j < buffer_counter2; j++){
                                batch_flag = 0;
                                for(k = 0; k < mini_batch_size; k++){//checking if this instance has not been selected in this batch
                                    if(j == last_index2[k]){
                                        batch_flag = 1;
                                        break;
                                    }
                                }
                                if(!batch_flag){//if this instance has not been selected we compute the probability to been selected
                                    temp_value = (float)pow((double)(1/((double)(1+replayed_times2[j]))),alpha);
                                    if(temp_value > max){
                                        max = temp_value;
                                        last_index2[i] = j;
                                    }
                                }
                                
                            }
                            replayed_times2[last_index2[i]]++;// the replayed times value for this selected instance increases
                        }
                        
                        else{ // if random >= f1 then we select negative instance
                            max = -2;
                            for(j = 0; j < buffer_counter; j++){
                                batch_flag = 0;
                                for(k = 0; k < mini_batch_size; k++){//checking if this instance has not been selected in this batch
                                    if(j == last_index1[k]){
                                        batch_flag = 1;
                                        break;
                                    }
                                }
                                if(!batch_flag){//if this instance has not been selected we compute the probability to been selected
                                    temp_value = (float)pow((double)(1/((double)(1+replayed_times1[j]))),alpha);
                                    
                                    if(temp_value > max){
                                        max = temp_value;
                                        last_index1[i] = j;
                                    }
                                }
                                
                            }
                            replayed_times1[last_index1[i]]++;// the replayed times value for this selected instance increases
                        }
                        
                    }
                    
                    // For all the instances in the batch we set the y target value
                    for(i = 0; i < mini_batch_size; i++){
                        if(last_index2[i] == -1){//if last index2[i] == -1 then the instance i is a negative instance (index stored in last_index1)
                            copy_array(all_arrays2[last_index1[i]],input[i],size);
                            // Rewards[i] = -1 means that we reached the end of the game (lost)
                            if(rewards[last_index1[i]] == -1){

                                for(j = 0; j < m->fcls[m->n_fcl-1]->output; j++){
                                    if(j == actions[last_index1[i]])
                                        output[i][j] = -1;
                                }
                            }
                            
                            // If rewards[i] != -1 it means that the game was not ended and we must perform the best action value with the model m
                            else{
                                
                                // Performing the feed forward and selecting the max value given in output by the feed forward
                                model_tensor_input_ff(m,channels,input_rows,input_cols,input[i]);
                                for(j = 0; i < m->fcls[m->n_fcl-1]->output; i++){
                                    if(m->fcls[m->n_fcl-1]->post_activation[i] > max)
                                        max = m->fcls[m->n_fcl-1]->post_activation[i];
                                    
                                }
                                reset_model(m);
                                
                                // The output is set to r[i]+lambda*max value across the actions.
                                for(j = 0; j < m->fcls[m->n_fcl-1]->output; j++){
                                    if(j == actions[last_index1[i]])
                                        output[i][j] = rewards[last_index1[i]]+lambda*max;
                                }
                            }
                        }
                        
                        else{//if last index2[i] != -1 then the instance i is a positive instance (index stored in last_index2)
                            // Rewards[i] = -1 means that we reached the end of the game (lost)
                            copy_array(all_arrays2_positive[last_index2[i]],input[i],size);
                            if(rewards2[last_index2[i]] == -1){

                                for(j = 0; j < m->fcls[m->n_fcl-1]->output; j++){
                                    if(j == actions2[last_index2[i]])
                                        output[i][j] = -1;
                                }
                            }
                            
                            // If rewards[i] != -1 it means that the game was not ended and we must perform the best action value with the model m
                            else{
                   
                                // Performing the feed forward and selecting the max value given in output by the feed forward
                                model_tensor_input_ff(m,channels,input_rows,input_cols,input[i]);
                                for(j = 0; i < m->fcls[m->n_fcl-1]->output; i++){
                                    if(m->fcls[m->n_fcl-1]->post_activation[i] > max)
                                        max = m->fcls[m->n_fcl-1]->post_activation[i];
                                    
                                }
                                reset_model(m);
                                
                                // The output is set to r[i]+lambda*max value across the actions.
                                for(j = 0; j < m->fcls[m->n_fcl-1]->output; j++){
                                    if(j == actions2[last_index2[i]])
                                        output[i][j] = rewards2[last_index2[i]]+lambda*max;
                                }
                            }
                        }
                    }
                    
                    // Now, computing the feed forward across the batch for the real network actual_m
                    model_tensor_input_ff_multicore(batch_m,channels,input_rows,input_cols,input,mini_batch_size,THREADS);
                    
                    for(i = 0; i < mini_batch_size; i++){
                        if(last_index2[i] == -1){//if last index2[i] == -1 then the instance i is a negative instance (index stored in last_index1)
                                                        
                            // Setting the error DL/Do where o is the tanh function
                            for(j = 0; j < m->fcls[m->n_fcl-1]->output; j++){
                                if(j == actions[last_index1[i]])
                                    output[i][j] = derivative_mse(batch_m[i]->fcls[m->n_fcl-1]->post_activation[j],output[i][j]);
                                else
                                    output[i][j] = 0;
                            }

                        }
                    
                        else{//if last index2[i] != -1 then the instance i is a positive instance (index stored in last_index2)

                            // Setting the error DL/Do where o is the tanh function
                            for(j = 0; j < m->fcls[m->n_fcl-1]->output; j++){
                                if(j == actions[last_index2[i]])
                                    output[i][j] = derivative_mse(batch_m[i]->fcls[m->n_fcl-1]->post_activation[j],output[i][j]);
                                else
                                    output[i][j] = 0;
                            }
     
                        }
                    }
                    
                    model_tensor_input_bp_multicore(batch_m,channels,input_rows,input_cols,input,mini_batch_size,THREADS,output,m->fcls[m->n_fcl-1]->output,ret_err);
                    for(i = 0; i < mini_batch_size; i++){
                        sum_model_partial_derivatives(batch_m[i],sum_m,sum_m);
                        reset_model(batch_m[i]);
                    }
                    
                    // Gradient descent (with L2 Regularization)
                    update_model(sum_m,lr,momentum,mini_batch_size,RADAM,&b1,&b2,L2_REGULARIZATION,n_weights,l2_value*mini_batch_size,&t);
                    paste_model(sum_m,actual_m);
                    reset_model(actual_m);
                    reset_model(sum_m);
                    for(i = 0; i < mini_batch_size; i++){
                        paste_model(sum_m,batch_m[i]);
                    }
                    free(last_index1);
                    free(last_index2);
                    slow_paste_model(actual_m,m,tau);// update model m with a tau step
                    
                    save_model(actual_m,buffer_counter+buffer_counter2);
                }
                
                else if(mini_batch_counter == mini_batch_limit)
                    mini_batch_counter = 0;
            }
        }
        // Closing pipes and deallocating resources
        close(fd1[1]);
        close(fd2[0]);
        close(fd3[0]);
        free(array);//size
        free(temp_array);//size
        for(i = 0; i < buffer_counter; i++){
            free(all_arrays[i]);
            free(all_arrays2[i]);
        }
        
        free(all_arrays);
        free(all_arrays2);
        
        
        for(i = 0; i < buffer_counter2; i++){
            free(all_arrays_positive[i]);
            free(all_arrays2_positive[i]);
        }
        
        free(all_arrays_positive);
        free(all_arrays2_positive);
        free(rewards);
        free(rewards2);
        free(actions);
        free(actions2);
        free(ret_err);
        free(replayed_times1);
        free(replayed_times2);
        
        for(i = 0; i < mini_batch_size; i++){
            free(input);
            free(output);
            free_model(batch_m[i]);
        }
        free(batch_m);
        free(input);
        free(output);
        free_model(m);
        free_model(sum_m);
        free_model(actual_m);
    }
    
    return 0;
    

}
