#include <llab.h>


int main(){
    
    // transformer treated as an autoencoder, it tries to replicate the input. To the encoder is given batch of the image (batch of 28)
    // and the decoder for each other batch of 28 tries to re imitate the input.
    // simple transformer with 1 encoder and 1 decoder and 28 logit units (models) at the end for the output, no sgd batch size = 1
    int i,j,k,z;
    srand(time(NULL));
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "../data/train.bin";
    int size = 0, training_instances = 100, input_dimension = 28*28, epochs = 10;
    char temp[2];
    temp[1] = '\0';
    
    // Reading the data in a char** vector
    read_file_in_char_vector(ksource,filename,&size);
    float** inputs = (float**)malloc(sizeof(float*)*training_instances);
    float** inputs2 = (float**)malloc(sizeof(float*)*training_instances);
    float* pos_enc = sin_cos_positional_encoding_vector(28,28);
    // Putting the data in float** vectors
    for(i = 0; i < training_instances; i++){
        inputs[i] = (float*)malloc(sizeof(float)*input_dimension);
        inputs2[i] = (float*)malloc(sizeof(float)*input_dimension);
        for(j = 0; j < input_dimension; j++){
            temp[0] = ksource[0][i*(input_dimension+1)+j];
            inputs[i][j] = atof(temp);
        }
        memcpy(inputs2[i],inputs[i],sizeof(float)*28*28);
        sum1D(inputs2[i],pos_enc,inputs2[i],28*28);
    }
    
    
    // last layer after the decoder
    fcl** model_fcl = (fcl**)malloc(sizeof(fcl*));
    model_fcl[0] = fully_connected(100,28,0,NO_DROPOUT,SIGMOID,0,0,NO_NORMALIZATION);
    model_fcl[0]->training_mode = FREEZE_TRAINING;
    // you can freeze the training of the logit units and watch how only the transformer tries to adapt to learn
    model* final_model = network(1,0,0,1,NULL,NULL,model_fcl);
    set_model_error(final_model,FOCAL_LOSS,0,0,2,NULL,28);
    
    
    
    model** final_models = (model**)malloc(sizeof(model*)*28);
    for(i = 0; i < 28; i++){
        final_models[i] = copy_model(final_model);
        reinitialize_w_model(final_models[i]);
    }
    
    
    
    //encoder model
    /*
    fcl** fcls = (fcl**)malloc(sizeof(fcl*)*2);
    fcls[0] = fully_connected(100,100,0,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION);
    fcls[1] = fully_connected(100,100,1,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    model* m = network(2,0,0,2,NULL,NULL,fcls);
    */
    
    cl** cls = (cl**)malloc(sizeof(cl*)*2);
    cls[0] = convolutional(1,100,1,1,1,10,1,1,0,0,1,1,0,0,1,1,NO_NORMALIZATION,RELU,NO_POOLING,0,CONVOLUTION,0);
    cls[1] = convolutional(1,1,1000,1,10,1,1,10,0,0,1,1,0,0,1,1,NO_NORMALIZATION,NO_ACTIVATION,NO_POOLING,0,CONVOLUTION,1);
    //cls[0]->training_mode = FREEZE_TRAINING;
    //cls[1]->training_mode = FREEZE_TRAINING;
    model* m = network(2,0,2,0,NULL,cls,NULL);
    
    fcl** fcl_en = (fcl**)malloc(sizeof(fcl*));
    fcl_en[0] = fully_connected(100,100,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcl_en[0]->training_mode = FREEZE_TRAINING;
    model* m_en = network(1,0,0,1,NULL,NULL,fcl_en);
    
    fcl** fcl_de = (fcl**)malloc(sizeof(fcl*));
    fcl_de[0] = fully_connected(100,100,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    model* m_de = network(1,0,0,1,NULL,NULL,fcl_de);
    
    fcl** fcl_de_en = (fcl**)malloc(sizeof(fcl*));
    fcl_de_en[0] = fully_connected(100,100,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    model* m_de_en = network(1,0,0,1,NULL,NULL,fcl_de_en);
    
    // encoder linear fcls
    fcl** fcls2 = (fcl**)malloc(sizeof(fcl*)*6);
    fcls2[0] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls2[1] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls2[2] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls2[3] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls2[4] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls2[5] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    
    fcls2[0]->training_mode = FREEZE_TRAINING;
    fcls2[1]->training_mode = FREEZE_TRAINING;
    fcls2[2]->training_mode = FREEZE_TRAINING;
    fcls2[3]->training_mode = FREEZE_TRAINING;
    fcls2[4]->training_mode = FREEZE_TRAINING;
    fcls2[5]->training_mode = FREEZE_TRAINING;
    // encoder normalization
    scaled_l2_norm** l2 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*2);
    l2[0] = scaled_l2_normalization_layer(100);
    l2[1] = scaled_l2_normalization_layer(100);
    l2[0]->training_mode = FREEZE_TRAINING;
    l2[1]->training_mode = FREEZE_TRAINING;
    float beta1_adam = m->beta1_adam;
    float beta2_adam = m->beta2_adam;
    //decoder model
    fcl** fcls4 = (fcl**)malloc(sizeof(fcl*)*2);
    fcls4[0] = fully_connected(100,100,0,NO_DROPOUT,RELU,0,0,NO_NORMALIZATION);
    fcls4[1] = fully_connected(100,100,1,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    model* m2 = network(2,0,0,2,NULL,NULL,fcls4);
    
    // decoder linear fcls
    fcl** fcls3 = (fcl**)malloc(sizeof(fcl*)*12);
    fcls3[0] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[1] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[2] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[3] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[4] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[5] = fully_connected(28*28,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[6] = fully_connected(100,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[7] = fully_connected(100,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[8] = fully_connected(100,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[9] = fully_connected(100,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[10] = fully_connected(100,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    fcls3[11] = fully_connected(100,50,0,NO_DROPOUT,NO_ACTIVATION,0,0,NO_NORMALIZATION);
    
    // decoder normalization
    scaled_l2_norm** l3 = (scaled_l2_norm**)malloc(sizeof(scaled_l2_norm*)*3);
    l3[0] = scaled_l2_normalization_layer(100);
    l3[1] = scaled_l2_normalization_layer(100);
    l3[2] = scaled_l2_normalization_layer(100);
    
    // initializing the encoder
    transformer_encoder** e = (transformer_encoder**)malloc(sizeof(transformer_encoder*));
    e[0] = transformer_encoder_layer(m,m_en,fcls2,l2,100,2,TRANSFORMER_NO_RESIDUAL,SCALED_L2_NORMALIZATION,TRANSFORMER_RESIDUAL,NO_NORMALIZATION,STANDARD_ATTENTION);
    
    // initializing the decoder
    transformer_decoder** d = (transformer_decoder**)malloc(sizeof(transformer_decoder*));
    d[0] = transformer_decoder_layer(100,100,2,2,TRANSFORMER_NO_RESIDUAL,SCALED_L2_NORMALIZATION,TRANSFORMER_RESIDUAL,SCALED_L2_NORMALIZATION,TRANSFORMER_RESIDUAL,SCALED_L2_NORMALIZATION,MASKED_ATTENTION,STANDARD_ATTENTION,100,m2,m_de,m_de_en,fcls3,l3);
    
    // must be specified the conenction encoder decoder
    int** con = (int**)malloc(sizeof(int*));
    con[0] = (int*)malloc(sizeof(int));
    con[0][0] = 1;
    
    // initializing the transformer
    transformer* t = transf(1,1,e,d,con);
    long long unsigned int time = 1;
    // training, remember inputs2 is the input for encoder and decoder, must predict inputs
    for(i = 0; i < epochs; i++){//epochs
        for(j = 0; j < 28; j++){
            save_model(final_models[j],i*29+j);
        }
        save_transf(t,i*29+28);
        // shuffling the inputs-outputs
        shuffle_float_matrices(inputs,inputs2,training_instances);
        
        for(j = 0; j < training_instances; j++){
            // preparing the input for the decoder
            float* enc = (float*)calloc(28*28,sizeof(float));
            memcpy(&enc[28],inputs,27*28*sizeof(float));
            for(k = 0; k < 28; k++){
                enc[k] = 1;
            }
            sum1D(enc,pos_enc,enc,28*28);
            // feed forward transformer
            printf("Training instance number %d/%d\n",j+1,100);
            //transf_ff(t,inputs2[j],28*28,enc,28*28,RUN_ALL_TRANSF);
            encoder_transformer_ff(inputs2[j],t->te[0],28*28);
            // ff and bp for the logits of the transf
            float* err1 = NULL;
            for(k = 0; k < 28; k++){
                if(!k)
                    //err1 = ff_error_bp_model_once(final_models[k],1,1,100,get_output_layer_from_encoder_transf(t->td[t->n_td-1]->e),&inputs[j][k*28]);
                    err1 = ff_error_bp_model_once(final_models[k],1,1,100,get_output_layer_from_encoder_transf(t->te[0]),&inputs[j][k*28]);
                else{
                    //float* err2 = ff_error_bp_model_once(final_models[k],1,1,100,get_output_layer_from_encoder_transf(t->td[t->n_td-1]->e),&inputs[j][k*28]);
                    float* err2 = ff_error_bp_model_once(final_models[k],1,1,100,get_output_layer_from_encoder_transf(t->te[0]),&inputs[j][k*28]);
                    sum1D(err1,err2,err1,100);
                }
            }
            // transf bp
            //transf_bp(t,inputs2[j],28*28,enc,28*28,err1,RUN_ALL_TRANSF);
            encoder_transformer_bp(inputs2[j],t->te[0],28*28,err1);
            // clipping gradient for the entire model (can be commented - not recommended)
            //general_clipping_gradient(final_models,NULL,&t,28,0,1,5);
            //clipping_gradient_transf_encoder(t->te[0],10);
            // here comes the update
            update_transformer_encoder(t->te[0],0.001,0.9,1,DIFF_GRAD,&beta1_adam,&beta2_adam,NO_REGULARIZATION,0,0,&time);
            reset_transf(t);
            beta1_adam*=m->beta1_adam;
            beta2_adam*=m->beta2_adam;
            time++;
            for(k = 0; k < 28; k++){
            //    update_model(final_models[k],0.001,0.9,1,NESTEROV,NULL,NULL,NO_REGULARIZATION,0,0,NULL);
                reset_model(final_models[k]);
            }
            free(enc);
        }
    }
    
    // freeing the space allocated
    free(ksource[0]);
    free(ksource);
    for(i = 0; i < 28; i++){
        free_model(final_models[i]);
    }
    free_model(final_model);
    free(final_models);
    free_transf(t);
    
    
    // test
    model** mm = (model**)malloc(sizeof(model*)*28);
    char temp2[256];
    char temp3[5];
    temp3[0] = '.';
    temp3[1] = 'b';
    temp3[2] = 'i';
    temp3[3] = 'n';
    temp3[4] = '\0';
    long long unsigned int** cm;
    printf("Testing phase!\n");
    double error = 0;
    // Testing
    for(i = 0; i < epochs; i++){
        printf("Model N. %d/%d\n",i+1,epochs);
        // Loading the logit models
        for(j = 0; j < 28; j++){
            itoa(i*29+j,temp2);
            strcat(temp2,temp3);
            mm[j] = load_model(temp2);
        }
        // loading the transformer
        itoa(i*29+28,temp2);
        strcat(temp2,temp3);
        FILE* fi = fopen(temp2,"r");
        transformer* tt = load_transf(fi);
        fclose(fi);
        
        // let compute the error
        for(j = 0; j < training_instances; j++){
            
            float* enc = (float*)calloc(28*28,sizeof(float));
            // preparing the input
            memcpy(&enc[28],inputs,27*28*sizeof(float));
            for(k = 0; k < 28; k++){
                enc[k] = 1;
            }
            sum1D(enc,pos_enc,enc,28*28);
            
            
            // transf ff
            //transf_ff(tt,inputs2[j],28*28,enc,28*28,RUN_ALL_TRANSF);
            encoder_transformer_ff(inputs2[j],tt->te[0],28*28);
            float* err = (float*)calloc(28,sizeof(float));
            for(k = 0; k < 27; k++){
                //model_tensor_input_ff(mm[k],1,1,100,get_output_layer_from_encoder_transf(tt->td[tt->n_td-1]->e));
                model_tensor_input_ff(mm[k],1,1,100,get_output_layer_from_encoder_transf(tt->te[0]));
                focal_loss_array(mm[k]->output_layer,&inputs[j][k*28],err,2,28);
                for(z = 0; z < 28; z++){
                    if(mm[k]->output_layer[z] < 0.5) mm[k]->output_layer[z] = 0;
                    else mm[k]->output_layer[z] = 1;
                }
                if(!j)
                    cm = confusion_matrix(mm[k]->output_layer, &inputs[j][k*28],NULL, 28,0.5);
                else
                    cm = confusion_matrix(mm[k]->output_layer, &inputs[j][k*28],cm, 28,0.5);
                    
                //reset_transformer_encoder(tt->te[0]);
                //sum1D(mm[k]->output_layer,&pos_enc[(k+1)*28],&enc[(k+1)*28],28);
                //reset_transf_decoders(tt);
                //transf_ff(tt,inputs2[j],28*28,enc,28*28,RUN_ONLY_DECODER);
            }
            model_tensor_input_ff(mm[k],1,1,100,get_output_layer_from_encoder_transf(tt->te[0]));
            //model_tensor_input_ff(mm[k],1,1,100,get_output_layer_from_encoder_transf(tt->td[0]->e));
            focal_loss_array(mm[k]->output_layer,&inputs[j][k*28],err,2,28);
            
            //error
            for(k = 0; k < 28; k++){
                error+=err[k];
                reset_model(mm[k]);
            }
            
            reset_transf(tt);
            free(enc);
            free(err);
            
            
        }
        //print the error
        printf("%lf\n",error);
        error = 0;
        for(j = 0;j < 28; j++){
            free_model(mm[j]);
        }
        free_transf(tt);
        printf("Accuracy, Precision, Sensitivity, Specificity:\n");
        print_accuracy(cm,28);
        print_precision(cm,28);
        print_sensitivity(cm,28);
        print_specificity(cm,28);
        for(j = 0; j < 28*2; j++){
            free(cm[j]);
        }
        free(cm);
    }
    
    // freeing the space allocated
    free(mm);
    free(pos_enc);
    free_matrix(inputs,training_instances);
    free_matrix(inputs2,training_instances);
}
