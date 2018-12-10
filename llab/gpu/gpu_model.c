#include "llab.h"


/* This function returns a model allocated on the context ctx. the param m of the gpu_model
 * is a copy of the input model
 * 
 * Input:
 * 
 * 
 *             @ model* m:= the model that must be copied on gpu
 *             @ cl_context ctx:= the context where the gpu_model is allocated
 * 
 * */
gpu_model* init_gpu_model(model* m, cl_context ctx ){
    int i,j,ret,count,count2,k;
    gpu_model* gm = (gpu_model*)malloc(sizeof(gpu_model));
    
    cl_mem** rls = NULL;
    cl_mem** cls = NULL;
    cl_mem** fcls = NULL;
    
    if(m->n_rl)
        rls = (cl_mem**)malloc(sizeof(cl_mem*)*m->n_rl);
        
    
    if(m->n_cl)
        cls = (cl_mem**)malloc(sizeof(cl_mem*)*m->n_cl);
        
    
    if(m->n_fcl)
        fcls = (cl_mem**)malloc(sizeof(cl_mem*)*m->n_fcl);
        
    
    for(i = 0; i < m->n_rl; i++){
        count = 0;
        count2 = 0;
        count2+=m->rls[i]->n_cl;
        for(j = 0; j < m->rls[i]->n_cl; j++){
            count+=m->rls[i]->cls[j]->n_kernels;
        }
        rls[i] = (cl_mem*)malloc(sizeof(cl_mem)*(1+4*(m->rls[i]->cl_output->n_kernels+count)+12*(1+count2)));
        load_on_gpu_rl_layer(&rls[i],m->rls[i],ctx);
    }
    
    for(i = 0; i < m->n_cl; i++){
        cls[i] = (cl_mem*)malloc(sizeof(cl_mem)*(12+m->cls[i]->n_kernels*4));
        load_on_gpu_cl_layer(&cls[i],m->cls[i],ctx);
    }
    
    for(i = 0; i < m->n_fcl; i++){
        fcls[i] = (cl_mem*)malloc(sizeof(cl_mem)*16);
        load_on_gpu_fcl_layer(&fcls[i],m->fcls[i],ctx);
    }
    
    gm->m = copy_model(m);
    gm->rls = rls;
    gm->cls = cls;
    gm->fcls = fcls;
    
    return gm;
}


/* This function save all the vectors of a cl layer in  cl_mem objects already allocated
 * 
 * Inputs:
 * 
 *             @ cl_mem** cls:= the cl_mem objects where will be stored the cl vectors
 *             @ cl c:= the convolutional layer that must be stored on the context
 *             @ cl_context ctx:= the context of the gpu
 * 
 * */
int load_on_gpu_cl_layer(cl_mem** cls, cl* c,cl_context ctx){
    int i,j,ret;
    j = 0;
    
    for(i = 0; i < c->n_kernels; i++){
        (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->channels*c->kernel_cols*c->kernel_rows*sizeof(float),c->kernels[i],&ret);
        if(ret != CL_SUCCESS){
            fprintf(stderr,"Error: loading on context some convolutional vectors\n");
            exit(1);
        }
        j++;
        
        (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->channels*c->kernel_cols*c->kernel_rows*sizeof(float),c->d_kernels[i],&ret);
        if(ret != CL_SUCCESS){
            fprintf(stderr,"Error: loading on context some convolutional vectors\n");
            exit(1);
        }
        j++;
        
        (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->channels*c->kernel_cols*c->kernel_rows*sizeof(float),c->d1_kernels[i],&ret);
        if(ret != CL_SUCCESS){
            fprintf(stderr,"Error: loading on context some convolutional vectors\n");
            exit(1);
        }
        j++;
        
        (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->channels*c->kernel_cols*c->kernel_rows*sizeof(float),c->d2_kernels[i],&ret);
        if(ret != CL_SUCCESS){
            fprintf(stderr,"Error: loading on context some convolutional vectors\n");
            exit(1);
        }
        j++;
        
    }
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*sizeof(float),c->biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*sizeof(float),c->d_biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*sizeof(float),c->d1_biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*sizeof(float),c->d2_biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows1*c->cols1*sizeof(float),c->pre_activation,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows1*c->cols1*sizeof(float),c->post_activation,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows1*c->cols1*sizeof(float),c->post_normalization,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows2*c->cols2*sizeof(float),c->post_pooling,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows1*c->cols1*sizeof(float),c->temp,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows1*c->cols1*sizeof(float),c->temp2,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->n_kernels*c->rows1*c->cols1*sizeof(float),c->temp3,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    (*cls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,c->input_rows*c->channels*c->input_cols*sizeof(float),c->error2,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some convolutional vectors\n");
        exit(1);
    }
    j++;
    
    return j;
    
    
}

/* This function save all the vectors of a rl layer in  cl_mem objects already allocated
 * 
 * Inputs:
 * 
 *             @ cl_mem** rls:= the cl_mem objects where will be stored the rl vectors
 *             @ rl r:= the residual layer that must be stored on the context
 *             @ cl_context ctx:= the context of the gpu
 * 
 * */
int load_on_gpu_rl_layer(cl_mem** rls, rl* r, cl_context ctx){
    int i,j,ret;
    j = 0;
    (*rls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,r->channels*r->input_rows*r->input_cols*sizeof(float),r->input,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some residual vectors\n");
        exit(1);
    }
    j++;
    
    j+= load_on_gpu_cl_layer(&rls[j],r->cl_output,ctx);
    
    for(i = 0; i < r->n_cl; i++){
        j += load_on_gpu_cl_layer(&rls[j],r->cls[i],ctx);
    }
    
    return j;
    
}




/* This function save all the vectors of a fcl layer in  cl_mem objects already allocated
 * 
 * Inputs:
 * 
 *             @ cl_mem** fcls:= the cl_mem objects where will be stored the fcl vectors
 *             @ fcl f:= the fully-connected layer that must be stored on the context
 *             @ cl_context ctx:= the context of the gpu
 * 
 * */
int load_on_gpu_fcl_layer(cl_mem** fcls, fcl* f, cl_context ctx){
    int i,j,ret;
    j = 0;
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->input*f->output*sizeof(float),f->weights,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->input*f->output*sizeof(float),f->d_weights,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->input*f->output*sizeof(float),f->d1_weights,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->input*f->output*sizeof(float),f->d2_weights,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->d_biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->d1_biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->d2_biases,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->pre_activation,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->post_activation,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->dropout_mask,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->dropout_temp,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->temp,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->output*sizeof(float),f->temp3,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->input*sizeof(float),f->temp2,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    (*fcls)[j] = clCreateBuffer(ctx,CL_MEM_READ_WRITE,f->input*sizeof(float),f->error2,&ret);
    if(ret != CL_SUCCESS){
        fprintf(stderr,"Error: loading on context some fully-connected vectors\n");
        exit(1);
    }
    j++;
    
    
    return j;
    
}
