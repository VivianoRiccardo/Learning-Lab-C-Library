#include "llab.h"


/*
void gpu_set_up(){
    
    
    
    
}
*/


/* This function returns all the platform available for opencl
 * 
 * Input:
 * 
 *             @ cl_uint* n_platforms:= a number that will be filled with number of platforms available
 * 
 * */
cl_platform_id* get_platform_ids(cl_uint* n_platforms){
    
    int err;
    cl_uint num_platforms;
    cl_platform_id *platform_ids;
    
    err = clGetPlatformIDs(0,NULL,&num_platforms);
    if(err != CL_SUCCESS || !num_platforms){
        fprintf(stderr,"Error: No OpenCL platforms\n");
        exit(1);
    }
    
    platform_ids = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms,platform_ids,NULL);
    (*n_platforms) = num_platforms;
    return platform_ids;
}

/* This function returns the number of gpu of a given platform
 * 
 * Input:
 * 
 *             @ cl_platform_id platform_id:= the platform id passed
 *             @ cl_uint* n_devices:= a number that will be filled with the number of devices of the given platform
 * 
 * */
cl_device_id* get_device_ids(cl_platform_id platform_id, cl_uint* n_devices){
    
    int err;
    cl_device_id* device_ids;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_GPU,0,NULL,&num_devices);
    if(err!=CL_SUCCESS || !num_devices){
        fprintf(stderr,"Error: clGetDeviceIds returned an error\n");
        exit(1);
    }    
    
    device_ids = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
    err = clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_GPU,num_devices,device_ids,NULL);
    
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceIds returned an error\n");
        exit(1);
    }
    (*n_devices) = num_devices;
    return device_ids;
    
}

/* This function retuns the context created for 1 or more devices passed
 * 
 * Inputs:
 * 
 *             @ cl_device_id* device_id:= the device ids passed
 *             @ cl_uint n_devices:= the size of cl_device_id* device_id
 * 
 * */
cl_context get_contex(cl_device_id* device_id, cl_uint n_devices){
    int err;
    cl_context context = NULL;
    context = clCreateContext(NULL,n_devices,device_id,NULL,NULL,&err);
    if(err!=CL_SUCCESS || context == NULL){
        fprintf(stderr,"Error: clCreateContext returned an error\n");
        exit(1);
    }
    
    context;
    
}

/* This functions returns the max global memory size of a given device passed as param
 * 
 * Input:
 * 
 *             @ cl_device_id device_id:= the device
 * 
 * */
cl_ulong get_gpu_global_mem_size(cl_device_id device_id){
    int err;
    cl_ulong d_info;
    
    err = clGetDeviceInfo(device_id,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(d_info),&d_info,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceInfo returned an err\n");
        exit(1);
    }
    
    return d_info;
}



/* This functions returns the max global clock frequency in Mhz of a given device passed as param
 * 
 * Input:
 * 
 *             @ cl_device_id device_id:= the device
 * 
 * */
cl_uint get_gpu_max_clock_frequency(cl_device_id device_id){
    int err;
    cl_uint d_info;
    
    err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(d_info),&d_info,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceInfo returned an err\n");
        exit(1);
    }
    
    return d_info;
}


/* This functions returns the number of work items available of a given device passed as param
 * 
 * Input:
 * 
 *             @ cl_device_id device_id:= the device
 * 
 * */
cl_uint get_gpu_work_items(cl_device_id device_id){
    int err;
    cl_uint d_info;
    
    err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(d_info),&d_info,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceInfo returned an err\n");
        exit(1);
    }
    
    return d_info;
}

/* This function says if the compiler for source code is available or not
 * 
 * Input:
 *             @ cl_device_id device_id
 * 
 * */
int compiler_source_is_available(cl_device_id device_id){
    int err;
    cl_bool d_info;
    
    err = clGetDeviceInfo(device_id,CL_DEVICE_COMPILER_AVAILABLE,sizeof(d_info),&d_info,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceInfo returned an err\n");
        exit(1);
    }
    
    if(d_info == CL_FALSE)
        return 0;
    else
        return 1;
}


/* This functions returns the maximum number of work items per work group of a given device passed as param
 * 
 * Input:
 * 
 *             @ cl_device_id device_id:= the device
 * 
 * */
size_t get_gpu_work_items_per_work_group(cl_device_id device_id){
    int err;
    size_t d_info;
    
    err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(d_info),&d_info,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceInfo returned an err\n");
        exit(1);
    }
    
    return d_info;
}


/* This functions returns the maximum number of work items per each dimension of a given device passed as param
 * 
 * Input:
 * 
 *             @ cl_device_id device_id:= the device
 * 
 * */
cl_uint* get_gpu_work_items_per_dimension(cl_device_id device_id){
    int err;
    cl_uint* d_info = (cl_uint*)malloc(sizeof(cl_uint)*3);
    
    err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(cl_uint)*3,d_info,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clGetDeviceInfo returned an err\n");
        exit(1);
    }
    
    return d_info;
}


/* This function create a command queues per device passed as argument
 * 
 * Inputs:
 * 
 *             @ cl_context ctx:= the context where of the devices see get_context function
 *            @ cl_device_id* device_ids:= the devices
 *             @ cl_uint num_devices:= the size of cl_device_id* device_ids array
 * 
 * */
cl_command_queue* get_queue_from_gpus(cl_context ctx, cl_device_id* device_ids, cl_uint num_devices){
    int err,i;
    cl_command_queue* cmd_queues;
    cmd_queues = (cl_command_queue *)malloc(num_devices*sizeof(cl_command_queue));
    for(i = 0; i < num_devices; i++){
        cmd_queues[i] = clCreateCommandQueue(ctx, device_ids[i], 0, &err);
        if(err!=CL_SUCCESS){
            fprintf(stderr,"Error: clCreateCommandQueue returned an err\n");
            exit(1);
        }
    }
    cmd_queues;
}

/* This function build a program for the gpu
 * 
 * Input:
 *             
 *             @ cl_context ctx:= the context used to create the program
 * 
 * */
cl_program get_program(cl_context ctx, cl_device_id* device_id, cl_uint num_devices){
    char options[] = "-cl-unsafe-math-optimizations -cl-mad-enable";
    cl_program prog;
    int err,temp;
    char *fname = "/usr/lib/llab_cl_files/convolutional.cl";
    size_t kfilesize;
    char *ksource;
    
    read_file_in_char_vector(&ksource,fname,&temp);
    kfilesize = (size_t)temp;
    
    prog = clCreateProgramWithSource(ctx,1,(const char**)&ksource,&kfilesize,&err);
    if(err!= CL_SUCCESS){
        fprintf(stderr,"Error: clCreateProgramWithSource returned an err\n");
        exit(1);
    }
    
    err = clBuildProgram(prog, 0, NULL, options, NULL, NULL);
    
    if(err == CL_BUILD_PROGRAM_FAILURE){
        fprintf(stderr,"Error: clBuildProgram return aned err\n");
        size_t log_size;
        clGetProgramBuildInfo(prog, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(prog, device_id[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        
        // Print the log
        printf("%s\n", log);
        exit(1);
    }
    
    if(err!= CL_SUCCESS){
        fprintf(stderr,"Error: clBuildProgram returned an err\n");
        exit(1);
    }
    
    return prog;
}



/* This functions creates the kernel for the functions passed to the program
 * 
 * Input:
 * 
 *             @ cl_program program:= the program associated
 * 
 * */
cl_kernel get_kernel(cl_program program){
    cl_kernel kernel;
    int err;
    kernel = clCreateKernel(program, "convolutional_back_propagation", &err);
    
    if(err!=CL_SUCCESS){
        fprintf(stderr,"Error: clCreateKernel returned an err\n");
        exit(1);
    }
    
    return kernel;
}

/* This function returns a gpu_model
 * 
 * Inputs:
 * 
 *             @ model* m:= the model that you want to load on the gpu
 *             @ cl_device_id device_id:= the device id of the gpu so that we can check 
 *                                        if the model is to big to be handles by the gpu or not
 *             @ cl_context ctx:= the context of the gpu
 * 
 * */
gpu_model* put_model_on_gpu(model* m,cl_device_id device_id, cl_context ctx){
    int i,j,ret,counter = 0;
    gpu_model* gm;
    if(get_gpu_global_mem_size(device_id) < (cl_ulong)size_of_model(m)){
        fprintf(stderr,"Error: your device can't handle a model with these dimensions\n");
        exit(1);
    }
    
    gm = init_gpu_model(m,ctx);
    
    return gm;
}







