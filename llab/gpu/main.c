#include "./llab.h"



int main(){
    
    cl_uint n_platforms = 0, n_devices = 0;
    cl_context ctx;
    cl_program prog;
    
    cl_platform_id* platform_id = get_platform_ids(&n_platforms);
    cl_device_id* device_id = get_device_ids(platform_id[0], &n_devices);
    ctx = get_contex(device_id, n_devices);
    prog = get_program(ctx,device_id,n_devices);
    cl_kernel kernel = get_kernel(prog);
    
}
