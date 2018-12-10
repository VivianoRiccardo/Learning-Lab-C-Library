__kernel void convolutional_feed_forward ( __global float* input, __local float* kernels, __private int input_i, __private int input_j, __private int kernel_i, __private int kernel_j, __private float bias, __private int channels, __global float* output, __private int stride, __private int padding, __private int output_depth_offset, __private int output_rows_offset, __private int output_cols_offset, __private int n_kernels) {
    int oi,oj,i,j,c,n;
    int output_i = (input_i-kernel_i)/stride + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride + 1 + 2*padding;
    int rows,cols,depth;
    depth = get_global_id(2)+output_depth_offset;
    rows = get_global_id(1)+output_rows_offset;
    cols = get_global_id(0)+output_cols_offset;
    if(depth >= 0 && depth < n_kernels && rows >= padding && rows < output_i-padding && cols < output_j-padding){
    
            for(c = 0; c < channels; c++){
                for(i = 0; i < kernel_i; i++){
                    for(j = 0; j < kernel_j; j++){
                        output[depth*output_i*output_j+rows*output_j+cols] += kernels[c*kernel_i*kernel_j + i*kernel_j + j]*input[c*input_i*input_j + i*input_j + j+(cols-padding)*stride+(rows-padding)*stride*input_j];
                    }
                }
            }
            output[depth*output_i*output_j+rows*output_j+cols] += bias;    
        }       
}

