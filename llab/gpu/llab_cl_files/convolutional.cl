/* This function computes the feed forwad of a feature map using the previous convolutional layer
 * 
 * Input:
 *             @ float* input:= a tensor of input of 3 dimensions: channels, rows and cols
 *                              number of feature maps of this layer = number of channels
 *                              dimensions: channels*input_i*input_j
 *             @ float* kernel:= is a tensor of weights used to compute the feature map using the previous convolutional layer
 *                               dimensions: channels*kernel_i*kernel_j
 *             @ int input_i := the number of rows of each feature map of the previous layer (input)
 *             @ int input_j:= the number of columns of each feature map of the previous layer (input)
 *             @ int kernel_i:= the number of rows of each channel of the kernel
 *             @ int kernel_j:= the number of columns of each channel of the kernel
 *             @ float bias:= the bias of the feature map of the current layer
 *             @ int channels:= the depth of the input and the kernel
 *             @ float* output:= the current tensor computed using the input, kernel and bias
 *                               dimensions: n_kernels*(((input_i-kernel_i)/stride + 1 +2*padding)*((input_j-kernel_j)/stride + 1 +2*padding))
 *             @ int stride:= the stride used by the kernel on the feature maps of the inputs
 *             @ int padding:= the optional padding added to the output
 *			   @ int output_depth_offset:= in case the processors are less then the depth*input_rows*input_cols we must shift the computation, otherwise the processor will thing to be the initialone
 *			   @ int output_rows_offset:= same of depth offset
 *			   @ int output_cols_offset:= same of depth offset
 *			   @ int n_kernels:= we are computing now the entire tensor so we need the n_kernels
 *			   
 * */


__kernel void convolutional_feed_forward ( __local float* input, __local float* kernels, __private int input_i, __private int input_j, __private int kernel_i, __private int kernel_j, __private float bias, __private int channels, __global float* output, __private int stride, __private int padding, __private int output_depth_offset, __private int output_rows_offset, __private int output_cols_offset, __private int n_kernels) {
    int oi,oj,i,j,c,n;
    int output_i = (input_i-kernel_i)/stride + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride + 1 + 2*padding;
    int rows,cols,depth;
    depth = get_global_id(2)+output_depth_offset;
    rows = get_global_id(1)+output_rows_offset;
    cols = get_global_id(0)+output_cols_offset;
    if(depth >= 0 && depth < n_kernels && rows >= padding && rows < output_i-padding && cols < output_j-padding && cols >= padding){
    
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

__kernel void convolutional_back_propagation ( __local float* input, __local float* kernels, __private int input_i, __private int input_j, __private int kernel_i, __private int kernel_j, __private float bias, __private int channels, __local float* output_error,__global float* input_error,__global float* kernel_error,__global float* bias_error, __private int stride, __private int padding, __private int output_depth_offset, __private int output_rows_offset, __private int output_cols_offset, __private int n_kernels) {
    int oi,oj,i,j,c,n;
    int output_i = (input_i-kernel_i)/stride + 1 + 2*padding;
    int output_j = (input_j-kernel_j)/stride + 1 + 2*padding;
    int rows,cols,depth;
    depth = get_global_id(2)+output_depth_offset;
    rows = get_global_id(1)+output_rows_offset;
    cols = get_global_id(0)+output_cols_offset;
    if(depth >= 0 && depth < n_kernels && rows >= padding && rows < output_i-padding && cols < output_j-padding && cols >= padding){
    
            for(c = 0; c < channels; c++){
                for(i = 0; i < kernel_i; i++){
                    for(j = 0; j < kernel_j; j++){
                        kernel_error[c*kernel_i*kernel_j + i*kernel_j + j] += output_error[depth*output_i*output_j+rows*output_j+cols]*input[c*input_i*input_j + i*input_j + j+(cols-padding)*stride+(rows-padding)*stride*input_j];
                    }
                }
            }
            (*bias_error) += output_error[depth*output_i*output_j+rows*output_j+cols];
        }       
}


void max_pooling_feed_forward(__local float* input,__global float* output,__private int input_i,__private int input_j,__private int sub_pool_i,__private int sub_pool_j,__private int stride,__private int padding, __private int output_depth_offset, __private int output_rows_offset, __private int output_cols_offset, __private int n_kernels){
    int i,j,k1,k2;
    int output_i = (input_i-sub_pool_i)/stride + 1 + 2*padding;
    int output_j = (input_j-sub_pool_j)/stride + 1 + 2*padding;
    float max = -9999;
    
    int rows,cols,depth;
    
    depth = get_global_id(2)+output_depth_offset;
    rows = get_global_id(1)+output_rows_offset;
    cols = get_global_id(0)+output_cols_offset;
    
    if(depth >= 0 && depth < n_kernels && rows >= 0 && rows < output_i-padding && cols < output_j-padding && cols >= 0){
    
		for(i = 0; i < output_i - 2*padding; i++){
			for(j = 0; j < output_j - 2*padding; j++){
				for(k1 = 0; k1 < sub_pool_i; k1++){
					for(k2 = 0; k2 < sub_pool_j; k2++){                   
						if(input[input_j*(i*stride+k1) + j*stride + k2] > max)
							max = input[input_j*(i*stride+k1) + j*stride + k2];
					}
				}
				output[(padding+i)*output_j+padding+j] = max;
				max = -99999;        
			}
		}
    }
    
     
}
