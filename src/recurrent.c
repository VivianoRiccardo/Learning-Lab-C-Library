/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files ((the "LICENSE")), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "llab.h"

/* this function computes the feed forward of an lstm cell. The cell is computed to be with size*size dimensions, different from the m*n dimension (you can pad to match the
 * 2 dimensions)
 * 
 * Input:
 * 
 *             @ float* x:= the input coming from below
 *             @ float* h:= the last hidden state
 *             @ float* c:= the last cell state
 *             @ float* cell_state:= the current cell state
 *             @ float* hidden_state:= the current hidden_state
 *             @ float** w:= the weights w
 *             @ float** u:= the weights u
 *             @ float** b:= the biases b
 *             @ float** z:= the pre_activated outputs
 *             @ int size:= the size of the cell
 * 
 * */

void lstm_ff(float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int size){
    
    int i,j;
    
    float* f_t = (float*)malloc(sizeof(float)*size);
    float* i_t = (float*)malloc(sizeof(float)*size);
    float* o_t = (float*)malloc(sizeof(float)*size);
    float* tanhh_zc = (float*)malloc(sizeof(float)*size);
    
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            z[0][i] += w[0][i*size+j]*x[j] + u[0][i*size+j]*h[j]; //z_f
            z[1][i] += w[1][i*size+j]*x[j] + u[1][i*size+j]*h[j]; //z_i
            z[2][i] += w[2][i*size+j]*x[j] + u[2][i*size+j]*h[j]; //z_o
            z[3][i] += w[3][i*size+j]*x[j] + u[3][i*size+j]*h[j]; //z_c
        }
        
        z[0][i] += b[0][i];
        z[1][i] += b[1][i];
        z[2][i] += b[2][i];
        z[3][i] += b[3][i];
        
        
        
        
        f_t[i] = sigmoid(z[0][i]); //f_t
        i_t[i] = sigmoid(z[1][i]); //i_t
        o_t[i] = sigmoid(z[2][i]); //o_t
        tanhh_zc[i] = tanhh(z[3][i]); //tanhh(z_c)
        
        
        cell_state[i] = tanhh_zc[i]*i_t[i] + c[i]*f_t[i]; /*cell state of output we calculate c is the previous c state*/
        hidden_state[i] = o_t[i]*tanhh(cell_state[i]); /*hidden state of output we calculate*/
        
        

        
    }
    free(f_t);
    free(i_t);
    free(o_t);
    free(tanhh_zc);
}
/* this function computes the feed forward of an lstm cell. The cell is computed to be with size*size dimensions, different from the m*n dimension (you can pad to match the
 * 2 dimensions)
 * 
 * Input:
 * 
 *             @ float* x:= the input coming from below
 *             @ float* h:= the last hidden state
 *             @ float* c:= the last cell state
 *             @ float* cell_state:= the current cell state
 *             @ float* hidden_state:= the current hidden_state
 *             @ float** w:= the weights w
 *             @ float** u:= the weights u
 *             @ float** b:= the biases b
 *             @ float** z:= the pre_activated outputs
 *             @ int size:= the size of the cell
 * 
 * */

void lstm_ff_edge_popup(int** w_active_output_neurons, int** u_active_output_neurons, int** w_indices,int** u_indices, float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int size, float k_percentage){
    
    int i,j, size2 = size*size;
    
    float* f_t = (float*)malloc(sizeof(float)*size);
    float* i_t = (float*)malloc(sizeof(float)*size);
    float* o_t = (float*)malloc(sizeof(float)*size);
    float* tanhh_zc = (float*)malloc(sizeof(float)*size);
    
    for(i = size2-k_percentage*size2; i < size2; i++){
        z[0][(int)(w_indices[0][i]/size)] += w[0][w_indices[0][i]]*x[(w_indices[0][i]%size)]; //z_f
        z[1][(int)(w_indices[1][i]/size)] += w[1][w_indices[1][i]]*x[(w_indices[1][i]%size)]; //z_i
        z[2][(int)(w_indices[2][i]/size)] += w[2][w_indices[2][i]]*x[(w_indices[2][i]%size)]; //z_o
        z[3][(int)(w_indices[3][i]/size)] += w[3][w_indices[3][i]]*x[(w_indices[3][i]%size)]; //z_c
    }
    for(i = size2-k_percentage*size2; i < size2; i++){
        z[0][(int)(u_indices[0][i]/size)] += u[0][u_indices[0][i]]*h[(u_indices[0][i]%size)]; //z_f
        z[1][(int)(u_indices[1][i]/size)] += u[1][u_indices[1][i]]*h[(u_indices[1][i]%size)]; //z_i
        z[2][(int)(u_indices[2][i]/size)] += u[2][u_indices[2][i]]*h[(u_indices[2][i]%size)]; //z_o
        z[3][(int)(u_indices[3][i]/size)] += u[3][u_indices[3][i]]*h[(u_indices[3][i]%size)]; //z_c
    }
    
    for(i = 0; i < size; i++){
        
        if(w_active_output_neurons[0][i] || u_active_output_neurons[0][i])
            f_t[i] = sigmoid(z[0][i]); //f_t
        if(w_active_output_neurons[1][i] || u_active_output_neurons[1][i])
        i_t[i] = sigmoid(z[1][i]); //i_t
        if(w_active_output_neurons[2][i] || u_active_output_neurons[2][i])
        o_t[i] = sigmoid(z[2][i]); //o_t
        if(w_active_output_neurons[3][i] || u_active_output_neurons[3][i])
            tanhh_zc[i] = tanhh(z[3][i]); //tanhh(z_c)
        
        
        cell_state[i] = tanhh_zc[i]*i_t[i] + c[i]*f_t[i]; /*cell state of output we calculate c is the previous c state*/
        hidden_state[i] = o_t[i]*tanhh(cell_state[i]); /*hidden state of output we calculate*/
        
        

        
    }
    free(f_t);
    free(i_t);
    free(o_t);
    free(tanhh_zc);
}


/* This function computes the backpropagation of an lstm cell
 * 
 * Input:
 * 
 *             @ int flag:= 0 if is the last cell in orizontal and vertical, = 1 if is the last cell in orizontal but not in vertical, = 2 in vertical but not in orizontal, = 3 all the others
 *             @ int size:= the size of the cell
 *             @ float** dw:= where must be stored the partial derivatives of w
 *             @ float** du:= where must be stored the partial derivatives of u
 *             @ float** db:= where must be stored the partial derivatives of b
 *             @ float** w:= the weights w
 *             @ float** u:= the weights u
 *             @ float** z:= the pre activeted outputs computed during the training
 *             @ float* dy:= the error coming from above
 *             @ float* x_t:= the input
 *                @ float* c_t:= the current cell state
 *                @ float* h_minus:= the previous hidden state
 *                @ float* c_minus:= the previous cell state
 *                @ float** z_up:= the z coming from up
 * */
 
 /* dparams should be initialized with all 0s*/
 /* dparams should be initialized with all 0s*/
float** lstm_bp(int flag, int size, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus){
    
    /* different cases for:
     * last cell in orizontal and vertical
     * last cells in orizontal but not in vertical
     * last cells in vertical but not in orizontal
     * all the others
     * */
     
     
     
    int i,j;
    
    float temp;
    float temp2;
    float temp3;
    float temp4;
    float temp5;
    
    float* z_f = z[0]; //z_f
    float* z_i = z[1]; //z_i
    float* z_o = z[2]; //z_o
    float* z_c = z[3]; //z_c
    
    float* dw_f = dw[0];
    float* du_f = du[0];
    float* db_f = db[0];
    float* dw_i = dw[1];
    float* du_i = du[1];
    float* db_i = db[1];
    float* dw_o = dw[2];
    float* du_o = du[2];
    float* db_o = db[2];
    float* dw_c = dw[3];
    float* du_c = du[3];
    float* db_c = db[3];
    
    
    float* w_f = w[0];
    float* w_i = w[1];
    float* w_o = w[2];
    float* w_c = w[3];
    
    float* df_up;
    float* di_up;
    float* do_up;
    float* dc_up;
    
    if(dfioc_up!=NULL){
        df_up = dfioc_up[0];
        di_up = dfioc_up[1];
        do_up = dfioc_up[2];
        dc_up = dfioc_up[3];
    }
    
    float* z_f_up;
    float* z_i_up;
    float* z_o_up;
    float* z_c_up;
    
    if(z_up!=NULL){
        z_f_up = z_up[0];
        z_i_up = z_up[1];
        z_o_up = z_up[2];
        z_c_up = z_up[3];
    }
    
    float* w_f_up;
    float* w_i_up;
    float* w_o_up;
    float* w_c_up;
    
    if(w_up!=NULL){
        
        w_f_up = w_up[0];
        w_i_up = w_up[1];
        w_o_up = w_up[2];
        w_c_up = w_up[3];
        
    }
    
    
    float* df_plus;
    float* di_plus;
    float* do_plus;
    float* dc_plus;
    
    if(dfioc_plus!=NULL){
        
        df_plus = dfioc_plus[0];
        di_plus = dfioc_plus[1];
        do_plus = dfioc_plus[2];
        dc_plus = dfioc_plus[3];
        
    }
    
    float* z_f_plus;
    float* z_i_plus;
    float* z_o_plus;
    float* z_c_plus;
    
    if(z_plus!=NULL){
        z_f_plus = z_plus[0];
        z_i_plus = z_plus[1];
        z_o_plus = z_plus[2];
        z_c_plus = z_plus[3];
        
    }
    
    float* u_f = u[0];
    float* u_i = u[1];
    float* u_o = u[2];
    float* u_c = u[3];
    
    float* do_t = (float*)malloc(sizeof(float)*size);
    float* dc_t = (float*)malloc(sizeof(float)*size);
    float* di_t = (float*)malloc(sizeof(float)*size);
    float* df_t = (float*)malloc(sizeof(float)*size);
     
    /*last cell in orizontal and vertical*/ 
    if( flag == 0){
        
        for(i = 0; i < size; i++){
            
            
            do_t[i] = dy[i]*tanhh(c_t[i]);    
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]);
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            temp5 = sigmoid(z_i[i]);
            temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*temp5*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*temp5*temp2; //db_c_i
            db_i[i] += di_t[i]*temp3; //db_i_i
            db_f[i] += df_t[i]*temp4; //db_f_i
            
            //do dc di and df should be given back
        }
    }
    
    /*last cells in orizontal but not in vertical*/
    else if(flag == 1){
        /* we must recalculate y that corresponds to dh*/
        
        for(i = 0; i < size; i++){
            
            if(w_up != NULL){
                for(j = 0; j < size; j++){
                    dy[j] +=  df_up[i]*derivative_sigmoid(z_f_up[i])*w_f_up[i*size+j];
                    dy[j] +=  di_up[i]*derivative_sigmoid(z_i_up[i])*w_i_up[i*size+j];
                    dy[j] +=  do_up[i]*derivative_sigmoid(z_o_up[i])*w_o_up[i*size+j];
                    dy[j] +=  dc_up[i]*sigmoid(z_i_up[i])*derivative_tanhh(z_c_up[i])*w_c_up[i*size+j];
                }
            }
        }
        
        for(i = 0; i < size; i++){
            
        
            
            /* and then we can compute what we computed before*/
            dy[i]*=dropout_mask[i];
            
            do_t[i] = dy[i]*tanhh(c_t[i]);    
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]);
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            temp5 = sigmoid(z_i[i]);
            temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*temp5*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*temp5*temp2; //db_c_i
            db_i[i] += di_t[i]*temp3; //db_i_i
            db_f[i] += df_t[i]*temp4; //db_f_i
        }
    }
    
    /*last cells in vertical but not in orizontal*/
    else if(flag == 2){
        /* we must recalculate y that corresponds to dh*/
        
        for(i = 0; i < size; i++){
            
            for(j = 0; j < size; j++){
                dy[j] +=  df_plus[i]*derivative_sigmoid(z_f_plus[i])*u_f[i*size+j]*dropout_mask_plus[j];
                dy[j] +=  di_plus[i]*derivative_sigmoid(z_i_plus[i])*u_i[i*size+j]*dropout_mask_plus[j];
                dy[j] +=  do_plus[i]*derivative_sigmoid(z_o_plus[i])*u_o[i*size+j]*dropout_mask_plus[j];
                dy[j] +=  dc_plus[i]*sigmoid(z_i_plus[i])*derivative_tanhh(z_c_plus[i])*u_c[i*size+j]*dropout_mask_plus[j];
            }
        }
        
        for(i = 0; i < size; i++){
                
            /* and then we can compute what we computed before*/
            
            do_t[i] = dy[i]*tanhh(c_t[i]);    
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]) + dc_plus[i]*sigmoid(z_f_plus[i]);
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            temp5 = sigmoid(z_i[i]);
            temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*temp5*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*temp5*temp2; //db_c_i
            db_i[i] += di_t[i]*temp3; //db_i_i
            db_f[i] += df_t[i]*temp4; //db_f_i
        }
    }
    
    /*all other cells*/
    else{
        /* we must recalculate y that corresponds to dh*/
        
        
        for(i = 0; i < size; i++){
            
            for(j = 0; j < size; j++){
                dy[j] +=  df_plus[i]*derivative_sigmoid(z_f_plus[i])*u_f[i*size+j]*dropout_mask_plus[j];
                dy[j] +=  di_plus[i]*derivative_sigmoid(z_i_plus[i])*u_i[i*size+j]*dropout_mask_plus[j];
                dy[j] +=  do_plus[i]*derivative_sigmoid(z_o_plus[i])*u_o[i*size+j]*dropout_mask_plus[j];
                dy[j] +=  dc_plus[i]*sigmoid(z_i_plus[i])*derivative_tanhh(z_c_plus[i])*u_c[i*size+j]*dropout_mask_plus[j];
                
                if(w_up != NULL){
                    dy[j] +=  df_up[i]*derivative_sigmoid(z_f_up[i])*w_f_up[i*size+j]*dropout_mask[j];
                    dy[j] +=  di_up[i]*derivative_sigmoid(z_i_up[i])*w_i_up[i*size+j]*dropout_mask[j];
                    dy[j] +=  do_up[i]*derivative_sigmoid(z_o_up[i])*w_o_up[i*size+j]*dropout_mask[j];
                    dy[j] +=  dc_up[i]*sigmoid(z_i_up[i])*derivative_tanhh(z_c_up[i])*w_c_up[i*size+j]*dropout_mask[j];
                }
                
            }
        }
        
        for(i = 0; i < size; i++){

        
            
            /* and then we can compute what we computed before*/
            
            do_t[i] = dy[i]*tanhh(c_t[i]);    
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]) + dc_plus[i]*sigmoid(z_f_plus[i]);
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            temp5 = sigmoid(z_i[i]);
            temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*temp5*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*temp5*temp2; //db_c_i
            db_i[i] += di_t[i]*temp3; //db_i_i
            db_f[i] += df_t[i]*temp4; //db_f_i
        }
    }
    
    float** dfioc_t = (float**)malloc(sizeof(float*)*4);
    
    
    dfioc_t[0] = df_t;
    dfioc_t[1] = di_t;
    dfioc_t[2] = do_t;
    dfioc_t[3] = dc_t;
    
    return dfioc_t;
      
}

/* This function computes the backpropagation of an lstm cell
 * 
 * Input:
 * 
 *             @ int flag:= 0 if is the last cell in orizontal and vertical, = 1 if is the last cell in orizontal but not in vertical, = 2 in vertical but not in orizontal, = 3 all the others
 *             @ int size:= the size of the cell
 *             @ float** dw:= where must be stored the partial derivatives of w
 *             @ float** du:= where must be stored the partial derivatives of u
 *             @ float** db:= where must be stored the partial derivatives of b
 *             @ float** w:= the weights w
 *             @ float** u:= the weights u
 *             @ float** z:= the pre activeted outputs computed during the training
 *             @ float* dy:= the error coming from above
 *             @ float* x_t:= the input
 *                @ float* c_t:= the current cell state
 *                @ float* h_minus:= the previous hidden state
 *                @ float* c_minus:= the previous cell state
 *                @ float** z_up:= the z coming from up
 * */
 
 /* dparams should be initialized with all 0s*/
 /* dparams should be initialized with all 0s*/
float** lstm_bp_edge_popup(int flag, int size, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus, int** w_active_output_neurons, int** u_active_output_neurons, int** w_indices_up, int** u_indices, float** d_w_scores, float** d_u_scores, float k_percentage, int** w_active_output_neurons_up, int** u_active_output_neurons_up){
    
/* different cases for:
     * last cell in orizontal and vertical
     * last cells in orizontal but not in vertical
     * last cells in vertical but not in orizontal
     * all the others
     * */
     
     
     
    int i,j, size2 = size*size*k_percentage, size_2 = size*size;
    
    float temp;
    float temp2;
    float temp3;
    float temp4;
    float temp5;
    
    float* z_f = z[0]; //z_f
    float* z_i = z[1]; //z_i
    float* z_o = z[2]; //z_o
    float* z_c = z[3]; //z_c
    
    
    
    
    float* w_f = w[0];
    float* w_i = w[1];
    float* w_o = w[2];
    float* w_c = w[3];
    
    float* df_up;
    float* di_up;
    float* do_up;
    float* dc_up;
    
    if(dfioc_up!=NULL){
        df_up = dfioc_up[0];
        di_up = dfioc_up[1];
        do_up = dfioc_up[2];
        dc_up = dfioc_up[3];
    }
    
    float* z_f_up;
    float* z_i_up;
    float* z_o_up;
    float* z_c_up;
    
    if(z_up!=NULL){
        z_f_up = z_up[0];
        z_i_up = z_up[1];
        z_o_up = z_up[2];
        z_c_up = z_up[3];
    }
    
    float* w_f_up;
    float* w_i_up;
    float* w_o_up;
    float* w_c_up;
    
    if(w_up!=NULL){
        
        w_f_up = w_up[0];
        w_i_up = w_up[1];
        w_o_up = w_up[2];
        w_c_up = w_up[3];
        
    }
    
    
    float* df_plus;
    float* di_plus;
    float* do_plus;
    float* dc_plus;
    
    if(dfioc_plus!=NULL){
        
        df_plus = dfioc_plus[0];
        di_plus = dfioc_plus[1];
        do_plus = dfioc_plus[2];
        dc_plus = dfioc_plus[3];
        
    }
    
    float* z_f_plus;
    float* z_i_plus;
    float* z_o_plus;
    float* z_c_plus;
    
    if(z_plus!=NULL){
        z_f_plus = z_plus[0];
        z_i_plus = z_plus[1];
        z_o_plus = z_plus[2];
        z_c_plus = z_plus[3];
        
    }
    
    float* u_f = u[0];
    float* u_i = u[1];
    float* u_o = u[2];
    float* u_c = u[3];
    
    float* do_t = (float*)malloc(sizeof(float)*size);
    float* dc_t = (float*)malloc(sizeof(float)*size);
    float* di_t = (float*)malloc(sizeof(float)*size);
    float* df_t = (float*)malloc(sizeof(float)*size);
     
    /*last cell in orizontal and vertical*/ 
    if( flag == 0){
        
        for(i = 0; i < size; i++){
            temp = 0;
            temp2 = 0;
            temp3 = 0;
            temp4 = 0;
            temp5 = 0;
            do_t[i] = dy[i]*tanhh(c_t[i]); 
            if(w_active_output_neurons[2][i])   
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]);
            if(w_active_output_neurons[3][i])
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            if(w_active_output_neurons[1][i]){
                temp5 = sigmoid(z_i[i]);
                temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            }
            
            df_t[i] = dc_t[i]*c_minus[i];
            if(w_active_output_neurons[0][i])
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                d_w_scores[2][i*size+j] += w_o[i*size+j]*do_t[i]*temp*x_t[j]; //dw_o_ixj
                d_u_scores[2][i*size+j] += u_o[i*size+j]*do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                d_w_scores[3][i*size+j] += w_c[i*size+j]*dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                d_u_scores[3][i*size+j] += u_c[i*size+j]*dc_t[i]*temp5*temp2*h_minus[j]; //dw_c_ixj
                
                d_w_scores[1][i*size+j] += w_i[i*size+j]*di_t[i]*temp3*x_t[j]; //dw_i_ixj
                d_u_scores[1][i*size+j] += u_i[i*size+j]*di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                d_w_scores[0][i*size+j] += w_f[i*size+j]*df_t[i]*temp4*x_t[j]; //dw_f_ixj
                d_u_scores[0][i*size+j] += u_f[i*size+j]*df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            
            
            //do dc di and df should be given back
        }
    }
    
    /*last cells in orizontal but not in vertical*/
    else if(flag == 1){
        /* we must recalculate y that corresponds to dh*/
        if(w_up != NULL){
            for(i = size2; i < size_2; i++){
                if(w_active_output_neurons_up[0][(int)(w_indices_up[0][i]/size)]){
                    dy[w_indices_up[0][i]%size] += df_up[(int)(w_indices_up[0][i]/size)]*derivative_sigmoid(z_f_up[(int)(w_indices_up[0][i]/size)])*w_f_up[w_indices_up[0][i]];
                }
                if(w_active_output_neurons_up[1][(int)(w_indices_up[1][i]/size)]){
                    dy[w_indices_up[1][i]%size] += di_up[(int)(w_indices_up[1][i]/size)]*derivative_sigmoid(z_i_up[(int)(w_indices_up[1][i]/size)])*w_i_up[w_indices_up[1][i]];
                }
                if(w_active_output_neurons_up[2][(int)(w_indices_up[2][i]/size)]){
                    dy[w_indices_up[2][i]%size] += do_up[(int)(w_indices_up[2][i]/size)]*derivative_sigmoid(z_o_up[(int)(w_indices_up[2][i]/size)])*w_o_up[w_indices_up[2][i]];
                }
                if((w_active_output_neurons_up[1][(int)(w_indices_up[3][i]/size)] || u_active_output_neurons_up[1][(int)(w_indices_up[3][i]/size)]) && (w_active_output_neurons_up[3][(int)(w_indices_up[3][i]/size)])){
                    dy[w_indices_up[3][i]%size] +=  dc_up[(int)(w_indices_up[3][i]/size)]*sigmoid(z_i_up[(int)(w_indices_up[3][i]/size)])*derivative_tanhh(z_c_up[(int)(w_indices_up[3][i]/size)])*w_c_up[w_indices_up[3][i]];
                }
                
                
            }
        }

        for(i = 0; i < size; i++){
            /* and then we can compute what we computed before*/
            dy[i]*=dropout_mask[i];
            temp = 0;
            temp2 = 0;
            temp3 = 0;
            temp4 = 0;
            temp5 = 0;
            do_t[i] = dy[i]*tanhh(c_t[i]); 
            if(w_active_output_neurons[2][i])   
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]);
            if(w_active_output_neurons[3][i])
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            if(w_active_output_neurons[1][i]){
                temp5 = sigmoid(z_i[i]);
                temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            }
            
            df_t[i] = dc_t[i]*c_minus[i];
            if(w_active_output_neurons[0][i])
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                d_w_scores[2][i*size+j] += w_o[i*size+j]*do_t[i]*temp*x_t[j]; //dw_o_ixj
                d_u_scores[2][i*size+j] += u_o[i*size+j]*do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                d_w_scores[3][i*size+j] += w_c[i*size+j]*dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                d_u_scores[3][i*size+j] += u_c[i*size+j]*dc_t[i]*temp5*temp2*h_minus[j]; //dw_c_ixj
                
                d_w_scores[1][i*size+j] += w_i[i*size+j]*di_t[i]*temp3*x_t[j]; //dw_i_ixj
                d_u_scores[1][i*size+j] += u_i[i*size+j]*di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                d_w_scores[0][i*size+j] += w_f[i*size+j]*df_t[i]*temp4*x_t[j]; //dw_f_ixj
                d_u_scores[0][i*size+j] += u_f[i*size+j]*df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            
            
            //do dc di and df should be given back
        }
    }
    
    /*last cells in vertical but not in orizontal*/
    else if(flag == 2){
        /* we must recalculate y that corresponds to dh*/
        for(i = size2; i < size_2; i++){
            if(u_active_output_neurons[0][(int)(u_indices[0][i]/size)]){
                dy[u_indices[0][i]%size] += df_plus[(int)(u_indices[0][i]/size)]*derivative_sigmoid(z_f_plus[(int)(u_indices[0][i]/size)])*u_f[u_indices[0][i]]*dropout_mask_plus[u_indices[0][i]%size];
            }
            if(u_active_output_neurons[1][(int)(u_indices[1][i]/size)]){
                dy[u_indices[1][i]%size] += di_plus[(int)(u_indices[1][i]/size)]*derivative_sigmoid(z_i_plus[(int)(u_indices[1][i]/size)])*u_i[u_indices[1][i]]*dropout_mask_plus[u_indices[1][i]%size];
            }
            if(u_active_output_neurons[2][(int)(u_indices[2][i]/size)]){
                dy[u_indices[2][i]%size] += do_plus[(int)(u_indices[2][i]/size)]*derivative_sigmoid(z_o_plus[(int)(u_indices[2][i]/size)])*u_o[u_indices[2][i]]*dropout_mask_plus[u_indices[2][i]%size];
            }
            if((u_active_output_neurons[1][(int)(u_indices[3][i]/size)] || w_active_output_neurons[1][(int)(u_indices[3][i]/size)]) && (u_active_output_neurons[3][(int)(u_indices[3][i]/size)])){
                dy[u_indices[3][i]%size] +=  dc_plus[(int)(u_indices[3][i]/size)]*sigmoid(z_i_plus[(int)(u_indices[3][i]/size)])*derivative_tanhh(z_c_plus[(int)(u_indices[3][i]/size)])*u_c[u_indices[3][i]]*dropout_mask_plus[u_indices[3][i]%size];
            }
            
            
        }
        
        
        for(i = 0; i < size; i++){
            /* and then we can compute what we computed before*/
            temp = 0;
            temp2 = 0;
            temp3 = 0;
            temp4 = 0;
            temp5 = 0;
            do_t[i] = dy[i]*tanhh(c_t[i]); 
            if(w_active_output_neurons[2][i])   
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]);
            if(w_active_output_neurons[3][i])
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            if(w_active_output_neurons[1][i]){
                temp5 = sigmoid(z_i[i]);
                temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            }
            
            df_t[i] = dc_t[i]*c_minus[i];
            if(w_active_output_neurons[0][i])
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                d_w_scores[2][i*size+j] += w_o[i*size+j]*do_t[i]*temp*x_t[j]; //dw_o_ixj
                d_u_scores[2][i*size+j] += u_o[i*size+j]*do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                d_w_scores[3][i*size+j] += w_c[i*size+j]*dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                d_u_scores[3][i*size+j] += u_c[i*size+j]*dc_t[i]*temp5*temp2*h_minus[j]; //dw_c_ixj
                
                d_w_scores[1][i*size+j] += w_i[i*size+j]*di_t[i]*temp3*x_t[j]; //dw_i_ixj
                d_u_scores[1][i*size+j] += u_i[i*size+j]*di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                d_w_scores[0][i*size+j] += w_f[i*size+j]*df_t[i]*temp4*x_t[j]; //dw_f_ixj
                d_u_scores[0][i*size+j] += u_f[i*size+j]*df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
           
            
            //do dc di and df should be given back
        }
    }
    
    /*all other cells*/
    else{
        /* we must recalculate y that corresponds to dh*/
        if(w_up != NULL){
            for(i = size2; i < size_2; i++){
                if(w_active_output_neurons_up[0][(int)(w_indices_up[0][i]/size)]){
                    dy[w_indices_up[0][i]%size] += df_up[(int)(w_indices_up[0][i]/size)]*derivative_sigmoid(z_f_up[(int)(w_indices_up[0][i]/size)])*w_f_up[w_indices_up[0][i]]*dropout_mask[w_indices_up[0][i]%size];
                }
                if(w_active_output_neurons_up[1][(int)(w_indices_up[1][i]/size)]){
                    dy[w_indices_up[1][i]%size] += di_up[(int)(w_indices_up[1][i]/size)]*derivative_sigmoid(z_i_up[(int)(w_indices_up[1][i]/size)])*w_i_up[w_indices_up[1][i]]*dropout_mask[w_indices_up[1][i]%size];
                }
                if(w_active_output_neurons_up[2][(int)(w_indices_up[2][i]/size)]){
                    dy[w_indices_up[2][i]%size] += do_up[(int)(w_indices_up[2][i]/size)]*derivative_sigmoid(z_o_up[(int)(w_indices_up[2][i]/size)])*w_o_up[w_indices_up[2][i]]*dropout_mask[w_indices_up[2][i]%size];
                }
                if((w_active_output_neurons_up[1][(int)(w_indices_up[3][i]/size)] || u_active_output_neurons_up[1][(int)(w_indices_up[3][i]/size)]) && (w_active_output_neurons_up[3][(int)(w_indices_up[3][i]/size)])){
                    dy[w_indices_up[3][i]%size] +=  dc_up[(int)(w_indices_up[3][i]/size)]*sigmoid(z_i_up[(int)(w_indices_up[3][i]/size)])*derivative_tanhh(z_c_up[(int)(w_indices_up[3][i]/size)])*w_c_up[w_indices_up[3][i]]*dropout_mask[w_indices_up[3][i]%size];
                }
                
                
            }
        }
        for(i = size2; i < size_2; i++){
            if(u_active_output_neurons[0][(int)(u_indices[0][i]/size)]){
                dy[u_indices[0][i]%size] += df_plus[(int)(u_indices[0][i]/size)]*derivative_sigmoid(z_f_plus[(int)(u_indices[0][i]/size)])*u_f[u_indices[0][i]]*dropout_mask_plus[u_indices[0][i]%size];
            }
            if(u_active_output_neurons[1][(int)(u_indices[1][i]/size)]){
                dy[u_indices[1][i]%size] += di_plus[(int)(u_indices[1][i]/size)]*derivative_sigmoid(z_i_plus[(int)(u_indices[1][i]/size)])*u_i[u_indices[1][i]]*dropout_mask_plus[u_indices[1][i]%size];
            }
            if(u_active_output_neurons[2][(int)(u_indices[2][i]/size)]){
                dy[u_indices[2][i]%size] += do_plus[(int)(u_indices[2][i]/size)]*derivative_sigmoid(z_o_plus[(int)(u_indices[2][i]/size)])*u_o[u_indices[2][i]]*dropout_mask_plus[u_indices[2][i]%size];
            }
            if((u_active_output_neurons[1][(int)(u_indices[3][i]/size)] || w_active_output_neurons[1][(int)(u_indices[3][i]/size)]) && (u_active_output_neurons[3][(int)(u_indices[3][i]/size)])){
                dy[u_indices[3][i]%size] +=  dc_plus[(int)(u_indices[3][i]/size)]*sigmoid(z_i_plus[(int)(u_indices[3][i]/size)])*derivative_tanhh(z_c_plus[(int)(u_indices[3][i]/size)])*u_c[u_indices[3][i]]*dropout_mask_plus[u_indices[3][i]%size];
            }
            
            
        }

        
        for(i = 0; i < size; i++){
            /* and then we can compute what we computed before*/
            temp = 0;
            temp2 = 0;
            temp3 = 0;
            temp4 = 0;
            temp5 = 0;
            do_t[i] = dy[i]*tanhh(c_t[i]); 
            if(w_active_output_neurons[2][i])   
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]);
            if(w_active_output_neurons[3][i])
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            if(w_active_output_neurons[1][i]){
                temp5 = sigmoid(z_i[i]);
                temp3 = derivative_sigmoid_given_the_sigmoid(temp5);
            }
            
            df_t[i] = dc_t[i]*c_minus[i];
            if(w_active_output_neurons[0][i])
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                d_w_scores[2][i*size+j] += w_o[i*size+j]*do_t[i]*temp*x_t[j]; //dw_o_ixj
                d_u_scores[2][i*size+j] += u_o[i*size+j]*do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                d_w_scores[3][i*size+j] += w_c[i*size+j]*dc_t[i]*temp5*temp2*x_t[j]; //dw_c_ixj
                d_u_scores[3][i*size+j] += u_c[i*size+j]*dc_t[i]*temp5*temp2*h_minus[j]; //dw_c_ixj
                
                d_w_scores[1][i*size+j] += w_i[i*size+j]*di_t[i]*temp3*x_t[j]; //dw_i_ixj
                d_u_scores[1][i*size+j] += u_i[i*size+j]*di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                d_w_scores[0][i*size+j] += w_f[i*size+j]*df_t[i]*temp4*x_t[j]; //dw_f_ixj
                d_u_scores[0][i*size+j] += u_f[i*size+j]*df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
           
            
            //do dc di and df should be given back
        }
    }
    
    float** dfioc_t = (float**)malloc(sizeof(float*)*4);
    
    
    dfioc_t[0] = df_t;
    dfioc_t[1] = di_t;
    dfioc_t[2] = do_t;
    dfioc_t[3] = dc_t;
    
    return dfioc_t;
}
