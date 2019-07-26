/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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
            temp3 = derivative_sigmoid(z_i[i]);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*sigmoid(z_i[i])*temp2; //db_c_i
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
            temp3 = derivative_sigmoid(z_i[i]);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*sigmoid(z_i[i])*temp2; //db_c_i
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
            temp3 = derivative_sigmoid(z_i[i]);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*sigmoid(z_i[i])*temp2; //db_c_i
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
                
                //printf("%f  \n",dy[j]);
                if(w_up != NULL){
                    dy[j] +=  df_up[i]*derivative_sigmoid(z_f_up[i])*w_f_up[i*size+j]*dropout_mask[j];
                    dy[j] +=  di_up[i]*derivative_sigmoid(z_i_up[i])*w_i_up[i*size+j]*dropout_mask[j];;
                    dy[j] +=  do_up[i]*derivative_sigmoid(z_o_up[i])*w_o_up[i*size+j]*dropout_mask[j];;
                    dy[j] +=  dc_up[i]*sigmoid(z_i_up[i])*derivative_tanhh(z_c_up[i])*w_c_up[i*size+j]*dropout_mask[j];;
                }
                //printf("%f  \n",dy[j]);
                
            }
        }
        
        for(i = 0; i < size; i++){

        
            
            /* and then we can compute what we computed before*/
            
            do_t[i] = dy[i]*tanhh(c_t[i]);    
            temp = derivative_sigmoid(z_o[i]);
            
            dc_t[i] = dy[i]*sigmoid(z_o[i])*derivative_tanhh(c_t[i]) + dc_plus[i]*sigmoid(z_f_plus[i]);
            temp2 = derivative_tanhh(z_c[i]);
            
            di_t[i] = dc_t[i]*tanhh(z_c[i]);
            temp3 = derivative_sigmoid(z_i[i]);
            
            df_t[i] = dc_t[i]*c_minus[i];
            temp4 = derivative_sigmoid(z_f[i]);
            
            for(j = 0; j < size; j++){
                
                dw_o[i*size+j] += do_t[i]*temp*x_t[j]; //dw_o_ixj
                du_o[i*size+j] += do_t[i]*temp*h_minus[j]; //du_o_ixj
                
                dw_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*x_t[j]; //dw_c_ixj
                du_c[i*size+j] += dc_t[i]*sigmoid(z_i[i])*temp2*h_minus[j]; //du_c_ixj
                
                dw_i[i*size+j] += di_t[i]*temp3*x_t[j]; //dw_i_ixj
                du_i[i*size+j] += di_t[i]*temp3*h_minus[j]; //du_i_ixj
                
                dw_f[i*size+j] += df_t[i]*temp4*x_t[j]; //dw_f_ixj
                du_f[i*size+j] += df_t[i]*temp4*h_minus[j]; //du_f_ixj
            }
            
            db_o[i] += do_t[i]*temp; //db_o_i
            db_c[i] += dc_t[i]*sigmoid(z_i[i])*temp2; //db_c_i
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
