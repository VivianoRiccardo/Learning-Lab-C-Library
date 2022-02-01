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

#ifndef __CLIPPING_GRADIENT_H__
#define __CLIPPING_GRADIENT_H__

void clipping_gradient(model* m, float threshold);
void clip_rls(rl** rls, int n, float threshold,float norm);
void clip_cls(cl** cls, int n, float threshold, float norm);
void clip_fcls(fcl** fcls, int n, float threshold, float norm);
float sum_all_quadratic_derivative_weights_rls(rl** rls, int n);
float sum_all_quadratic_derivative_weights_cls(cl** cls, int n);
float sum_all_quadratic_derivative_weights_fcls(fcl** fcls, int n);
void clip_lstms(lstm** lstms, int n, float threshold, float norm);
float sum_all_quadratic_derivative_weights_lstms(lstm** lstms, int n);
void clipping_gradient_rmodel(rmodel* m, float threshold);
float sum_all_quadratic_derivative_weights_bns(bn** bns, int n);
void clip_bns(bn** bns, int n, float threshold, float norm);
void clipping_gradient_vae_model(vaemodel* m, float threshold);
void general_clipping_gradient(model** m, rmodel** r,transformer** t, transformer_encoder** e, transformer_decoder** d, int n_m, int n_r, int n_t,int n_e, int n_d, float threshold);
float sum_all_quadratic_derivative_weights_scaled_l2_norm(scaled_l2_norm** l, int n);
void clip_scaled_l2(scaled_l2_norm** l, int n, float threshold, float norm);
float sum_all_quadratic_derivative_weights_m(model* m);
void clipping_gradient_transf_encoder(transformer_encoder* t, float threshold);
void clipping_gradient_transf_decoder(transformer_decoder* t, float threshold);
void clipping_gradient_transf(transformer* t, float threshold);
void adaptive_gradient_clipping_lstm(lstm* f ,float threshold, float epsilon);
void adaptive_gradient_clipping_fcl(fcl* f ,float threshold, float epsilon);
void adaptive_gradient_clipping_cl(cl* f ,float threshold, float epsilon);
void adaptive_gradient_clipping_rl(rl* f ,float threshold, float epsilon);
void adaptive_gradient_clipping_model(model* m, float threshold, float epsilon);
void adaptive_gradient_clipping_rmodel(rmodel* r, float threshold, float epsilon);
void adaptive_gradient_clipping_encoder_transformer(transformer_encoder* e, float threshold, float epsilon);
void adaptive_gradient_clipping_decoder_transformer(transformer_decoder* t, float threshold, float epsilon);
void adaptive_gradient_clipping_transformer(transformer* t, float threshold, float epsilon);
void dueling_categorical_dqn_clipping_gradient(dueling_categorical_dqn* dqn, float threshold);
void adaptive_gradient_clipping_dueling_categorical_dqn(dueling_categorical_dqn* dqn, float threshold, float epsilon);

#endif
