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

#ifndef __RECURRENT_ENCODER_DECODER_H__
#define __RECURRENT_ENCODER_DECODER_H__

recurrent_enc_dec* recurrent_enc_dec_network(rmodel* encoder, rmodel* decoder);
void free_recurrent_enc_dec(recurrent_enc_dec* r);
recurrent_enc_dec* copy_recurrent_enc_dec(recurrent_enc_dec* r);
void paste_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy);
void slow_paste_recurrent_enc_dec(recurrent_enc_dec* r, recurrent_enc_dec* copy, float tau);
void reset_recurrent_enc_dec(recurrent_enc_dec* r);
void save_recurrent_enc_dec(recurrent_enc_dec* r, int n1, int n2, int n3);
recurrent_enc_dec* load_recurrent_enc_dec(char* file1, char* file2, char* file3);

#endif
