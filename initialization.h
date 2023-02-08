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

#ifndef __INITIALIZATION_H__
#define __INITIALIZATION_H__

float r2();
float generate_from_random_distribution(float lo, float hi);
float drand ();
float random_normal ();
float random_general_gaussian(float mean, float std);
float random_general_gaussian_xavier_init(float n);
float random_general_gaussian_xavier_init2(float n1,float n2);
float random_general_gaussian_kaiming_init(float n);
float signed_r2(float n);
float signed_kaiming_constant(float n);

#endif
