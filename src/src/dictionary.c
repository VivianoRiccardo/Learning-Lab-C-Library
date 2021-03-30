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


// returns 1 if the array has been already seen in the past, 0 otherwise. Research in time: O(k), k = size

int check_int_array(int* array, mystruct** ms, int size, int index){
    
    if(index >= size)
        return 1;
    
    
    if((*ms) == NULL){
        (*ms) = (mystruct*)malloc(sizeof(mystruct));
        (*ms)->brother = NULL;
        (*ms)->son = NULL;
        (*ms)->c = array[index];
        if(index == size-1)
            return 0;
        else
            return check_int_array(array,&((*ms)->son),size,index+1);
    }
    
    else{
        if((*ms)->c == array[index])
            return check_int_array(array,&((*ms)->son),size,index+1);
        else{
            if((*ms)->brother == NULL){
                (*ms)->brother = (mystruct*)malloc(sizeof(mystruct));
                (*ms)->brother->brother = NULL;
                (*ms)->brother->son = NULL;
                (*ms)->brother->c = array[index];
                if(index == size-1)
                    return 0;
                else
                    return check_int_array(array,&((*ms)->brother->son),size,index+1);
            }
            
            else if((*ms)->brother->c != array[index])
                return check_int_array(array,&((*ms)->brother->brother),size,index);
            else
                return check_int_array(array,&((*ms)->brother->son),size,index+1);
                
        }
            
    }
}
     

void free_my_struct(mystruct** ms){
    if((*ms) == NULL)
        return;
    
    free_my_struct(&(*ms)->brother);
    free_my_struct(&(*ms)->son);
    free((*ms));
    return;
}
