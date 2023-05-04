#include "llab.h"

struct_conn_handler* struct_handler(int id, int struct_type_flag, int error_flag, int n_inputs, int n_outputs, vector_struct* input, struct_conn_handler** inputs, struct_conn_handler** outputs, vector_struct* output, float lambda, float huber1, float huber2, float* alpha, model* m, rmodel* r, transformer_encoder* e, vector_struct* v, scaled_l2_norm* l2){ 
    
    if(n_inputs == 0 && inputs != NULL){
        fprintf(stderr,"Error: you have put that there are no inputs, but still you have some previous struct!\n");
        exit(1);
    }
    if(n_outputs == 0 && outputs != NULL){
        fprintf(stderr,"Error: you have put that there are no outputs, but still you have some next struct!\n");
        exit(1);
    }
    
    if(n_inputs && inputs == NULL){
        fprintf(stderr,"Error: you have put that there are inputs, but still you have no previous struct!\n");
        exit(1);
    }
    if(n_outputs && outputs == NULL){
        fprintf(stderr,"Error: you have put that there are outputs, but still you have no some next struct!\n");
        exit(1);
    }
    
    struct_conn_handler* s = (struct_conn_handler*)malloc(sizeof(struct_conn_handler));
    s->struct_type_flag = struct_type_flag;
    s->error_flag = error_flag;
    s->n_inputs = n_inputs;
    s->input = input;
    s->inputs = inputs;
    s->outputs = outputs;
    s->output = output;
    s->lambda = lambda;
    s->huber1 = huber1;
    s->huber2 = huber2;
    s->alpha = alpha;
    s->m = m;
    s->r = r;
    s->e = e;
    s->v = v;
    s->l2 = l2;
    s->visited = 0;
    s->depth = 0;
    s->id = id;
    return s;
}
 
void free_struct_handler(struct_conn_handler* s){
    free(s);
    return;
}

int there_are_no_cycles(struct_conn_handler* s, int depth){
    int i,ret=1;
    s->visited = 1;
    s->depth = depth;
    for(i = 0; i < s->n_inputs; i++){
        if (!s->inputs[i]->visited){
            ret = there_are_no_cycles(s->inputs[i], depth-1);
            if(!ret)
                return ret;
        }
        else
            if(s->inputs[i]->depth > depth)
                return 0;
    }
    
    for(i = 0; i < s->n_outputs; i++){
        if (!s->outputs[i]->visited){
            ret = there_are_no_cycles(s->outputs[i], depth+1);
            if(!ret)
                return ret;
        }
        else
            if(s->outputs[i]->depth < depth)
                return 0;
    }
    
    return ret;    
}
