#include <llab.h>

#define INPUT 2
#define OUTPUT 1


void compute_fitnesses(genome** gg,int actual_genomes,int global_inn_numb_nodes,int global_inn_numb_connections){
    int i,j;
    float inputs[2] = {0,0};
    float* output;
    for(i = 0; i < actual_genomes; i++){
        if(gg[i]->fitness == 0){
            inputs[0] = 0;
            inputs[1] = 0;
            output = feed_forward(gg[i],inputs,global_inn_numb_nodes,global_inn_numb_connections);
            gg[i]->fitness += 1-output[0];
            free(output);
            inputs[0] = 1;
            inputs[1] = 0;
            output = feed_forward(gg[i],inputs,global_inn_numb_nodes,global_inn_numb_connections);
            gg[i]->fitness += output[0];
            free(output);
            inputs[0] = 0;
            inputs[1] = 1;
            output = feed_forward(gg[i],inputs,global_inn_numb_nodes,global_inn_numb_connections);
            gg[i]->fitness += output[0];
            free(output);
            inputs[0] = 1;
            inputs[1] = 1;
            output = feed_forward(gg[i],inputs,global_inn_numb_nodes,global_inn_numb_connections);
            gg[i]->fitness += 1-output[0];
            free(output);

        }

    }
}

int main(){
    
    srand(time(NULL));
    neat* nes = init(100000,INPUT,OUTPUT);

    /* START THE GENERATION ITERATIONS */
    for(nes->k = 0; nes->k < GENERATIONS+1; nes->k++){
        printf(">>>>>>>>>> Generation: %d\n",nes->k);
        printf(">>>>>>>>>> Number genomes: %d\n",nes->actual_genomes);
        /* feedforward of the genomes and computing fitness*/
       compute_fitnesses(nes->gg,nes->actual_genomes,nes->global_inn_numb_nodes,nes->global_inn_numb_connections); //just create this function and compute your fitnesses as you want
       neat_generation_run(nes,nes->gg);
       printf("best fitness for this generation: %f\n",nes->n);
       printf("num species: %d\n",nes->z);
       printf("biggest specie: %d\n",nes->max);
       printf(">>>>>>>>>> Total number genomes computed: %d\n",nes->count);
   }

    
    /*deallocation*/
    free_neat(nes);
    
}
