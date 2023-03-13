#include <llab.h>

#define INPUT 2
#define OUTPUT 1


void compute_fitnesses(genome** gg,int actual_genomes,int global_inn_numb_nodes,int global_inn_numb_connections){
    int i,j;
    float inputs[2] = {0,0};
    float* output;
    for(i = 0; i < actual_genomes; i++){
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

int main(){
    
    srand(time(NULL));
    neat* nes = init(INPUT,OUTPUT,INITIAL_POPULATION,SPECIES_THERESHOLD,MAX_POPULATION,GENERATIONS,PERCENTAGE_SURVIVORS_PER_SPECIE,CONNECTION_MUTATION_RATE,NEW_CONNECTION_ASSIGNMENT_RATE,ADD_CONNECTION_BIG_SPECIE_RATE,ADD_CONNECTION_SMALL_SPECIE_RATE,ADD_NODE_SPECIE_RATE,ACTIVATE_CONNECTION_RATE,REMOVE_CONNECTION_RATE,CHILDREN,CROSSOVER_RATE,SAVING,LIMITING_SPECIES,LIMITING_THRESHOLD,SAME_FITNESS_LIMIT,1,AGE_SIGNIFICANCE);
    /* START THE GENERATION ITERATIONS */
    for(nes->k = 0; nes->k < GENERATIONS+1; nes->k++){ 
       /* feedforward of the genomes and computing fitness*/
       compute_fitnesses(nes->gg,nes->actual_genomes,nes->global_inn_numb_nodes,nes->global_inn_numb_connections); //just create this function and compute your fitnesses as you want
       neat_generation_run(nes);
       printf(">>>>>>>>>> Generation: %d\n",nes->k);
       printf(">>>>>>>>>> Number genomes: %d\n",nes->actual_genomes);
       printf("best fitness for this generation: %f\n",nes->n);
       printf("num species: %d\n",nes->n_species);
       printf("biggest specie: %d\n",nes->max);
       printf("Total number genomes computed: %d\n",nes->count);
       if(nes->n >= 3.5) break;
   }
   
   // testing saves and loads and char* genome array and char neat
   genome* load  = load_genome_complete("1.bin");
   free_genome(load, nes->global_inn_numb_connections);
   /*deallocation*/
   
   char* c = get_genome_array(nes->gg[nes->actual_genomes-1],nes->global_inn_numb_connections);
   genome* g = init_genome_from_array(nes->global_inn_numb_connections,c);
   g->fitness = 0;
   compute_fitnesses(&g,1,nes->global_inn_numb_nodes,nes->global_inn_numb_connections);
   
   free_genome(g,nes->global_inn_numb_connections);
   free(c);
   char* n_c = get_neat_in_char_vector(nes);
   neat* nes2 = init_from_char(n_c,INPUT,OUTPUT,INITIAL_POPULATION,SPECIES_THERESHOLD,MAX_POPULATION,GENERATIONS,PERCENTAGE_SURVIVORS_PER_SPECIE,CONNECTION_MUTATION_RATE,NEW_CONNECTION_ASSIGNMENT_RATE,ADD_CONNECTION_BIG_SPECIE_RATE,ADD_CONNECTION_SMALL_SPECIE_RATE,ADD_NODE_SPECIE_RATE,ACTIVATE_CONNECTION_RATE,REMOVE_CONNECTION_RATE,CHILDREN,CROSSOVER_RATE,SAVING,LIMITING_SPECIES,LIMITING_THRESHOLD,SAME_FITNESS_LIMIT,1,AGE_SIGNIFICANCE);
   free_neat(nes);
   free_neat(nes2);
   free(n_c);
}
