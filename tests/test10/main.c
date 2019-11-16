#include <llab.h>

#define INPUT 2
#define OUTPUT 1
#define SPECIES_THERESHOLD 3
#define INITIAL_POPULATION 100
#define GENERATIONS 600
#define PERCENTAGE_SURVIVORS_PER_SPECIE 0.10
#define CONNECTION_MUTATION_RATE 0.8
#define NEW_CONNECTION_ASSIGMENT_RATE 0.1
#define ADD_CONNECTION_BIG_SPECIE_RATE 0.3
#define ADD_CONNECTION_SMALL_SPECIE_RATE 0.03
#define ADD_NODE_SPECIE_RATE 0.05
#define ACTIVATE_CONNECTION_RATE 0.25//there is activate_connection_rate% that a connetion remain disabled
#define REMOVE_CONNECTION_RATE 0.04//there is remove_connection_rate% that a connection can be removed
#define CHILDREN 1//new offsprings = children*(1+b*3) where b is round_up(mean fitness specie/mean fitness population)
#define CROSSOVER_RATE 0.1 
#define SAVING 10//each <saving> generation the best genomes is saved


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
    
    /*allocation*/
    int i,j,z,k,w,flag,min,max,total_species = 0,count = 0;
    int global_inn_numb_connections,global_inn_numb_nodes, global_inn_numb_nodestotal_species = 0, actual_genomes = INITIAL_POPULATION, n_survivors,temp_gg2_counter = 0,temp_gg3_counter = 0, n_species;
    int* dict_connections;
    int** matrix_nodes;
    int** matrix_connections;
    species* s;
    genome* g;
    genome** gg;
    genome** temp_gg1;
    genome** temp_gg2;
    genome** temp_gg3;
    float a,b,n;
    
    
    /*initialization empty genome*/
    g = init_genome(INPUT,OUTPUT);
    
    /*initializing global params*/
    init_global_params(INPUT,OUTPUT,&global_inn_numb_nodes,&global_inn_numb_connections,&dict_connections,&matrix_nodes,&matrix_connections);

    
    /*filling the gg list with init genome, we allocate a big space for gg.
     * gg is a list filled with genomes for each generation, this number of genomes
     * could vary during the generations*/
    gg = (genome**)malloc(sizeof(genome*)*INITIAL_POPULATION*10000);
    temp_gg2 = (genome**)malloc(sizeof(genome*)*INITIAL_POPULATION*10000);//is used for crossover
    temp_gg3 = (genome**)malloc(sizeof(genome*)*INITIAL_POPULATION*10000);//is used to save the rapresentative genomes for each generation
    
    gg[0] = copy_genome(g);
    free_genome(g,global_inn_numb_connections);

    /*initialize first specie with a single rapresentative genome (the empty genome)*/
    s = create_species(gg,1,global_inn_numb_connections,SPECIES_THERESHOLD,&total_species);
    
    
    for(i = 1; i < INITIAL_POPULATION; i++){
        gg[i] = copy_genome(gg[0]);
        add_random_connection(gg[i],&global_inn_numb_connections,&matrix_connections,&dict_connections);
        
    }
    count+=actual_genomes;

    /* START THE GENERATION ITERATIONS */
    for(k = 0; k < GENERATIONS+1; k++){
        printf(">>>>>>>>>> Generation: %d\n",k);
        printf(">>>>>>>>>> Number genomes: %d\n",actual_genomes);
        /* feedforward of the genomes and computing fitness*/
       compute_fitnesses(gg,actual_genomes,global_inn_numb_nodes,global_inn_numb_connections); //just create this function and compute your fitnesses as you want
        
        //save best genome
        n = -1;
        for(i = 0; i < actual_genomes; i++){
            if(gg[i]->fitness > n){
                j = i;
                n = gg[i]->fitness;
            }
            
        }
        g = copy_genome(gg[j]);

        printf("best fitness for this generation: %f\n",n);
        if(n >= 3.5)
            exit(0);
        
        if(k%SAVING == 0 || k == GENERATIONS)
            save_genome(gg[j],global_inn_numb_connections,k+1);
        
        if(k == GENERATIONS)
        break;
    

        /*speciation*/
        s = put_genome_in_species(gg,actual_genomes,global_inn_numb_connections,SPECIES_THERESHOLD,&total_species,&s);
        
        
        /* we copied the genomes in species, now deallocate the genomes in gg */
        for(i = 0; i < actual_genomes; i++){
            free_genome(gg[i],global_inn_numb_connections);
        }
        
        min = 9999999;
        max = -1;
        
        z = 0;
        for(i = 0; i < total_species; i++){
            if(s[i].numb_all_other_genomes > 0){
                z++;
                if(s[i].numb_all_other_genomes > max)
                    max = s[i].numb_all_other_genomes;
                if(s[i].numb_all_other_genomes < min)
                    min = s[i].numb_all_other_genomes;
            }
        }
        printf("num species: %d\n",z);
        printf("biggest specie: %d\n",max);
        n_species = z;
        a = get_mean_fitness(s,total_species);
        
        actual_genomes = 0;temp_gg2_counter = 0; temp_gg3_counter = 0;

        for(i = 0; i < total_species; i++){
            /*compute mean fitnesses of species*/
            if(s[i].numb_all_other_genomes > 0){
                b = get_mean_specie_fitness(s,i);
                b/=a;
                temp_gg1 = sort_genomes_by_fitness(s[i].all_other_genomes,s[i].numb_all_other_genomes);
                /*if a specie didn't improve its for at least 15 generations we kill that specie except in the case where the number of speicies are few*/
                if(s[i].rapresentative_genome->specie_rip < 15 || n_species < 5){
                    /*b >= 1 means the mean fintess of this specie is above the mean fitness of the population
                     * in that case or in the case in which the best fitness of the specie doesn't improve we incremant the rip counter*/
                    if(s[i].all_other_genomes[0]->fitness <= s[i].rapresentative_genome->fitness)
                        s[i].rapresentative_genome->specie_rip++;
                    else if(s[i].all_other_genomes[0]->fitness > s[i].rapresentative_genome->fitness){
                        s[i].rapresentative_genome->fitness = s[i].all_other_genomes[0]->fitness; 
                        s[i].rapresentative_genome->specie_rip=0;
                    }
                    else
                        s[i].rapresentative_genome->specie_rip=0;
                    b = round_up(b);
                    /*in temp_gg3 we save the rapresentative genome of this specie*/
                        
                    temp_gg3[temp_gg3_counter] = copy_genome(s[i].rapresentative_genome);    
                    /*in temp_gg2 we save the best genome of this specie*/
                    if(b >= 1){
                        if(s[i].numb_all_other_genomes>1){
                            temp_gg2[temp_gg2_counter] = copy_genome(temp_gg1[0]);
                            temp_gg2_counter++;
                            temp_gg2[temp_gg2_counter] = copy_genome(temp_gg1[1]);
                            temp_gg2_counter++;
                        }
                    }
                    
                    temp_gg3_counter++;
                    for(z = 0; z < (CHILDREN*(1+b*3)); z+=round_up(s[i].numb_all_other_genomes*PERCENTAGE_SURVIVORS_PER_SPECIE)){
                        for(w = 0; w < s[i].numb_all_other_genomes; w++){
                            if(w >= round_up(s[i].numb_all_other_genomes*PERCENTAGE_SURVIVORS_PER_SPECIE)){
                                break;
                            }
                            gg[actual_genomes] = copy_genome(temp_gg1[w]);
                            /*mutations*/
                            activate_connections(gg[actual_genomes],global_inn_numb_connections,ACTIVATE_CONNECTION_RATE);
                            connections_mutation(gg[actual_genomes],global_inn_numb_connections, CONNECTION_MUTATION_RATE,NEW_CONNECTION_ASSIGMENT_RATE);
                            while(r2() < REMOVE_CONNECTION_RATE){
                                remove_random_connection(gg[actual_genomes],global_inn_numb_connections);
                            }
                            /*big species*/
                            if(s[i].numb_all_other_genomes >= (min+max)/2){
                                
                                if(r2() < ADD_CONNECTION_BIG_SPECIE_RATE){
                                    add_random_connection(gg[actual_genomes],&global_inn_numb_connections,&matrix_connections,&dict_connections);
                                }
                            }
                            
                            /*small specie*/
                            else{
                                if(r2() < ADD_CONNECTION_SMALL_SPECIE_RATE){
                                    add_random_connection(gg[actual_genomes],&global_inn_numb_connections,&matrix_connections,&dict_connections);
                                }
                            }
                                    
                            if(r2() < ADD_NODE_SPECIE_RATE)
                                split_random_connection(gg[actual_genomes],&global_inn_numb_nodes,&global_inn_numb_connections,&dict_connections,&matrix_nodes,&matrix_connections);
                            
                            
                            
                            actual_genomes++;    
                        }
                    }
                    free(temp_gg1);
                }
                
            }
            
        }
        
        for(i = 0; i < actual_genomes; i++){
            gg[i]->fitness = 0;
        }
                
        //these lines save for the next generations the best genomes of the surviving species too
            //but the tests show that is better keeping disabled these lines
        
        //for(i = 0; i < temp_gg2_counter; i++){
            //gg[actual_genomes] = copy_genome(temp_gg2[i]);
            //actual_genomes++;
        //}
        
        
        
        
        
        free_species(s,total_species,global_inn_numb_connections);
        total_species = 0;
        s = create_species(temp_gg3,temp_gg3_counter,global_inn_numb_connections,SPECIES_THERESHOLD,&total_species);
        
        for(i = 0; i < temp_gg3_counter; i++){
            free_genome(temp_gg3[i],global_inn_numb_connections);
        }
        
        for(i = 0; i < temp_gg2_counter; i++){
            temp_gg3[i] = copy_genome(temp_gg2[i]);
        }
        
        for(i = 0; i < temp_gg2_counter-1; i+=2){
                if(r2() < CROSSOVER_RATE){
                    gg[actual_genomes] = crossover(temp_gg2[i],temp_gg2[i+1],global_inn_numb_connections,global_inn_numb_nodes);
                    gg[actual_genomes]->fitness = 0;
                    actual_genomes++;
                }
                    
            
        }
        
        for(i = 0; i < temp_gg2_counter; i++){
            free_genome(temp_gg2[i],global_inn_numb_connections);
            free_genome(temp_gg3[i],global_inn_numb_connections);
        }
        
        gg[actual_genomes] = copy_genome(g);
        actual_genomes++;
        free_genome(g,global_inn_numb_connections);
        count+=actual_genomes;
        printf(">>>>>>>>>> Total number genomes computed: %d\n",count);
        
        
        
        
        
    }

    
    /*deallocation*/
    
    free_species(s,total_species,global_inn_numb_connections);
    free(temp_gg2);
    free(temp_gg3);
    
    for(i = 0; i < actual_genomes; i++){
        free_genome(gg[i],global_inn_numb_connections);
    }
    for(i = 0; i < global_inn_numb_nodes; i++){
        free(matrix_nodes[i]);
    }
    
    for(i = 0; i < global_inn_numb_connections; i++){
        free(matrix_connections[i]);
    }
    
    free_genome(g,global_inn_numb_connections);
    free(gg);
    free(matrix_nodes);
    free(matrix_connections);
    free(dict_connections);
    
}
