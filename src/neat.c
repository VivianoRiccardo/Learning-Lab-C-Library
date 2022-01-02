#include "llab.h"


neat* init(int input, int output, int initial_population, int species_threshold, int max_population,int generations, float percentage_survivors_per_specie, float connection_mutation_rate, float  new_connection_assignment_rate, float add_connection_big_specie_rate, float add_connection_small_specie_rate, float add_node_specie_rate, float activate_connection_rate, float remove_connection_rate, int children, float crossover_rate, int saving, int limiting_species, int limiting_threshold, int same_fitness_limit, int keep_parents, float age_significance){

    
    if(input <= 0 || output <= 0){
        fprintf(stderr,"Error: the inputs must be >= 1 ad same of the outputs!\n");
        exit(1);
    }
    
    if(initial_population <= 0){
        fprintf(stderr,"Error: the initial population must be >= 1\n");
        exit(1);
    }
    
    if(species_threshold < 0){
        fprintf(stderr,"Error: the specie threshold can't be < 0\n");
        exit(1);
    }
    
    if(max_population < initial_population || max_population < 1){
        fprintf(stderr,"Error: max_population must be >= initial_population\n");
        exit(1);
    }
    
    if(generations <= 0){
        fprintf(stderr,"Error: generations must be >= 1\n");
        exit(1);
    }
    
    if(percentage_survivors_per_specie > 1 || percentage_survivors_per_specie <= 0){
        fprintf(stderr,"Error: the percentace_survivors_per_specie must be in (0,1]\n");
        exit(1);
    }
    
    if(connection_mutation_rate > 1 || connection_mutation_rate <= 0){
        fprintf(stderr,"Error: the connection_mutation_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(new_connection_assignment_rate > 1 || new_connection_assignment_rate <= 0){
        fprintf(stderr,"Error: the new_connection_assignment_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(add_connection_big_specie_rate > 1 || add_connection_big_specie_rate <= 0){
        fprintf(stderr,"Error: the add_connection_big_specie_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(add_connection_small_specie_rate > 1 || add_connection_small_specie_rate <= 0){
        fprintf(stderr,"Error: the add_connection_small_specie_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(add_node_specie_rate > 1 || add_node_specie_rate <= 0){
        fprintf(stderr,"Error: the add_node_specie_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(activate_connection_rate > 1 || activate_connection_rate <= 0){
        fprintf(stderr,"Error: the activate_connection_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(remove_connection_rate > 1 || remove_connection_rate <= 0){
        fprintf(stderr,"Error: the remove_connection_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(crossover_rate > 1 || crossover_rate <= 0){
        fprintf(stderr,"Error: the crossover_rate must be in (0,1]\n");
        exit(1);
    }
    
    
    if(children <= 0){
        fprintf(stderr,"Error: the children must be in >= 1\n");
        exit(1);
    }
    
    if(saving <= 0){
        fprintf(stderr,"Error: the saving must be in >= 1\n");
        exit(1);
    }
    
    if(limiting_species <= 0){
        fprintf(stderr,"Error: the limiting_species must be in >= 1\n");
        exit(1);
    }
    
    if(same_fitness_limit <= 0){
        fprintf(stderr,"Error: the same_fitness_limit must be in >= 1\n");
        exit(1);
    }
    
    if(limiting_threshold <= 0 || limiting_threshold >= limiting_species){
        fprintf(stderr,"Error: the limiting_threshold must be in >= 1 and limiting_threshold must be < limiting_species\n");
        exit(1);
    }
    
    /*allocation*/
    int i,j,z,k,w,flag,min,max,total_species = 0,count = initial_population;
    int global_inn_numb_connections,global_inn_numb_nodes, actual_genomes = initial_population, n_survivors,temp_gg2_counter = 0,temp_gg3_counter = 0, n_species;
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
    g = init_genome(input,output);
    
    /*initializing global params*/
    init_global_params(input,output,&global_inn_numb_nodes,&global_inn_numb_connections,&dict_connections,&matrix_nodes,&matrix_connections);

    
    /*filling the gg list with init genome, we allocate a big space for gg.
     * gg is a list filled with genomes for each generation, this number of genomes
     * could vary during the generations*/
    gg = (genome**)malloc(sizeof(genome*)*max_population*2);
    temp_gg2 = (genome**)malloc(sizeof(genome*)*max_population*2);//is used for crossover
    
    gg[0] = copy_genome(g);
    free_genome(g,global_inn_numb_connections);
    g = NULL;
    
    
    for(i = 1; i < initial_population; i++){
        gg[i] = copy_genome(gg[0]);
        add_random_connection(gg[i],&global_inn_numb_connections,&matrix_connections,&dict_connections);
    }
    
    
    
    /*initialize first species*/
    s = create_species(gg,initial_population,global_inn_numb_connections,species_threshold,&total_species);
    
    neat* nes = (neat*)malloc(sizeof(neat));
    nes->total_species = total_species;
    nes->count = count;
    nes->actual_genomes = actual_genomes;
    nes->global_inn_numb_connections = global_inn_numb_connections;
    nes->global_inn_numb_nodes= global_inn_numb_nodes;
    nes->matrix_connections = matrix_connections;
    nes->matrix_nodes = matrix_nodes;
    nes->dict_connections = dict_connections;
    nes->gg = gg;
    nes->s = s;
    nes->temp_gg2 = temp_gg2;
    nes->max_population = max_population;
    nes->new_max_pop = max_population;
    nes->g = NULL;
    nes->species_threshold = species_threshold;
    nes->initial_population = initial_population;
    nes->generations = generations;
    nes->percentage_survivors_per_specie = percentage_survivors_per_specie;
    nes->connection_mutation_rate = connection_mutation_rate;
    nes->new_connection_assignment_rate = new_connection_assignment_rate;
    nes->old_conn_rate = new_connection_assignment_rate;
    nes->add_connection_big_specie_rate = add_connection_big_specie_rate;
    nes->add_connection_small_specie_rate = add_connection_small_specie_rate;
    nes->add_node_specie_rate = add_node_specie_rate;
    nes->activate_connection_rate = activate_connection_rate;
    nes->remove_connection_rate = remove_connection_rate;
    nes->children = children;
    nes->crossover_rate = crossover_rate;
    nes->saving = saving;
    nes->limiting_species = limiting_species;
    nes->limiting_threshold = limiting_threshold;
    nes->last_fitness = -1;
    nes->fitness_counter = 0;
    nes->same_fitness_limit = same_fitness_limit;
    nes->keep_parents = 0;
    nes->age_significance = age_significance;
    return nes;
}
void neat_generation_run(neat* nes){
    
    genome** gg = nes->gg;
    
    //save best genome in nes->n its fitness and nes->j its index
    nes->n = -1;
    for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
        if(gg[nes->i]->fitness > nes->n){
            nes->j = nes->i;
            nes->n = gg[nes->i]->fitness;
        }
        
    }
    
    // looking for same best fitness according to the previous generations
    if(nes->n <= nes->last_fitness)
        nes->fitness_counter++;
    else{
        nes->fitness_counter = 0;
        nes->new_connection_assignment_rate = nes->old_conn_rate;
    }
    
    // If there is always the same fitness as best fitness in the population for about same_fitness_limit times then we increase the connection rate
    // because increasing new conncections will amplify the number of possible species (because there will be more weights)
    if(nes->fitness_counter >= nes->same_fitness_limit){
        nes->fitness_counter = 0;
        if(nes->new_connection_assignment_rate < 0.97)
            nes->new_connection_assignment_rate+=0.03;
    }
    // we save the best actual fitness in the last_fitness
    if(nes->n > nes->last_fitness)
    nes->last_fitness = nes->n;
    
    // in nes->g we save the best genome
    free(nes->g);
    nes->g = copy_genome(gg[nes->j]);

    //save best genome of the generation in a file
    if(nes->k%nes->saving == 0 || nes->k == nes->generations)
        save_genome(gg[nes->j],nes->global_inn_numb_connections,nes->k+1);
    
    // if we have reached the number of generations then just end
    if(nes->k == nes->generations)
    return;
    
    // if the population is higher then max_population param then we eliminate the weakest genomes
    if(nes->actual_genomes > nes->new_max_pop){
        nes->temp_gg1 = sort_genomes_by_fitness(gg,nes->actual_genomes);
        for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
            if(nes->i < nes->new_max_pop)
                gg[nes->i] = nes->temp_gg1[nes->i];
            else
                free_genome(nes->temp_gg1[nes->i],nes->global_inn_numb_connections);
        }
        free(nes->temp_gg1);
        nes->actual_genomes = nes->new_max_pop;
    }
    
    /*speciation*/
    nes->s = put_genome_in_species(gg,nes->actual_genomes,nes->global_inn_numb_connections,nes->species_threshold,&nes->total_species,&nes->s);
    
    
    
    /* we copied the genomes in species, now deallocate the genomes in gg */
    for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
        free_genome(gg[nes->i],nes->global_inn_numb_connections);
    }
    
    // delete species that does not have any genome anymore
    delete_species_without_population(&nes->s,&nes->total_species,nes->global_inn_numb_connections);
    
    // we get the oldest specie age
    int oldest_age = get_oldest_age(nes->s,nes->total_species);
    
    // now compute the number of species with at least 1 genome inside, the biggest specie and the avarage size of a specie in the entire population
    nes->max = -1;
    nes->sum = 0;
    nes->z = nes->total_species;
    for(nes->i = 0; nes->i < nes->total_species; nes->i++){
        nes->sum+=nes->s[nes->i].numb_all_other_genomes;
        if(nes->s[nes->i].numb_all_other_genomes > nes->max)
            nes->max = nes->s[nes->i].numb_all_other_genomes;
    }

    nes->sum/=(float)nes->z;// avarage size of a specie
    nes->n_species = nes->z;// number of species useless parameter is same as nes_total_species
    
    // compute the man fitness, age significance of a specie is also important
    nes->a = get_mean_fitness(nes->s,nes->total_species,oldest_age,nes->age_significance);
    
    // some init parameter
    nes->actual_genomes = 0;nes->temp_gg2_counter = 0;
    
    for(nes->i = 0; nes->i < nes->total_species; nes->i++){
        /*compute mean fitnesses of specie*/
        
        nes->b = get_mean_specie_fitness(nes->s,nes->i,oldest_age,nes->age_significance);
        nes->b/=nes->a;//nes->b>=1 is higher then mean fitness among species
        
        //nes->temp_gg1 has the sorted genomes of this specie (temp_gg1[k]->fitness > temp_gg1[k+1]->fitness for each k)
        nes->temp_gg1 = sort_genomes_by_fitness(nes->s[nes->i].all_other_genomes,nes->s[nes->i].numb_all_other_genomes);
        
        // if we have few species (<10) or the current specie is not to kill (specie_rip == nes->limiting_specie) we are going with mutations
        if(nes->s[nes->i].rapresentative_genome->specie_rip < nes->limiting_species || nes->n_species < 10){
            
            // if a specie didn't improve its best fitness we increment the specie_rip
            if((nes->temp_gg1[0]->fitness <= nes->s[nes->i].best_fitness)){
                nes->s[nes->i].specie_rip++;
            }
            
            else{
                nes->s[nes->i].specie_rip = 0;
            }
            
            // now we look at a special case:
            // if there are few species (<10) should we increment the rip parameters of some no well performing species?
            // intuitively: no, because we have already few species, let them survive a little bit more
            if(((nes->n_species < 10 && nes->s[nes->i].rapresentative_genome->specie_rip > nes->limiting_species-nes->limiting_threshold))){
                nes->s[nes->i].specie_rip = nes->limiting_species-nes->limiting_threshold;
                if(r2() < 0.2)nes->s[nes->i].rapresentative_genome->specie_rip--;
            }
            
            /*in temp_gg2 we save the best genome of this specie if the specie is higher than the mean fitness*/
            if(nes->b >= 1){
                nes->temp_gg2[nes->temp_gg2_counter] = copy_genome(nes->temp_gg1[0]);
                nes->temp_gg2_counter++;
            }
            
            // going with mutations
            // nes->b is how well this specie is doing respect to the other species
            // we mutiply it by 3.67
            // then the value will be mutiplied for children
            double bb = round_up(nes->b*3.67);
            
            // we get the mutations from the nes->percentage_survivors_per_specie best genomes
            for(nes->z = 0; nes->z < (nes->children*(bb)); nes->z+=round_up(nes->s[nes->i].numb_all_other_genomes*nes->percentage_survivors_per_specie)){
                for(nes->w = 0; nes->w < nes->s[nes->i].numb_all_other_genomes; nes->w++){
                    if(nes->w >= round_up(nes->s[nes->i].numb_all_other_genomes*nes->percentage_survivors_per_specie) || nes->z+nes->w>=nes->children*(bb)){
                        break;
                    }
                    // we copy the genome
                    gg[nes->actual_genomes] = copy_genome(nes->temp_gg1[nes->w]);
                    /*mutations*/
                    activate_connections(gg[nes->actual_genomes],nes->global_inn_numb_connections,nes->activate_connection_rate);
                    connections_mutation(gg[nes->actual_genomes],nes->global_inn_numb_connections, nes->connection_mutation_rate,nes->new_connection_assignment_rate);
                        
                    /*big species*/
                    if(nes->s[nes->i].numb_all_other_genomes >= nes->sum){
                        
                        if(nes->s[nes->i].rapresentative_genome->specie_rip < nes->limiting_species-nes->limiting_threshold){
                            if(r2() < nes->add_connection_big_specie_rate){
                                add_random_connection(gg[nes->actual_genomes],&nes->global_inn_numb_connections,&nes->matrix_connections,&nes->dict_connections);
                            }
                            
                            else if(r2() < nes->remove_connection_rate){
                                remove_random_connection(gg[nes->actual_genomes],nes->global_inn_numb_connections);
                            }
                        }
                        
                        else{
                            if(r2() < nes->add_connection_big_specie_rate){
                                remove_random_connection(gg[nes->actual_genomes],nes->global_inn_numb_connections);
                            }
                            else if(r2() < nes->remove_connection_rate){
                                add_random_connection(gg[nes->actual_genomes],&nes->global_inn_numb_connections,&nes->matrix_connections,&nes->dict_connections);
                            }
                        }
                    }
                    
                    /*small specie*/
                    else{
                        if(nes->s[nes->i].rapresentative_genome->specie_rip < nes->limiting_species-nes->limiting_threshold){
                            if(r2() < nes->add_connection_small_specie_rate){
                                add_random_connection(gg[nes->actual_genomes],&nes->global_inn_numb_connections,&nes->matrix_connections,&nes->dict_connections);
                            }
                            
                            else if(r2() < nes->remove_connection_rate){
                                remove_random_connection(gg[nes->actual_genomes],nes->global_inn_numb_connections);
                            }
                        }
                        
                        else{
                            if(r2() < nes->add_connection_small_specie_rate){
                                remove_random_connection(gg[nes->actual_genomes],nes->global_inn_numb_connections);
                            }
                            else if(r2() < nes->remove_connection_rate){
                                add_random_connection(gg[nes->actual_genomes],&nes->global_inn_numb_connections,&nes->matrix_connections,&nes->dict_connections);
                            }
                        }
                    }
                            
                    if(r2() < nes->add_node_specie_rate)
                        split_random_connection(gg[nes->actual_genomes],&nes->global_inn_numb_nodes,&nes->global_inn_numb_connections,&nes->dict_connections,&nes->matrix_nodes,&nes->matrix_connections);
                    
                    
                    
                    nes->actual_genomes++;    
                }
            }
        }
        free(nes->temp_gg1);
            
        
        
    }
    
    // update the best fitness of the species
    update_best_specie_fitnesses(nes->s,nes->total_species);
            
    //these lines save for the next generations the best genomes of the surviving species too
    //but the tests show that is better keeping disabled these lines
    if(nes->keep_parents){
        for(nes->i = 0; nes->i < nes->temp_gg2_counter; nes->i++){
            gg[nes->actual_genomes] = copy_genome(nes->temp_gg2[nes->i]);
            nes->actual_genomes++;
        }
    }





    free_species_except_for_rapresentatives(nes->s,nes->total_species,nes->global_inn_numb_connections);
    
    if(nes->temp_gg2_counter > 1){
        shuffle_genome_set(nes->temp_gg2,nes->temp_gg2_counter);
         for(nes->i = 0; nes->i < nes->temp_gg2_counter-1; nes->i+=2){
            if(r2() < nes->crossover_rate){
                gg[nes->actual_genomes] = crossover(nes->temp_gg2[nes->i],nes->temp_gg2[nes->i+1],nes->global_inn_numb_connections,nes->global_inn_numb_nodes);
                gg[nes->actual_genomes]->fitness = 0;
                nes->actual_genomes++;
            } 
         }
        
    }

    for(nes->i = 0; nes->i < nes->temp_gg2_counter; nes->i++){
        free_genome(nes->temp_gg2[nes->i],nes->global_inn_numb_connections);
    }
    
    gg[nes->actual_genomes] = copy_genome(nes->g);
    nes->actual_genomes++;
    free_genome(nes->g,nes->global_inn_numb_connections);
    nes->g = NULL;
    nes->count+=nes->actual_genomes;
    for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
        gg[nes->i]->fitness = 0;
    }
}

void free_neat(neat* nes){
    free_species(nes->s,nes->total_species,nes->global_inn_numb_connections);
    free(nes->temp_gg2);
    
    for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
        free_genome(nes->gg[nes->i],nes->global_inn_numb_connections);
    }
    for(nes->i = 0; nes->i < nes->global_inn_numb_nodes; nes->i++){
        free(nes->matrix_nodes[nes->i]);
    }
    
    for(nes->i = 0; nes->i < nes->global_inn_numb_connections; nes->i++){
        free(nes->matrix_connections[nes->i]);
    }
    if(nes->g!= NULL)
    free_genome(nes->g,nes->global_inn_numb_connections);
    free(nes->gg);
    free(nes->matrix_nodes);
    free(nes->matrix_connections);
    free(nes->dict_connections);
    free(nes);
}


char* get_neat_in_char_vector(neat* nes){
    int i,sum=0,sum_genome = get_genome_array_size(nes->gg[nes->actual_genomes-1],nes->global_inn_numb_connections);
    sum+=2*sizeof(int)+nes->global_inn_numb_connections*2*sizeof(int)+nes->global_inn_numb_connections*sizeof(int)+nes->global_inn_numb_nodes*sizeof(int)*2+sum_genome;
    char* c = (char*)malloc(sum);
    sum = 0;
    memcpy(c+sum,&(nes->global_inn_numb_connections),sizeof(int));
    sum+=sizeof(int);
    
    memcpy(c+sum,&(nes->global_inn_numb_nodes),sizeof(int));
    sum+=sizeof(int);
    
    for(i = 0; i < nes->global_inn_numb_connections; i++){
        memcpy(c+sum,nes->matrix_connections[i],sizeof(int)*2);
        sum+=sizeof(int)*2;
    }
    
    memcpy(c+sum,nes->dict_connections,sizeof(int)*nes->global_inn_numb_connections);
    sum+=sizeof(int)*nes->global_inn_numb_connections;
    
    for(i = 0; i < nes->global_inn_numb_nodes; i++){
        memcpy(c+sum,nes->matrix_nodes[i],sizeof(int)*2);
        sum+=sizeof(int)*2;
    }
    char* cc = get_genome_array(nes->gg[nes->actual_genomes-1],nes->global_inn_numb_connections);
    memcpy(c+sum,cc,sum_genome);
    free(cc);
    return c;
    
}

int get_lenght_of_char_neat(neat* nes){
    int sum_genome = get_genome_array_size(nes->gg[nes->actual_genomes-1],nes->global_inn_numb_connections);
    return 2*sizeof(int)+nes->global_inn_numb_connections*2*sizeof(int)+nes->global_inn_numb_connections*sizeof(int)+nes->global_inn_numb_nodes*sizeof(int)*2+sum_genome;
}

int get_number_of_genomes(neat* nes){
    return nes->actual_genomes;
}


neat* init_from_char(char* neat_c, int input, int output, int initial_population, int species_threshold, int max_population,int generations, float percentage_survivors_per_specie, float connection_mutation_rate, float  new_connection_assignment_rate, float add_connection_big_specie_rate, float add_connection_small_specie_rate, float add_node_specie_rate, float activate_connection_rate, float remove_connection_rate, int children, float crossover_rate, int saving, int limiting_species, int limiting_threshold, int same_fitness_limit, int keep_parents, float age_significance){

    
    if(input <= 0 || output <= 0){
        fprintf(stderr,"Error: the inputs must be >= 1 ad same of the outputs!\n");
        exit(1);
    }
    
    if(initial_population <= 0){
        fprintf(stderr,"Error: the initial population must be >= 1\n");
        exit(1);
    }
    
    if(species_threshold < 0){
        fprintf(stderr,"Error: the specie threshold can't be < 0\n");
        exit(1);
    }
    
    if(max_population < initial_population || max_population < 1){
        fprintf(stderr,"Error: max_population must be >= initial_population\n");
        exit(1);
    }
    
    if(generations <= 0){
        fprintf(stderr,"Error: generations must be >= 1\n");
        exit(1);
    }
    
    if(percentage_survivors_per_specie > 1 || percentage_survivors_per_specie <= 0){
        fprintf(stderr,"Error: the percentace_survivors_per_specie must be in (0,1]\n");
        exit(1);
    }
    
    if(connection_mutation_rate > 1 || connection_mutation_rate <= 0){
        fprintf(stderr,"Error: the connection_mutation_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(new_connection_assignment_rate > 1 || new_connection_assignment_rate <= 0){
        fprintf(stderr,"Error: the new_connection_assignment_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(add_connection_big_specie_rate > 1 || add_connection_big_specie_rate <= 0){
        fprintf(stderr,"Error: the add_connection_big_specie_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(add_connection_small_specie_rate > 1 || add_connection_small_specie_rate <= 0){
        fprintf(stderr,"Error: the add_connection_small_specie_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(add_node_specie_rate > 1 || add_node_specie_rate <= 0){
        fprintf(stderr,"Error: the add_node_specie_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(activate_connection_rate > 1 || activate_connection_rate <= 0){
        fprintf(stderr,"Error: the activate_connection_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(remove_connection_rate > 1 || remove_connection_rate <= 0){
        fprintf(stderr,"Error: the remove_connection_rate must be in (0,1]\n");
        exit(1);
    }
    
    if(crossover_rate > 1 || crossover_rate <= 0){
        fprintf(stderr,"Error: the crossover_rate must be in (0,1]\n");
        exit(1);
    }
    
    
    if(children <= 0){
        fprintf(stderr,"Error: the children must be in >= 1\n");
        exit(1);
    }
    
    if(saving <= 0){
        fprintf(stderr,"Error: the saving must be in >= 1\n");
        exit(1);
    }
    
    if(limiting_species <= 0){
        fprintf(stderr,"Error: the limiting_species must be in >= 1\n");
        exit(1);
    }
    
    if(same_fitness_limit <= 0){
        fprintf(stderr,"Error: the same_fitness_limit must be in >= 1\n");
        exit(1);
    }
    
    if(limiting_threshold <= 0 || limiting_threshold >= limiting_species){
        fprintf(stderr,"Error: the limiting_threshold must be in >= 1 and limiting_threshold must be < limiting_species\n");
        exit(1);
    }
    
    
    
    
    /*allocation*/
    int i,j,z,k,w,flag,min,max,total_species = 0,count = initial_population;
    int global_inn_numb_connections = 0,global_inn_numb_nodes = 0, actual_genomes = initial_population, n_survivors,temp_gg2_counter = 0,temp_gg3_counter = 0, n_species;
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
    
    
    int summ = 0;
    
    memcpy(&global_inn_numb_connections,neat_c+summ,sizeof(int));
    summ+=sizeof(int);
    memcpy(&global_inn_numb_nodes,neat_c+summ,sizeof(int));
    summ+=sizeof(int);
    matrix_connections = (int**)malloc(sizeof(int*)*global_inn_numb_connections);
    dict_connections = (int*)malloc(sizeof(int)*global_inn_numb_connections);
    matrix_nodes = (int**)malloc(sizeof(int*)*global_inn_numb_nodes);
    
    for(i = 0; i < global_inn_numb_connections; i++){
        matrix_connections[i] = (int*)malloc(sizeof(int)*2);
        memcpy(matrix_connections[i],neat_c+summ,sizeof(int)*2);
        summ+=sizeof(int)*2;
    }
    
    memcpy(dict_connections,neat_c+summ,sizeof(int)*global_inn_numb_connections);
    summ+=sizeof(int)*global_inn_numb_connections;
    
    for(i = 0; i < global_inn_numb_nodes; i++){
        matrix_nodes[i] = (int*)malloc(sizeof(int)*2);
        memcpy(matrix_nodes[i],neat_c+summ,sizeof(int)*2);
        summ+=sizeof(int)*2;
    }
    
    
    /*initialization empty genome*/
    g = init_genome_from_array(global_inn_numb_connections,neat_c+summ);
    
    
    /*filling the gg list with init genome, we allocate a big space for gg.
     * gg is a list filled with genomes for each generation, this number of genomes
     * could vary during the generations*/
    gg = (genome**)malloc(sizeof(genome*)*max_population*2);
    temp_gg2 = (genome**)malloc(sizeof(genome*)*max_population*2);//is used for crossover
    
    gg[0] = copy_genome(g);
    free_genome(g,global_inn_numb_connections);
    g = NULL;
    
    
    for(i = 1; i < initial_population; i++){
        gg[i] = copy_genome(gg[0]);
        add_random_connection(gg[i],&global_inn_numb_connections,&matrix_connections,&dict_connections);
    }
    
    
    
    /*initialize first species*/
    s = create_species(gg,initial_population,global_inn_numb_connections,species_threshold,&total_species);
    
    neat* nes = (neat*)malloc(sizeof(neat));
    nes->total_species = total_species;
    nes->count = count;
    nes->actual_genomes = actual_genomes;
    nes->global_inn_numb_connections = global_inn_numb_connections;
    nes->global_inn_numb_nodes= global_inn_numb_nodes;
    nes->matrix_connections = matrix_connections;
    nes->matrix_nodes = matrix_nodes;
    nes->dict_connections = dict_connections;
    nes->gg = gg;
    nes->s = s;
    nes->temp_gg2 = temp_gg2;
    nes->max_population = max_population;
    nes->new_max_pop = max_population;
    nes->g = NULL;
    nes->species_threshold = species_threshold;
    nes->initial_population = initial_population;
    nes->generations = generations;
    nes->percentage_survivors_per_specie = percentage_survivors_per_specie;
    nes->connection_mutation_rate = connection_mutation_rate;
    nes->new_connection_assignment_rate = new_connection_assignment_rate;
    nes->old_conn_rate = new_connection_assignment_rate;
    nes->add_connection_big_specie_rate = add_connection_big_specie_rate;
    nes->add_connection_small_specie_rate = add_connection_small_specie_rate;
    nes->add_node_specie_rate = add_node_specie_rate;
    nes->activate_connection_rate = activate_connection_rate;
    nes->remove_connection_rate = remove_connection_rate;
    nes->children = children;
    nes->crossover_rate = crossover_rate;
    nes->saving = saving;
    nes->limiting_species = limiting_species;
    nes->limiting_threshold = limiting_threshold;
    nes->last_fitness = -1;
    nes->fitness_counter = 0;
    nes->same_fitness_limit = same_fitness_limit;
    nes->keep_parents = keep_parents;
    nes->age_significance = age_significance;
    return nes;
}


void reset_fitnesses(neat* n){
    int i;
    for(i = 0; i < n->actual_genomes; i++){
        n->gg[i]->fitness = 0;
    }
}

float get_fitness_of_ith_genome(neat* nes, int index){
    if (index >= nes->actual_genomes){
        fprintf(stderr,"Error: index out of range!\n");
        exit(1);
    }
    return nes->gg[index]->fitness;
}


float* feed_forward_ith_genome(neat* nes, float* input, int index){
    if (index >= nes->actual_genomes){
        fprintf(stderr,"Error: index out of range!\n");
        exit(1);
    }
    return feed_forward(nes->gg[index],input,nes->global_inn_numb_nodes,nes->global_inn_numb_connections);
}


float** feed_forward_iths_genome(neat* nes, float** input, int* indices, int n_genome){
    int i,j;
    
    if(n_genome > nes->actual_genomes){
        fprintf(stderr,"Error: there are not so many genomes!\n");
        exit(1);
    }
    
    
    for(i = 0; i < n_genome; i++){
        if (indices[i] >= nes->actual_genomes){
            fprintf(stderr,"Error: index out of range!\n");
            exit(1);
        }
    }
    
    genome** g = (genome**)malloc(sizeof(genome*)*n_genome);
    for(i = 0; i < n_genome; i++){
        g[i] = nes->gg[indices[i]];
    }
    
    float** outputs = feed_forward_multi_thread(n_genome,input,g,nes->global_inn_numb_nodes,nes->global_inn_numb_connections);
    free(g);
    return outputs;
}

void reset_fitness_ith_genome(neat* nes, int index){
    if (index >= nes->actual_genomes){
        fprintf(stderr,"Error: index out of range!\n");
        exit(1);
    }
    nes->gg[index]->fitness = 0;
    return;
}

void increment_fitness_of_genome_ith(neat* nes, int index, float increment){
    if (index >= nes->actual_genomes){
        fprintf(stderr,"Error: index out of range!\n");
        exit(1);
    }
    nes->gg[index]->fitness += increment;
    return;
}

int get_global_innovation_number_nodes(neat* nes){
    return nes->global_inn_numb_nodes;
}

int get_global_innovation_number_connections(neat* nes){
    return nes->global_inn_numb_connections;
}


void save_ith_genome(neat* nes, int index, int n){
    if(index <= nes->actual_genomes){
        fprintf(stderr,"Error: index out of range!\n");
        exit(1);
    }
    
    save_genome(nes->gg[index],nes->global_inn_numb_connections,n);
}

float best_fitness(neat* nes){
    return nes->n;
}
