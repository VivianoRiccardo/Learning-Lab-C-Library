#include "llab.h"


neat* init(int max_buffer, int input, int output){
    /*allocation*/
    int i,j,z,k,w,flag,min,max,total_species = 0,count = 0;
    int global_inn_numb_connections,global_inn_numb_nodes, actual_genomes = INITIAL_POPULATION, n_survivors,temp_gg2_counter = 0,temp_gg3_counter = 0, n_species;
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
    gg = (genome**)malloc(sizeof(genome*)*INITIAL_POPULATION*max_buffer);
    temp_gg2 = (genome**)malloc(sizeof(genome*)*INITIAL_POPULATION*max_buffer);//is used for crossover
    temp_gg3 = (genome**)malloc(sizeof(genome*)*INITIAL_POPULATION*max_buffer);//is used to save the rapresentative genomes for each generation
    
    gg[0] = copy_genome(g);
    free_genome(g,global_inn_numb_connections);

    /*initialize first specie with a single rapresentative genome (the empty genome)*/
    s = create_species(gg,1,global_inn_numb_connections,SPECIES_THERESHOLD,&total_species);
    
    
    for(i = 1; i < INITIAL_POPULATION; i++){
        gg[i] = copy_genome(gg[0]);
        add_random_connection(gg[i],&global_inn_numb_connections,&matrix_connections,&dict_connections);
        
    }
    count+=actual_genomes;
    
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
    nes->temp_gg3 = temp_gg3;
    nes->max_population = MAX_POPULATION;
    nes->g = NULL;
    nes->species_threshold = SPECIES_THERESHOLD;
    nes->initial_population = INITIAL_POPULATION;
    nes->generations = GENERATIONS;
    nes->percentage_survivors_per_specie = PERCENTAGE_SURVIVORS_PER_SPECIE;
    nes->connection_mutation_rate = CONNECTION_MUTATION_RATE;
    nes->new_connection_assignment_rate = NEW_CONNECTION_ASSIGNMENT_RATE;
    nes->add_connection_big_specie_rate = ADD_CONNECTION_BIG_SPECIE_RATE;
    nes->add_connection_small_specie_rate = ADD_CONNECTION_SMALL_SPECIE_RATE;
    nes->add_node_specie_rate = ADD_NODE_SPECIE_RATE;
    nes->activate_connection_rate = ACTIVATE_CONNECTION_RATE;
    nes->remove_connection_rate = REMOVE_CONNECTION_RATE;
    nes->children = CHILDREN;
    nes->crossover_rate = CROSSOVER_RATE;
    nes->saving = SAVING;
    nes->limiting_species = LIMITING_SPECIES;
    nes->limiting_threshold = LIMITING_THRESHOLD;
    nes->last_fitness = -1;
    nes->fitness_counter = 0;
    nes->same_fitness_limit = SAME_FITNESS_LIMIT;
    return nes;
}
void neat_generation_run(neat* nes, genome** gg){

    //save best genome
    nes->n = -1;
    for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
        if(gg[nes->i]->fitness > nes->n){
            nes->j = nes->i;
            nes->n = gg[nes->i]->fitness;
        }
        
    }
    
    // looking for same best fitness according to the previous generation
    if(nes->n == nes->last_fitness)
        nes->fitness_counter++;
    else
        nes->fitness_counter = 0;
    nes->last_fitness = nes->n;
    
    free(nes->g);
    nes->g = copy_genome(gg[nes->j]);

    //save best genome of the generation
    if(nes->k%nes->saving == 0 || nes->k == nes->generations)
        save_genome(gg[nes->j],nes->global_inn_numb_connections,nes->k+1);

    if(nes->k == nes->generations)
    return;
    
    // if the population is more then max_population param then we eliminate the weakest genomes
    if(nes->actual_genomes > nes->max_population){
        nes->temp_gg1 = sort_genomes_by_fitness(gg,nes->actual_genomes);
        for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
            if(nes->i < nes->max_population)
                gg[nes->i] = nes->temp_gg1[nes->i];
            else
                free_genome(nes->temp_gg1[nes->i],nes->global_inn_numb_connections);
        }
        free(nes->temp_gg1);
        nes->actual_genomes = nes->max_population;
    }
    
    /*speciation*/
    nes->s = put_genome_in_species(gg,nes->actual_genomes,nes->global_inn_numb_connections,nes->species_threshold,&nes->total_species,&nes->s);


    /* we copied the genomes in species, now deallocate the genomes in gg */
    for(nes->i = 0; nes->i < nes->actual_genomes; nes->i++){
        free_genome(gg[nes->i],nes->global_inn_numb_connections);
    }
    
    // If there is always the same fitness as best fitness in the population for about same_fitness_limit times then we take the 2 best species and eliminate the others
    if(nes->fitness_counter >= nes->same_fitness_limit){
        nes->fitness_counter = 0;
        float max1 = -1;
        float max2 = -1;
        int index1 = -1;
        int index2 = -1;
        nes->actual_genomes = 0;
        
        for(nes->i = 0; nes->i < nes->total_species; nes->i++){
            if(nes->s[nes->i].numb_all_other_genomes > 0){
                if(get_mean_specie_fitness(nes->s,nes->i)> max1){
                    max1 = get_mean_specie_fitness(nes->s,nes->i);
                    index1 = nes->i;
                }
            }
        }
        
        for(nes->i = 0; nes->i < nes->total_species; nes->i++){
            if(nes->s[nes->i].numb_all_other_genomes > 0){
                if(get_mean_specie_fitness(nes->s,nes->i)> max2 && nes->i != index1){
                    max2 = get_mean_specie_fitness(nes->s,nes->i);
                    index2 = nes->i;
                }
            }
        }
        
        genome** temp_gg1 = sort_genomes_by_fitness(nes->s[index1].all_other_genomes,nes->s[index1].numb_all_other_genomes);
        int n = nes->s[index1].numb_all_other_genomes;
        for(nes->i = 0; nes->i < n; nes->i++){
            temp_gg1[nes->i] = copy_genome(temp_gg1[nes->i]);
            temp_gg1[nes->i]->specie_rip = 0;
        }
        
        genome** temp_gg2;
        int n2 = 0;
        if(index2 != -1){
            temp_gg2 = sort_genomes_by_fitness(nes->s[index2].all_other_genomes,nes->s[index2].numb_all_other_genomes);
            n2 += nes->s[index2].numb_all_other_genomes;
            for(nes->i = 0; nes->i < n2; nes->i++){
                temp_gg2[nes->i] = copy_genome(temp_gg2[nes->i]);
                temp_gg2[nes->i]->specie_rip = 0;
            }
        }
        nes->actual_genomes+=n;
        free_species(nes->s,nes->total_species,nes->global_inn_numb_connections);
        nes->total_species = 0;
        nes->s = create_species(temp_gg1,n,nes->global_inn_numb_connections,nes->species_threshold,&nes->total_species);
        nes->s = put_genome_in_species(temp_gg1,n,nes->global_inn_numb_connections,nes->species_threshold,&nes->total_species,&nes->s);
        if(index2 != -1){
            nes->actual_genomes+=n2;
            nes->s = put_genome_in_species(temp_gg2,n2,nes->global_inn_numb_connections,nes->species_threshold,&nes->total_species,&nes->s);
        }
        for(nes->i = 0; nes->i < n; nes->i++){
            free_genome(temp_gg1[nes->i],nes->global_inn_numb_connections);
        }
        free(temp_gg1);
        if(index2 != -1){
            for(nes->i = 0; nes->i < n2; nes->i++){
                free_genome(temp_gg2[nes->i],nes->global_inn_numb_connections);
            }
            free(temp_gg2);
        }
        
    }
    
    // now compute the number of species with at least 1 genome inside, the biggest specie and the avarage size of a specie in the entire population
    nes->max = -1;
    nes->sum = 0;
    nes->z = 0;
    for(nes->i = 0; nes->i < nes->total_species; nes->i++){
        if(nes->s[nes->i].numb_all_other_genomes > 0){
            nes->z++;
            nes->sum+=nes->s[nes->i].numb_all_other_genomes;
            if(nes->s[nes->i].numb_all_other_genomes > nes->max)
                nes->max = nes->s[nes->i].numb_all_other_genomes;
        }
    }

    nes->sum/=(float)nes->z;
    nes->n_species = nes->z;
    // compute the man fitness
    nes->a = get_mean_fitness(nes->s,nes->total_species);
    
    
    nes->actual_genomes = 0;nes->temp_gg2_counter = 0; nes->temp_gg3_counter = 0;

    for(nes->i = 0; nes->i < nes->total_species; nes->i++){
        /*compute mean fitnesses of species*/
        if(nes->s[nes->i].numb_all_other_genomes > 0){
            nes->b = get_mean_specie_fitness(nes->s,nes->i);
            nes->b/=nes->a;
            nes->temp_gg1 = sort_genomes_by_fitness(nes->s[nes->i].all_other_genomes,nes->s[nes->i].numb_all_other_genomes);
            /*if a specie didn't improve its for at least 15 generations we kill that specie except in the case where the number of speicies are few*/
            if(nes->s[nes->i].rapresentative_genome->specie_rip < nes->limiting_species || nes->z < 10){
                /*b >= 1 means the mean fintess of this specie is above the mean fitness of the population
                 * in that case or in the case in which the best fitness of the specie doesn't improve we incremant the rip counter*/
                if(nes->temp_gg1[0]->fitness <= nes->s[nes->i].rapresentative_genome->fitness || nes->b < 1)
                    nes->s[nes->i].rapresentative_genome->specie_rip++;
                else if(nes->temp_gg1[0]->fitness > nes->s[nes->i].rapresentative_genome->fitness){
                    nes->s[nes->i].rapresentative_genome->fitness = nes->temp_gg1[0]->fitness; 
                    nes->s[nes->i].rapresentative_genome->specie_rip=0;
                }
                else
                    nes->s[nes->i].rapresentative_genome->specie_rip=0;
                /*in temp_gg3 we save the rapresentative genome of this specie*/
                    
                nes->temp_gg3[nes->temp_gg3_counter] = copy_genome(nes->s[nes->i].rapresentative_genome);    
                /*in temp_gg2 we save the best genome of this specie*/
                if(nes->b >= 1){
                    if(nes->s[nes->i].numb_all_other_genomes>1){
                        nes->temp_gg2[nes->temp_gg2_counter] = copy_genome(nes->temp_gg1[0]);
                        nes->temp_gg2_counter++;
                        nes->temp_gg2[nes->temp_gg2_counter] = copy_genome(nes->temp_gg1[1]);
                        nes->temp_gg2_counter++;
                    }
                }
                
                nes->temp_gg3_counter++;
                double bb = round_up(nes->b*3.67);
                for(nes->z = 0; nes->z < (nes->children*(1+bb)); nes->z+=round_up(nes->s[nes->i].numb_all_other_genomes*nes->percentage_survivors_per_specie)){
                    for(nes->w = 0; nes->w < nes->s[nes->i].numb_all_other_genomes; nes->w++){
                        if(nes->w >= round_up(nes->s[nes->i].numb_all_other_genomes*nes->percentage_survivors_per_specie)){
                            break;
                        }
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
        
    }
            
    //these lines save for the next generations the best genomes of the surviving species too
        //but the tests show that is better keeping disabled these lines

    //for(nes->i = 0; nes->i < nes->temp_gg2_counter; nes->i++){
        //gg[nes->actual_genomes] = copy_genome(nes->temp_gg2[nes->i]);
        //nes->actual_genomes++;
    //}





    free_species(nes->s,nes->total_species,nes->global_inn_numb_connections);
    nes->total_species = 0;
    nes->s = create_species(nes->temp_gg3,nes->temp_gg3_counter,nes->global_inn_numb_connections,nes->species_threshold,&nes->total_species);

    for(nes->i = 0; nes->i < nes->temp_gg3_counter; nes->i++){
        free_genome(nes->temp_gg3[nes->i],nes->global_inn_numb_connections);
    }

    for(nes->i = 0; nes->i < nes->temp_gg2_counter; nes->i++){
        nes->temp_gg3[nes->i] = copy_genome(nes->temp_gg2[nes->i]);
    }

    for(nes->i = 0; nes->i < nes->temp_gg2_counter-1; nes->i+=2){
            if(r2() < nes->crossover_rate){
                gg[nes->actual_genomes] = crossover(nes->temp_gg2[nes->i],nes->temp_gg2[nes->i+1],nes->global_inn_numb_connections,nes->global_inn_numb_nodes);
                gg[nes->actual_genomes]->fitness = 0;
                nes->actual_genomes++;
            }
                
        
    }

    for(nes->i = 0; nes->i < nes->temp_gg2_counter; nes->i++){
        free_genome(nes->temp_gg2[nes->i],nes->global_inn_numb_connections);
        free_genome(nes->temp_gg3[nes->i],nes->global_inn_numb_connections);
    }

    gg[nes->actual_genomes] = copy_genome(nes->g);
    nes->actual_genomes++;
    free_genome(nes->g,nes->global_inn_numb_connections);
    nes->g = NULL;
    nes->count+=nes->actual_genomes;
}

void free_neat(neat* nes){
    free_species(nes->s,nes->total_species,nes->global_inn_numb_connections);
    free(nes->temp_gg2);
    free(nes->temp_gg3);
    
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
