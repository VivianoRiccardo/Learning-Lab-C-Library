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

#include "genome.h"
#include "species.h"
#include "neat_structure.h"
#include "feed_structure.h"
// Functions defined in neat_utils.c

float modified_sigmoid(float x);
genome* init_genome(int input, int output);
void print_genome(genome* g);
genome* copy_genome(genome* g);
int random_number(int min, int max); // random in number between min and max
void init_global_params(int input, int output, int* global_inn_numb_nodes,int* global_inn_numb_connections, int** dict_connections, int*** matrix_nodes, int*** matrix_connections);
void free_genome(genome* g,int global_inn_numb_connections);
connection** get_connections(genome* g, int global_inn_numb_connections); //connection** c rows = global_inn_numb_connections
int get_numb_connections(genome* g, int global_inn_numb_connections);
int shuffle_node_set(node** m,int n);
float random_float_number(float a);
int shuffle_connection_set(connection** m,int n);
int shuffle_genome_set(genome** m,int n);
int save_genome(genome* g, int global_inn_numb_connections, int numb);
int save_genome_complete(genome* g, int global_inn_numb_connections, int global_inn_numb_nodes, int numb);
genome* load_genome(int global_inn_numb_connections, char* filename);
genome* load_genome_complete(char* filename);
int get_global_innovation_number_connections_from_genome(genome* g);
int get_global_innovation_number_nodes_from_genome(genome* g);
int round_up(float num);
char* get_genome_array(genome* g, int global_inn_numb_connections);
genome* init_genome_from_array(int global_inn_numb_connections, char* g_array);
int get_genome_array_size(genome* g, int global_inn_numb_connections);
void adjust_genome(genome* g);


// Functions defined in mutations.c

void connections_mutation(genome* g, int global_inn_numb_connections, float first_thereshold, float second_thereshold);
int split_random_connection(genome* g,int* global_inn_numb_nodes,int* global_inn_numb_connections, int** dict_connections, int*** matrix_nodes, int*** matrix_connections);
int add_random_connection(genome* g,int* global_inn_numb_connections, int*** matrix_connections, int** dict_connections);
int remove_random_connection(genome* g, int global_inn_numb_connections);
genome* crossover(genome* g, genome* g2, int global_inn_numb_connections,int global_inn_numb_nodes);
int activate_connections(genome* g, int global_inn_numb_connections,float thereshold);
void activate_bias(genome* g);


// Functions defined in feedforward.c

float* feed_forward(genome* g1, float* inputs, int global_inn_numb_nodes, int global_inn_numb_connections);
int ff_reconstruction(genome* g, int** array, node* head, int len, ff** lists,int* size, int* global_j);
int recursive_computation(int** array, node* head, genome* g, connection* c,float* actual_value);


// Functions defined in species.c

float compute_species_distance(genome* g1, genome* g2, int global_inn_numb_connections);
species* create_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_threshold, int* total_species);
void free_species(species* s, int total_species, int global_inn_numb_connections);
species* put_genome_in_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_threshold, int* total_species, species** s);
void free_species_except_for_rapresentatives(species* s, int total_species, int global_inn_numb_connections);
int get_oldest_age(species* s, int total_species);
void delete_species_without_population(species** s, int* total_species, int global_inn_numb_connections);
void update_best_specie_fitnesses(species* s, int total_species);



// Functions defined in fitness.c

float get_mean_fitness(species* s, int n_species, int oldest_age, float age_significance);
float get_mean_specie_fitness(species* s, int i,int oldest_age, float age_significance);
genome** sort_genomes_by_fitness(genome** g, int size);

// Functions defined in neat.c

neat* init(int input, int output, int initial_popoulation, int species_threshold, int max_population,int generations, float percentage_survivors_per_specie, float connection_mutation_rate, float  new_connection_assignment_rate, float add_connection_big_specie_rate, float add_connection_small_specie_rate, float add_node_specie_rate, float activate_connection_rate, float remove_connection_rate, int children, float crossover_rate, int saving, int limiting_species, int limiting_threshold, int same_fitness_limit, int keep_parents, float age_significance);
void neat_generation_run(neat* nes);
void free_neat(neat* nes);
char* get_neat_in_char_vector(neat* nes);
neat* init_from_char(char* neat_c, int input, int output, int initial_population, int species_threshold, int max_population,int generations, float percentage_survivors_per_specie, float connection_mutation_rate, float  new_connection_assignment_rate, float add_connection_big_specie_rate, float add_connection_small_specie_rate, float add_node_specie_rate, float activate_connection_rate, float remove_connection_rate, int children, float crossover_rate, int saving, int limiting_species, int limiting_threshold, int same_fitness_limit, int keep_parents, float age_significance);
void reset_fitnesses(neat* n);
void reset_fitness_ith_genome(neat* nes, int index);
float** feed_forward_iths_genome(neat* nes, float** input, int* indices, int n_genome);
float* feed_forward_ith_genome(neat* nes, float* input, int index);
float get_fitness_of_ith_genome(neat* nes, int index);
void increment_fitness_of_genome_ith(neat* nes, int index, float increment);
int get_global_innovation_number_nodes(neat* nes);
int get_global_innovation_number_connections(neat* nes);
int get_lenght_of_char_neat(neat* nes);
int get_number_of_genomes(neat* nes);
void save_ith_genome(neat* nes, int index, int n);
float best_fitness(neat* nes);
