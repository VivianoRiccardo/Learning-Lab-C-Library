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
genome* load_genome(int global_inn_numb_connections);
int round_up(float num);


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
species* create_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_thereshold, int* total_species);
void free_species(species* s, int total_species, int global_inn_numb_connections);
species* put_genome_in_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_thereshold, int* total_species, species** s);
void free_species_except_for_rapresentatives(species* s, int total_species, int global_inn_numb_connections);
int get_oldest_age(species* s, int total_species);

// Functions defined in fitness.c

float get_mean_fitness(species* s, int n_species, int oldest_age, float age_significance);
float get_mean_specie_fitness(species* s, int i,int oldest_age, float age_significance);
genome** sort_genomes_by_fitness(genome** g, int size);

// Functions defined in neat.c

neat* init(int max_buffer, int input, int output);
void neat_generation_run(neat* nes, genome** gg);
void free_neat(neat* nes);
