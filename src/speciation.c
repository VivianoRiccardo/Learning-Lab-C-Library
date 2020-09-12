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

float compute_species_distance(genome* g1, genome* g2, int global_inn_numb_connections){
    connection** c1;
    connection** c2;
    int i,j,n1,n2, max1 = 0, max2 = 0, max_n;
    float excess = 0, disjoint = 0, matching = 0,m = 0,temp,tempp,temppp;
    int* array1 = (int*)calloc(global_inn_numb_connections,sizeof(int));
    int* array2 = (int*)calloc(global_inn_numb_connections,sizeof(int));
    float v1 = 1, v2 = 1, v3 = 0.4;
    
    c1 = get_connections(g1,global_inn_numb_connections);
    c2 = get_connections(g2,global_inn_numb_connections);
    
    n1 = get_numb_connections(g1,global_inn_numb_connections);
    n2 = get_numb_connections(g2,global_inn_numb_connections);
    
    if(n1 > n2)
        max_n = n1;
    else
        max_n = n2;
    
    if(max_n < 20)
        max_n = 1;
    
        
    for(i = 0; i < n1; i++){
        array1[c1[i]->innovation_number-1] = 1;
        if(c1[i]->innovation_number > max1)
            max1 = c1[i]->innovation_number;
    }
    
    for(i = 0; i < n2; i++){
        array2[c2[i]->innovation_number-1] = 1;
        if(c2[i]->innovation_number > max2)
            max2 = c2[i]->innovation_number;
    }
    
    if(max2 > max1){
        for(i = 0; i < n2; i++){
            if(!array1[c2[i]->innovation_number-1]){
                if(c2[i]->innovation_number > max1)
                    excess++;
                else
                    disjoint++;
            }
            else{
                matching++;
                for(j = 0; j<n2; j++){
                    if(c2[i]->innovation_number == c1[j]->innovation_number){
                        if(!c2[i]->flag)
                            tempp = 0;
                        else
                            tempp = c2[i]->weight;
                            
                        if(!c1[j]->flag)
                            temppp = 0;
                        else
                            temppp = c1[j]->weight;
                            
                        temp = tempp-temppp;
                        if(temp < 0)
                            temp = -temp;
                        m+=temp;    
                        break;
                    }
                }
            }
        }
    }
    
    else{
        for(i = 0; i < n1; i++){
            if(!array2[c1[i]->innovation_number-1]){
                if(c1[i]->innovation_number > max2)
                    excess++;
                else
                    disjoint++;
            }
            else{
                matching++;
                for(j = 0; j<n2; j++){
                    if(c2[j]->innovation_number == c1[i]->innovation_number){
                        if(!c2[j]->flag)
                            tempp = 0;
                        else
                            tempp = c2[j]->weight;
                            
                        if(!c1[i]->flag)
                            temppp = 0;
                        else
                            temppp = c1[i]->weight;
                            
                        temp = tempp-temppp;
                        if(temp < 0)
                            temp = -temp;
                        m+=temp;    
                        break;
                    }
                }
            }
        }
    }
    
    if(matching!=0)
        m/=matching;
        
    matching = m;
    
    excess*=v1;
    excess/=max_n;
    
    disjoint*=v2;
    disjoint/=max_n;
    
    matching*=v3;
    
    free(c1);
    free(c2);
    free(array1);
    free(array2);
    
    return excess+disjoint+matching;
    
}


species* create_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_thereshold, int* total_species){
    int i,j,count_s = 0,flag;
    species* s = NULL;
    
    
    for(i = 0; i < numb_genomes; i++){
        if(!count_s){
            s = (species*)malloc(sizeof(species));
            s[count_s].rapresentative_genome = copy_genome(g[i]);
            s[count_s].numb_all_other_genomes = 0;
            s[count_s].age = 1;
            s[count_s].all_other_genomes = NULL;
            count_s++;
        }
        
        else{
            flag = 0;
            for(j = 0; j < count_s; j++){
                if(compute_species_distance(s[j].rapresentative_genome,g[i],global_inn_numb_connections) < species_thereshold){
                    flag = 1;
                    break;
                }
            }
            
            if(!flag){
                s = (species*)realloc(s,sizeof(species)*(count_s+1));
                s[count_s].rapresentative_genome = copy_genome(g[i]);
                s[count_s].numb_all_other_genomes = 0;
                s[count_s].age = 1;
                s[count_s].all_other_genomes = NULL;

                count_s++;
            }
        }
        
        
    }
    
    (*total_species) = count_s;
    return s;
    
    
}

void free_species(species* s, int total_species, int global_inn_numb_connections){
    int i,j;
    for(i = 0; i < total_species; i++){
        free_genome(s[i].rapresentative_genome,global_inn_numb_connections);
        for(j = 0; j < s[i].numb_all_other_genomes; j++){
            free_genome(s[i].all_other_genomes[j],global_inn_numb_connections);
        }
        
        free(s[i].all_other_genomes);
    }
    
    free(s);
}

void free_species_except_for_rapresentatives(species* s, int total_species, int global_inn_numb_connections){
    int i,j;
    for(i = 0; i < total_species; i++){
        for(j = 0; j < s[i].numb_all_other_genomes; j++){
            free_genome(s[i].all_other_genomes[j],global_inn_numb_connections);
        }
        
        free(s[i].all_other_genomes);
        s[i].numb_all_other_genomes = 0;
    }
}

species* put_genome_in_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_thereshold, int* total_species, species** s){
    int i,j,k,z,count_s = (*total_species),flag;
    genome** temp_g;
    
    shuffle_genome_set(g,numb_genomes);
    
    for(i = 0; i < numb_genomes; i++){
        if(!count_s){
            (*s) = (species*)malloc(sizeof(species));
            (*s)[count_s].rapresentative_genome = copy_genome(g[i]);
            (*s)[count_s].numb_all_other_genomes = 1;
            (*s)[count_s].age = 1;
            (*s)[count_s].all_other_genomes = (genome**)malloc(sizeof(genome*));
            (*s)[count_s].all_other_genomes[0] = copy_genome(g[i]);
            count_s++;
        }
        
        else{
            flag = 0;
            for(j = 0; j < count_s; j++){
                if(compute_species_distance((*s)[j].rapresentative_genome,g[i],global_inn_numb_connections) < species_thereshold){
                    flag = 1;
                    if(!(*s)[j].numb_all_other_genomes){
                        (*s)[j].all_other_genomes = (genome**)malloc(sizeof(genome*));
                        (*s)[j].all_other_genomes[0] = copy_genome(g[i]);
                    }
                    else{
                        temp_g = (genome**)malloc(sizeof(genome*)*((*s)[j].numb_all_other_genomes+1));
                        for(z = 0; z < (*s)[j].numb_all_other_genomes; z++){
                            temp_g[z] = (*s)[j].all_other_genomes[z];
                        }
                        temp_g[z] = copy_genome(g[i]);
                        free((*s)[j].all_other_genomes);
                        (*s)[j].all_other_genomes = temp_g;
                        
                    }
                    
                    (*s)[j].numb_all_other_genomes++;
                    (*s)[j].age++;
                    
                    break;
                }
            }
            
            if(!flag){
                (*s) = (species*)realloc((*s),sizeof(species)*(count_s+1));
                (*s)[count_s].rapresentative_genome = copy_genome(g[i]);
                (*s)[count_s].numb_all_other_genomes = 1;
                (*s)[count_s].age = 1;
                (*s)[count_s].all_other_genomes = (genome**)malloc(sizeof(genome*));
                (*s)[count_s].all_other_genomes[0] = copy_genome(g[i]);
                count_s++;
            }
        }
        
        
    }
    (*total_species) = count_s;
    return (*s);
    
    
}


int get_oldest_age(species* s, int total_species){
    int i;
    int max = -1;
    for(i = 0; i < total_species; i++){
        if(s[i].numb_all_other_genomes > 0)
            if(s[i].age > max)
                max = s[i].age;
        
    }
    
    return max;
}

