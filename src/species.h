#ifndef __SPECIES_H__
#define __SPECIES_H__
#include "genome.h"


typedef struct species{
    genome* rapresentative_genome;
    genome** all_other_genomes;
    int numb_all_other_genomes;
}species;
    

#endif
