#include "llab.h"

void read_input_output_from_files(float** input, float** output, int batch_size, char** files, char* directory){
	int size = 0;
	char* ksource;
	int i,j;
	
	for(i = 0; i < batch_size; i++){
		char* temp = (char*)malloc(sizeof(char)*1024);
		temp[0] = '\0';
		strcat(temp,directory);
		strcat(temo,files[i]);
		read_file_in_char_vector(&ksource,files[i],&size);
		free(temp);
		for(j = 0; j < size; j++){
			printf("%c",ksource[j]);
		}
		printf("\n");
		exit(0);
	}
}


