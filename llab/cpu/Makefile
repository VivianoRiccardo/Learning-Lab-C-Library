.PHONY: all

all:
	gcc -c convolutional.c -o convolutional.o -O3 -mavx2 -lm -lpthread
	gcc -c gd.c -o gd.o -O3 -mavx2 -lm -lpthread
	gcc -c fully_connected.c -o fully_connected.o -O3 -mavx2 -lm -lpthread
	gcc -c fully_connected_layers.c -o fully_connected_layers.o -O3 -mavx2 -lm -lpthread
	gcc -c convolutional_layers.c -o convolutional_layers.o -O3 -mavx2 -lm -lpthread
	gcc -c residual_layers.c -o residual_layers.o -O3 -mavx2 -lm -lpthread
	gcc -c batch_norm_layers.c -o batch_norm_layers.o -O3 -mavx2 -lm -lpthread
	gcc -c math_functions.c -o math_functions.o -O3 -mavx2 -lm -lpthread
	gcc -c model.c -o model.o -O3 -mavx2 -lm -lpthread
	gcc -c bmodel.c -o bmodel.o -O3 -mavx2 -lm -lpthread
	gcc -c normalization.c -o normalization.o -O3 -mavx2 -lm -lpthread
	gcc -c utils.c -o utils.o -O3 -mavx2 -lm -lpthread
	gcc -c clipping_gradient.c -o clipping_gradient.o -O3 -mavx2 -lm -lpthread
	gcc -c recurrent.c -o recurrent.o -O3 -mavx2 -lm -lpthread
	gcc -c recurrent_layers.c -o recurrent_layers.o -O3 -mavx2 -lm -lpthread
	gcc -c rmodel.c -o rmodel.o -O3 -mavx2 -lm -lpthread
	gcc -c multi_core_model.c -o multi_core_model.o -O3 -mavx2 -lm -lpthread
	gcc -c multi_core_rmodel.c -o multi_core_rmodel.o -O3 -mavx2 -lm -lpthread
	gcc -c vae_model.c -o vae_model.o -O3 -mavx2 -lm -lpthread
	gcc -c multi_core_vae_model.c -o multi_core_vae_model.o -O3 -mavx2 -lm -lpthread
	ar r libllab.a *.o
	rm *.o
