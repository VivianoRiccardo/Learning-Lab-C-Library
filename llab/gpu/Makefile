.PHONY: all

all:
	gcc -c convolutional.c -o convolutional.o -O3 -mavx -lm -w
	gcc -c gd.c -o gd.o -O3 -lm -w
	gcc -c fully_connected.c -o fully_connected.o -O3 -mavx -lm -w
	gcc -c layers.c -o layers.o -O3 -mavx -lm -w
	gcc -c math_functions.c -o math_functions.o -O3 -mavx -lm -w
	gcc -c model.c -o model.o -O3 -mavx -lm -w
	gcc -c normalization.c -o normalization.o -O3 -mavx -lm -w
	gcc -c utils.c -o utils.o -O3 -mavx -lm -w
	gcc -c clipping_gradient.c -o clipping_gradient.o -O3 -mavx -lm
	gcc -c gpu_setup.c -o gpu_setup.o -O3 -mavx -lOpenCL -w
	gcc -c gpu_model.c -o gpu_model.o -O3 -mavx -lOpenCL -w
	ar r libllab.a *.o
	rm *.o
	sudo mv llab_cl_files /home
