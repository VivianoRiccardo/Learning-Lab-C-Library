.PHONY: all

all:
	gcc -c convolutional.c -o convolutional.o -O3 -mavx -lm
	gcc -c gd.c -o gd.o -O3 -mavx -lm
	gcc -c fully_connected.c -o fully_connected.o -O3 -mavx -lm
	gcc -c layers.c -o layers.o -O3 -mavx -lm
	gcc -c math_functions.c -o math_functions.o -O3 -mavx -lm
	gcc -c model.c -o model.o -O3 -mavx -lm
	gcc -c normalization.c -o normalization.o -O3 -mavx -lm
	gcc -c utils.c -o utils.o -O3 -mavx -lm
	ar r libllab.a convolutional.o gd.o fully_connected.o layers.o math_functions.o model.o normalization.o utils.o
	rm *.o
