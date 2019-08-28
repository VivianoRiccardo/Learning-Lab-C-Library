SRCS = $(wildcard *.c)

PROGS = $(../src/ %.c,%,$(SRCS))

CC:= gcc

CFLAGS:= -O3 -mavx2 -g

LDLIBS:= -lm -lpthread

PATH:= ../src/

all: $(PROGS)

%: %.c
	$(CC) -c $< -o $(PATH)$@.o $(CFLAGS) $(LDLIBS)

create: $(PROGS)
	ar r libllab.a *.o
	rm *.o
