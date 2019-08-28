DIR:= ../src/
SRCS = $(wildcard $(DIR)*.c)
PROGS = $(patsubst %.c,%,$(SRCS))

CC:= gcc

CFLAGS:= -O3 -mavx2 -g
LDLIBS:= -lm -lpthread


all: $(PROGS)

%: %.c
	$(CC) -c $< -o $@.o $(CFLAGS) $(LDLIBS)

create: $(DIR)
	ar r libllab.a $(DIR)*.o
	rm $(DIR)*.o
