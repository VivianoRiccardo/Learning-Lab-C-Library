DIR:= ../src/
DIRTEST:= ../tests/

EXEC:= main
EXEC8_1:= server_side
EXEC8_2:= client_side
T1:=test1/
T2:=test2/
T3:=test3/
T4:=test4/
T5:=test5/
T6:=test6/
T7:=test7/
T9:=test9/
T10:=test10/
T11:=test11/
T12:=test12/
T13:=test13/
T16:=test16/
T17:=test17/
T18:=test18/


SRCS = $(wildcard $(DIR)*.c)

PROGS = $(patsubst %.c,%,$(SRCS))
PROGTESTS = $(patsubst %.c,%,$(SRCSTEST))

CC:= gcc

CFLAGS:= -O3 -mavx2 -g -pg
LDLIBS:= -lm -lpthread
LABLIB:= -lllab

all: $(PROGS)

%: %.c
	$(CC) -c $< -o $@.o $(CFLAGS) $(LDLIBS)

create: $(DIR)
	ar r libllab.a $(DIR)*.o
	rm $(DIR)*.o

test: $(DIRTEST)
	$(CC) -o $(DIRTEST)$(T1)$(EXEC) $(DIRTEST)$(T1)*.c $(LABLIB) $(LDLIBS) $(CFLAGS) 
	$(CC) -o $(DIRTEST)$(T2)$(EXEC) $(DIRTEST)$(T2)*.c $(LABLIB) $(LDLIBS) $(CFLAGS) 
	$(CC) -o $(DIRTEST)$(T3)$(EXEC) $(DIRTEST)$(T3)*.c $(LABLIB) $(LDLIBS) $(CFLAGS) 
	$(CC) -o $(DIRTEST)$(T4)$(EXEC) $(DIRTEST)$(T4)*.c $(LABLIB) $(LDLIBS) $(CFLAGS) 
	$(CC) -o $(DIRTEST)$(T5)$(EXEC) $(DIRTEST)$(T5)*.c $(LABLIB) $(LDLIBS) $(CFLAGS) 
	$(CC) -o $(DIRTEST)$(T6)$(EXEC) $(DIRTEST)$(T6)*.c $(LABLIB) $(LDLIBS) $(CFLAGS) 
	$(CC) -o $(DIRTEST)$(T7)$(EXEC) $(DIRTEST)$(T7)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T9)$(EXEC) $(DIRTEST)$(T9)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T10)$(EXEC) $(DIRTEST)$(T10)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T11)$(EXEC) $(DIRTEST)$(T11)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T12)$(EXEC) $(DIRTEST)$(T12)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T13)$(EXEC) $(DIRTEST)$(T13)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T16)$(EXEC) $(DIRTEST)$(T16)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T17)$(EXEC) $(DIRTEST)$(T17)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
	$(CC) -o $(DIRTEST)$(T18)$(EXEC) $(DIRTEST)$(T18)*.c $(LABLIB) $(LDLIBS) $(CFLAGS)
