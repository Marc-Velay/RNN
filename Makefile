CPP=clang++
CFLAG=-Wall -Wextra -std=c++11 -g `pkg-config --cflags opencv` -lopencv_core   #-Werror -O3 

all: compile exec clean

compile: dataReader.o cellStructure.o matrix_maths.o RNNTrainer.o
	${CPP} -o RNN ${CFLAG} $^

%.o: src/%.cpp
	${CPP} -o $@ ${CFLAG} -c $<

exec:
	./RNN

clean:
	rm *.o RNN
