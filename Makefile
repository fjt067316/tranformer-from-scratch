CC = g++
CFLAGS = -pthread -std=c++11 -I./classes
TARGET = main.o
HEADERS = transformer.h fcl.h softmax.h helper.h decoder.h layer_norm.h encoder.h attention.h

all: $(TARGET)

$(TARGET): main.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)
