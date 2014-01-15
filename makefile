CC=g++
CFLAGS=-c -g -O0 -Wall
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgproc
SOURCES=test_ShapeBasedDetection.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=lb_trafficSignDetect

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)