BUILDDIR = build
CC = g++
CFLAGS  = -g -Wall
LDLIBS += -lOpenCL -lfftw3f
SOURCEDIR = src
HEADERDIR = -Iinclude -Isrc 
PROG = fft
LD = g++

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(addprefix $(BUILDDIR)/,$(SOURCES:src/%.cpp=%.o))

$(info $$SOURCES is [${SOURCES}])
$(info $$OBJECTS is [${OBJECTS}])


all: $(BUILDDIR)/$(PROG)

$(BUILDDIR)/$(PROG): $(OBJECTS)
	$(LD) $(LDFLAGS)  $(OBJECTS)  -o $(BUILDDIR)/$(PROG) $(LDLIBS)

$(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	$(CC) $(CFLAGS)  $(HEADERDIR) -c $< -o $@ $(LDLIBS)




    



