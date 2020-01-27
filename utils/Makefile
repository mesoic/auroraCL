# -------------------------------------------------------------
# Executables and CFLAGS
# -------------------------------------------------------------
MD := mkdir
RM := rm
CC := g++

# Compiling for C++11 for linux OS
CFLAGS	:= -std=c++11 -Wall -DHAVE_CL2
CLIBS 	:= -lOpenCL

# Check for 32/64bit via kernel(uname)
PROC_TYPE = $(strip $(shell uname -m | grep 64))
ifeq ($(PROC_TYPE),)
	CFLAGS+=-m32
else
	CFLAGS+=-m64
endif

# -------------------------------------------------------------
# Build Environment
# -------------------------------------------------------------
BIN_DIR	:= bin
SRC_DIR := src
SRC_EXT	:= cpp

SOURCES := $(shell find -type f -name *.$(SRC_EXT))
OBJECTS := $(subst $(SRC_DIR),$(BIN_DIR),$(SOURCES:.$(SRC_EXT)=))

all: $(OBJECTS)

$(OBJECTS) : $(SOURCES)
	$(CC) $(CFLAGS) $(subst $(BIN_DIR),$(SRC_DIR),$@.$(SRC_EXT)) -o $@ $(CLIBS)