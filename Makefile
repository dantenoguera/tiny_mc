# Compilers
CC = gcc

# Flags
CFLAGS = -std=gnu11 -Wall -Wextra -DVERBOSE=0 -O2 -march=native -ffast-math -flto
LDFLAGS = -lm

# Binary file
TARGET = tiny_mc

# Files
C_SOURCES = tiny_mc.c xoshiro128+.c logf.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

# Rules
all: $(TARGET)

$(TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
