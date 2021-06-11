CU=nvcc
CUFLAGS=-O2 -Xcompiler=-Wall -Xcompiler=-Wextra
LDFLAGS=

TARGETS=tiny_mc

all: $(TARGETS)

tiny_mc: tiny_mc.o
	$(CU) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

# CU, CUFLAGS no tienen regla implicita como CC/CFLAGS
%.o: %.cu
	$(CU) $(CUFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f *.o $(TARGETS)
