PLATFORM=gcc
include ../config/Makefile.in.$(PLATFORM)

# ===
# Main driver and sample run

lshallow: ldriver.o shallow2d.o stepper.o
	$(NVCC) $(LUA_CFLAGS) -o $@ $^ $(NVCCLIBS) $(LUA_LIBS)

ldriver.o: ldriver.cu shallow2d.cuh stepper.cuh
	$(NVCC) $(LUA_CFLAGS) -c $<

# ===
# CUDA Test Runs

shallow2d_test: shallow2d_test.o shallow2d.o shallow2d_base.o
	$(NVCC) -o $@ $^ $(NVCCLIBS)

shallow2d_test.o: shallow2d_test.cu shallow2d.cuh shallow2d_base.h
	$(NVCC) -c $<

shallow2d.o: shallow2d.cu
	$(NVCC) -c $<

shallow2d_base.o: shallow2d_base.c
	$(NVCC) -c $<

stepper_test: stepper_test.o stepper.o stepper_base.o
	$(NVCC) -o $@ $^ $(NVCCLIBS)

stepper_test.o: stepper_test.cu stepper.cuh stepper_base.h shallow2d.cuh
	$(NVCC) -c $<

stepper.o: stepper.cu stepper.cuh
	$(NVCC) -c $<

stepper_base.o: stepper_base.c stepper_base.h
	$(NVCC) -c $<

# ===
# Documentation

shallow.md: shallow2d.h shallow2d.c stepper.h stepper.c ldriver.c
	ldoc $^ -o $@

# ===
# Clean up

.PHONY: clean
clean:
	rm -f lshallow *.o
	rm -f shallow.md
