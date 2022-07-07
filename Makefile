CC:=clang++
FLAGS:=-Wall -std=c++17 -stdlib=libc++ -Iinclude -g
CUDA_FLAGS:=--cuda-gpu-arch=sm_86 -Xclang -fcuda-allow-variadic-functions -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -lcudart -ldl -lrt -lpthread
RELEASE_FLAGS:=-O3 -DNDEBUG -ffast-math

HEADERS:=$(wildcard include/*.h)
TESTS:=$(wildcard test/*.cu)

.PHONY: all tests compdb clean

all: compdb tests

release: FLAGS += ${RELEASE_FLAGS}

release: tests

tests: ${TESTS:test/%.cu=build/%} ${HEADERS}

compdb: Makefile
	compiledb --no-build make

clean:
	rm -f build/*

${TESTS:test/%.cu=build/%}: build/%: test/%.cu
	${CC} ${FLAGS} ${CUDA_FLAGS} -o $@ $^

