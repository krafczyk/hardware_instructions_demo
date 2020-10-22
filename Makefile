CXXFLAGS=-std=c++11

all: asm/loop1_O0.S bin/loop1_O0 asm/loop1_O1.S bin/loop1_O1 asm/loop1_O2.S bin/loop1_O2 asm/loop1_O3.S bin/loop1_O3 asm/loop1_O2_avx.S bin/loop1_O2_avx asm/loop1_O2_avx2.S bin/loop1_O2_avx2 asm/loop1_O2_native.S bin/loop1_O2_native asm/loop1_O2_i386.S bin/loop1_O2_i386 asm/loop1_O2_i686.S bin/loop1_O2_i686 asm/loop1_O2_nehalem.S bin/loop1_O2_nehalem asm/loop1_O3_native.S bin/loop1_O3_native asm/loop1_O3_skylake-avx512.S bin/loop1_O3_skylake-avx512 asm/loop1_O3_knl.S bin/loop1_O3_knl asm/loop1_O3_nehalem.S bin/loop1_O3_nehalem asm/loop1_O3_i386.S bin/loop1_O3_i386 bin/loop1_opencl_O3 bin/opencl_test

dirs:
	mkdir -p bin
	mkdir -p asm

asm/loop1_O0.S: loop1.cpp
	g++ -O0 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O0: loop1.cpp
	g++ -O0 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O1.S: loop1.cpp
	g++ -O1 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O1: loop1.cpp
	g++ -O1 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2.S: loop1.cpp
	g++ -O2 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2: loop1.cpp
	g++ -O2 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O3.S: loop1.cpp
	g++ -O3 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O3: loop1.cpp
	g++ -O3 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2_avx.S: loop1.cpp
	g++ -O2 -mavx ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2_avx: loop1.cpp
	g++ -O2 -mavx ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2_avx2.S: loop1.cpp
	g++ -O2 -mavx2 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2_avx2: loop1.cpp
	g++ -O2 -mavx2 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2_native.S: loop1.cpp
	g++ -O2 -march=native ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2_native: loop1.cpp
	g++ -O2 -march=native ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2_i386.S: loop1.cpp
	g++ -O2 -march=i386 -m32 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2_i386: loop1.cpp
	g++ -O2 -march=i386 -m32 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2_i686.S: loop1.cpp
	g++ -O2 -march=i686 -m32 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2_i686: loop1.cpp
	g++ -O2 -march=i686 -m32 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O2_nehalem.S: loop1.cpp
	g++ -O2 -march=nehalem ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O2_nehalem: loop1.cpp
	g++ -O2 -march=nehalem ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O3_native.S: loop1.cpp
	g++ -O3 -march=native ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O3_native: loop1.cpp
	g++ -O3 -march=native ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O3_skylake-avx512.S: loop1.cpp
	g++ -O3 -march=skylake-avx512 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O3_skylake-avx512: loop1.cpp
	g++ -O3 -march=skylake-avx512 ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O3_knl.S: loop1.cpp
	g++ -O3 -march=knl ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O3_knl: loop1.cpp
	g++ -O3 -march=knl ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O3_nehalem.S: loop1.cpp
	g++ -O3 -march=nehalem ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O3_nehalem: loop1.cpp
	g++ -O3 -march=nehalem ${CXXFLAGS} loop1.cpp -o $@

asm/loop1_O3_i386.S: loop1.cpp
	g++ -O3 -march=i386 -m32 ${CXXFLAGS} -S loop1.cpp -o $@

bin/loop1_O3_i386: loop1.cpp
	g++ -O3 -march=i386 -m32 ${CXXFLAGS} loop1.cpp -o $@

bin/loop1_opencl_O3: loop1_opencl.cpp
	g++ -O3 ${CXXFLAGS} loop1_opencl.cpp -l OpenCL -o $@

bin/opencl_test: opencl_test.cpp
	g++ -O3 ${CXXFLAGS} opencl_test.cpp -l OpenCL -o $@

clean:
	rm bin/*
	rm asm/*
