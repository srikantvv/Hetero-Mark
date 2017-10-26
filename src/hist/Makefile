prog = hist

all: host

host:
	g++ -static -std=c++11 -I../../../../cl-runtime/ $(prog).cpp $(prog)_benchmark.cc $(prog)_cl12_benchmark.cc -L../../../../cl-runtime/dbg64b/ -lOpenCL -o $(prog)
kernel:
	clc2 --enable-hsail-extensions --cl-std=CL2.0 $(prog)-kernel.cl -o $(prog).fe.bc
	llvm-link $(prog).fe.bc -l ../../../../HSAIL-HLC-Stable/lib/builtins-hsail.bc -o $(prog).linked.bc
	opt -O3 -gpu -whole -verify $(prog).linked.bc -o $(prog).opt.bc
	llc -filetype=obj -hsail-enable-gcn=0 -march=hsail-64 $(prog).opt.bc -o $(prog)_kernel.asm
	rm $(prog).fe.bc $(prog).linked.bc $(prog).opt.bc
dis:
	HSAILasm -disassemble $(prog)_kernel.asm -o $(prog)_kernel.dasm
ass:
	HSAILasm -assemble -o $(prog)_kernel.asm $(prog)_kernel.dasm
