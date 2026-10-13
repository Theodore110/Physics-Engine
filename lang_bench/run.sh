#!/bin/sh
# Same solver-kernel in C++ (GCC, Clang), Rust (safe, unchecked) and Zig, timed by one C harness.
# Needs: g++, clang++, rustc, zig (e.g. `pip install ziglang` then `python3 -m ziglang`).
set -e
G="-O3 -fno-math-errno -c"
g++     $G -o cpp_gcc.o   kernel.cpp
clang++ $G -o cpp_clang.o kernel.cpp
rustc -C opt-level=3 -C panic=abort --crate-type=staticlib -o librs_safe.a   kernel_safe.rs
rustc -C opt-level=3 -C panic=abort --crate-type=staticlib -o librs_unsafe.a kernel_unsafe.rs
python3 -m ziglang build-obj kernel.zig -O ReleaseFast -fPIC -femit-bin=zig.o
L="-lm -lpthread -ldl"
gcc -O2 main.c cpp_gcc.o   -lstdc++ -o b_cpp_gcc   $L
gcc -O2 main.c cpp_clang.o -lstdc++ -o b_cpp_clang $L
gcc -O2 main.c librs_safe.a   -o b_rs_safe   $L
gcc -O2 main.c librs_unsafe.a -o b_rs_unsafe $L
gcc -O2 main.c zig.o -o b_zig $L
# args: grid side N, steps, repeats.  Identical checksums prove identical computation.
for b in cpp_gcc cpp_clang rs_safe rs_unsafe zig; do printf "%-10s " $b; ./b_$b 200 60 7; done
