# Profile guided optimization (PGO)

echo "compiling..."

clang ../tiny_mc.c -o tiny_mc                \
    -fprofile-instr-generate=tiny_mc.profraw \
    -std=gnu11 -Wall -Wextra -DVERBOSE=0     \
    -O3 -ffast-math -march=native -lm

echo "executing..."
./tiny_mc

echo "\nre-compiling..."
llvm-profdata-10 merge -output tiny_mc.profdata tiny_mc.profraw
clang ../tiny_mc.c -o tiny_mc            \
    -fprofile-use=tiny_mc.profdata       \
    -std=gnu11 -Wall -Wextra -DVERBOSE=0 \
    -O3 -ffast-math -march=native -lm

echo "cleaning...\n"
rm tiny_mc.profraw
rm tiny_mc.profdata


