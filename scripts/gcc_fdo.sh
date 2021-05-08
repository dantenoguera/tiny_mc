# Feedback directed optimization (FDO)

echo "compiling..."

gcc ../tiny_mc.c -o tiny_mc -fprofile-generate \
    -std=gnu11 -Wall -Wextra -DVERBOSE=0       \
    -O3 -ffast-math -march=native -lm          \

echo "executing..."
./tiny_mc

echo "\nre-compiling..."
gcc ../tiny_mc.c -o tiny_mc -fprofile-use \
    -std=gnu11 -Wall -Wextra -DVERBOSE=0  \
    -O3 -ffast-math -march=native -lm     \

echo "cleaning...\n"
rm tiny_mc.gcda
