
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(void)
{
    int N = 100000;
    srand(time(NULL));
    FILE * fp;
    fp = fopen("intrand.out","w");
    float random;
    for (unsigned int i = 0; i < N; ++i) {
        random = rand() / (float)RAND_MAX;
        fprintf(fp,"%f\n", random);
    }
    fclose(fp);
}
    
