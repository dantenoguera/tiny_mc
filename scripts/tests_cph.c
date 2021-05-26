#include <x86intrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

#define SHELLS 101
#define PHOTONS 23

float heat[SHELLS] = {0.0f};
float heat2[SHELLS] = {0.0f};

static void photon(int * _nphotons)
{
    #pragma omp parallel
    {
        int cph = 0;
        // En número al que el cph de cada thread tiene que llegar viene dado por un elemento de _nphotons
        int nphotons = _nphotons[omp_get_thread_num()];
        float p_heat[SHELLS] = {0.0f};
        float p_heat2[SHELLS] = {0.0f};

        // Entonces, cph solo va hasta nphotons.
        // Basados en el ejemplo, algunos threads irán hasta cph = 6, mientras que otro hasta cph = 5
        // De esta forma logramos que el cph total sea igual a NPHOTONS.
        while (cph < nphotons)
        {
            p_heat[0]++;
            p_heat2[0]++;

            //#pragma omp atomic
            cph++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < SHELLS; i++) {
                heat[i] += p_heat[i];
                heat2[i] += p_heat2[i];
            }
        }
        printf("cph = %d\n", cph);
    }
}

int main()
{
    int threads = omp_get_max_threads();
    printf("hilos = %d\n", threads);
    // Me fijo cual es el resultado de la division PHOTONS / threads. Por ejemplo
    // si PHOTONS=23 y threads=4 => _res = 5
    int _res = PHOTONS / threads;
    // Me fijo cual es el resto de la division PHOTONS / threads. Por ejemplo
    // si PHOTONS=23 y threads=4 => _rem = 3
    int _rem = PHOTONS % threads;
    printf("%d  , %d  \n", _res,_rem);
    // Defino un arreglo donde en primer lugar, cada elemento sea igual a _res. Por ejemplo
    // Si _res = 5 => _nphotons[i] = _res para i en [0;threads)
    int _nphotons[threads];
    for( unsigned int k = 0; k < threads; k++ ){
      _nphotons[k] = _res;
    }
    // Ahora con el resto, le sumo 1 a cada elemento hasta alcanzar el valor de _rem. Por ejemplo
    // Si _res = 5 y _rem = 3 => _nphotons[i] = _res + 1  para i en [0;_rem) y _nphotons[i] = _res  para i en [_rem;threads)
    // Finalmente, el arreglo _nphotons quedaría:
    // _nphotons[0] = 6
    //  _nphotons[1] = 6
    //  _nphotons[2] = 6
    //  _nphotons[3] = 5
    // Y si sumamos elemento a elemento el total da 23 = PHOTONS
    for( unsigned int i = 0; i < _rem; i++ ){
      _nphotons[i] += 1;
    }
    for( unsigned int j=0; j < threads; j++ ){
      printf("%d\n", _nphotons[j]);
    }

    // Le paso el arreglo a la función photon
    photon(_nphotons);
    printf("heat[0] = %f\n", heat[0]);
    printf("heat2[0] = %f\n", heat2[0]);

    return 0;
}
