/*
 * Tomada de
 * https://korginc.github.io/logue-sdk/ref/minilogue-xd/v1.1-0/html/group__utils__float__math.html
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

float fastlog2f(float x)
{
    union {
        float f;
        unsigned int i;
    } vx = {x};

    union {
        unsigned int i;
        float f;
    } mx = {(vx.i & 0x007FFFFF) | 0x3f000000 };

    float y = vx.i;
    y *= 1.1920928955078125e-7f;
   return y - 124.22551499f
            - 1.498030302f * mx.f
            - 1.72587999f / (0.3520887068f + mx.f);
}

float fastlogf(float x)
{
    return 0.69314718f * fastlog2f(x);
}

void reverse(int arr[], int n)
{
    for (int low = 0, high = n - 1; low < high; low++, high--)
    {
        int temp = arr[low];
        arr[low] = arr[high];
        arr[high] = temp;
    }
}

__m256 fastlog2f_simd(__m256 x)
{
  union {
      __m256 f;
      __m256i i;
  } vx = {x};

  int aux[8];
  _mm256_storeu_si256 ((__m256i*)aux, vx.i);
  int n = sizeof(aux)/sizeof(aux[0]);
  reverse(aux,n);
  vx.i = _mm256_loadu_si256((__m256i*)aux);

  __m256i hexa1 = _mm256_set1_epi32(0x007FFFFF);
  __m256i hexa2 = _mm256_set1_epi32(0x3f000000);

  __m256i and12 = _mm256_castps_si256(
      _mm256_and_ps(_mm256_castsi256_ps(vx.i),_mm256_castsi256_ps(hexa1)));

  __m256i or12 = _mm256_castps_si256(
      _mm256_or_ps(_mm256_castsi256_ps(and12),_mm256_castsi256_ps(hexa2)));

  union {
      __m256i i;
      __m256 f;
  } mx = {or12};

  __m256 y = _mm256_cvtepi32_ps(vx.i);
  __m256 const1 = _mm256_set1_ps(1.1920928955078125e-7f);
  y = _mm256_mul_ps(y,const1);

  __m256 const2 =  _mm256_set1_ps(-124.22551499f);
  __m256 const3 =  _mm256_set1_ps(-1.498030302f);
  __m256 const4 =  _mm256_set1_ps(-1.72587999f);
  __m256 const5 =  _mm256_set1_ps(0.3520887068f);

  __m256 add1 = _mm256_add_ps(y,const2);
  __m256 add2 = _mm256_add_ps(const5,mx.f);
  __m256 add3 = _mm256_add_ps(_mm256_mul_ps(const3,mx.f),
                              _mm256_div_ps(const4,add2));

  return  _mm256_add_ps(add1,add3);
}

__m256 fastlogf_simd(__m256 x)
{
  __m256 const6 = _mm256_set1_ps(0.69314718f);
  return _mm256_mul_ps(const6,fastlog2f_simd(x));
}

int main(void)
{
    __m256 a = _mm256_set_ps(0.1f,0.2f,0.3f,0.5f,0.6f,0.7f,0.8f,0.9f);

    float c[8];
    _mm256_storeu_ps(c, fastlogf_simd(a));
    for(int i=0; i<8; i++) printf("%f ", c[i]); printf("\n");

    float g = fastlogf(0.5f);
    printf("%f\n", g);

    printf("%f\n", logf(0.5f));
}
