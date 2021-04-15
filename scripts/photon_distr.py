#!/usr/bin/env python3

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

f = open('values', 'w')

# grafica distribución de fotones por segundo (VERBOSE debe ser 0)
samples = [32768, 2 * 32768, 3 * 32768, 4 * 32768, 5 * 32768, 6 * 32768, 7 * 32768, 8 * 32768]
values = []
for s in samples:
    cmd = ['gcc', '-DVERBOSE=0', '-DPHOTONS=' + str(s) , '-std=gnu11', '-Wall', '-Wextra', '-o', 'tiny_mc', '../tiny_mc.c', '-lm']
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for i in range(20):
        p = subprocess.run('./tiny_mc', check=True, stdout=subprocess.PIPE, universal_newlines=True)
        values.append(float(p.stdout))
        f.write('{:10.4f}\n'.format(float(p.stdout)))

sns.kdeplot(values, shade=True)
f.close()
print(np.average(values), np.std(values))

plt.show()
