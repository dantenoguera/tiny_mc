#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress as lreg

cc = [['gcc'],['clang']]

oflags = [['-O0'],['-O1', '-ffast-math', '-march=native'], 
    ['-O2', '-ffast-math', '-march=native'], 
    ['-O3', '-ffast-math', '-march=native']]

samples = [32768, 2 * 32768, 3 * 32768, 4 * 32768, 5 * 32768, 6 * 32768, 7 * 32768, 8 * 32768]

fgcc = open('average_gcc', 'w')
fclang = open('average_clang', 'w')
fperf = open('perf.out', 'a')

for c in cc:
    if c == ['gcc']:
        fperf.write('# GCC\n')
        fperf.flush()
        os.fsync(fperf.fileno())
    elif c == ['clang']:
        fperf.write('# CLANG\n')
        fperf.flush()
        os.fsync(fperf.fileno())
    for fl in oflags:
        if c == ['gcc']:
            fgcc.write('# Flags: {:10}\n'.format(fl[0]))
            fperf.write('# Flags: {:10}\n'.format(fl[0]))
            fperf.flush()
            os.fsync(fperf.fileno())
        elif c == ['clang']:
            fclang.write('# Flags: {:10}\n'.format(fl[0]))
            fperf.write('# Flags: {:10}\n'.format(fl[0]))
            fperf.flush()
            os.fsync(fperf.fileno())

        val_average = []
        val_std = []

        for s in samples:
            fperf.write('# Photons: {:10}\n'.format(s))
            fperf.flush()
            os.fsync(fperf.fileno())
            values = []
            cmd = c + fl + ['-DVERBOSE=0', '-DPHOTONS=' + str(s)] + ['-std=gnu11', '-Wall', '-Wextra', '-o', 'tiny_mc', '../tiny_mc.c', '-lm']
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

            for i in range(20):
                p = subprocess.run('perf stat -e instructions -e cycles -x \'   \' -o perf.out --append ./tiny_mc', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                values.append(float(p.stdout))

            val_average.append(np.average(values))
            val_std.append(np.std(values))

            if c == ['gcc']:
                fgcc.write('{:10.1f} {:10.4f} {:10.4f}\n'.format(s,np.average(values),np.std(values)))
            elif c == ['clang']:
                fclang.write('{:10.1f} {:10.4f} {:10.4f}\n'.format(s,np.average(values),np.std(values)))

        linreg = lreg(samples,val_average) # y = a + bx

        if c == ['gcc']:
            fgcc.write('{:10.7f} {:10.4f} {:10.4f} {:10.4f}\n'.format(linreg.slope,linreg.intercept,linreg.intercept_stderr,linreg.rvalue**2))
        elif c == ['clang']:
            fclang.write('{:10.7f} {:10.4f} {:10.4f} {:10.4f}\n'.format(linreg.slope,linreg.intercept,linreg.intercept_stderr,linreg.rvalue**2))

fgcc.close()
fclang.close()
fperf.close()
