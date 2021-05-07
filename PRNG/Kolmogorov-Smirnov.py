# Kolmogorov-Smirnov test to analise the randomness of differents PRNG

import math as m
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pylab import cm

# Open file to read numbers
random = np.genfromtxt('xoshirovectrand.out', usecols=0)#[0:100]

# Build the theoretical probabilistic function
spaced = 100
x = np.linspace(0,1,spaced)
y = x

# Build the statistic measured
k_plus = []
k_minus = []
n = 100
counter = np.zeros(spaced)
# Loop every 100 measurements
for k in range(0,n):
    random_n = random[1000*k:1000*(k+1)-1]
    # Loop over every random number generated
    for i in range(0,len(random_n)):
        # Loop over every segment [0,end] with end belong [1/100,1]
        for j in range(0,spaced):
            # If random[i] belong to the segment
            if random_n[i] < x[j]:
                # Add 1 to counter
                counter[j] += 1

    y_meas = counter/len(random_n)

    # Standars deviations
    #k_plus = m.sqrt(len(random))*np.amax(y_meas - y)
    #k_minus = m.sqrt(len(random))*np.amax(y - y_meas)
    #print(k_plus,k_minus)

    aux_plus = np.linspace(1,len(random_n),len(random_n))/len(random_n)
    k_plus.append(m.sqrt(len(random_n))*np.amax(aux_plus - np.sort(random_n)))
    aux_minus = np.linspace(0,len(random_n)-1,len(random_n))/len(random_n)
    k_minus.append(m.sqrt(len(random_n))*np.amax(np.sort(random_n) - aux_minus))

k_plus = np.array(k_plus)
k_minus = np.array(k_minus)

# Build theoretical behavior of k's
y_k = 1-np.exp(-2*x**2)
# Builds statistic over k's
counter_k_plus = np.zeros(spaced)
counter_k_minus = np.zeros(spaced)
# Loop over every k^+
for i in range(0,len(k_plus)):
    # Loop over every segment [0,end] with end belong [1/100,1]
    for j in range(0,spaced):
        # If random[i] belong to the segment
        if k_plus[i] < x[j]:
            # Add 1 to counter
            counter_k_plus[j] += 1
y_k_plus = counter_k_plus/len(k_plus)
# Loop over every k^-
for i in range(0,len(k_minus)):
    # Loop over every segment [0,end] with end belong [1/100,1]
    for j in range(0,spaced):
        # If random[i] belong to the segment
        if k_minus[i] < x[j]:
            # Add 1 to counter
            counter_k_minus[j] += 1
y_k_minus = counter_k_minus/len(k_minus)


# Plotting process

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1, 1, 1)
mpl.style.use('classic')
plt.rcParams.update({"text.usetex": True,"font.family": "Tex Gyre Pagella"})
colors = cm.get_cmap('tab10', 10)
#Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which='major', size=7, width=1, direction='in',top=True)
ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in',top=True)
ax.yaxis.set_tick_params(which='major', size=7, width=1, direction='in',right=True)
ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in',right=True)

#Edit the major and minor tick locations
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.plot(x,y_k,'-',color='k',lw=1.5,label=r'Theoric Values')
ax.plot(x,y_k_plus,'-',color=colors(0),lw=1.5,label=r'$K^+$', drawstyle='steps')
ax.plot(x,y_k_minus,'-',color=colors(1),lw=1.5,label=r'$K^-$', drawstyle='steps')

ax.set_xticks([0.,0.2,0.4,0.6,0.8,1.0])
ax.set_xticklabels(['$0\%$','$20\%$','$40\%$',
                    '$60\%$','$80\%$','$100\%$'])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
    tick.label.set_fontname("Tex Gyre Pagella")
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
    tick.label.set_fontname("Tex Gyre Pagella")
    
ax.set_ylabel(r'$K$', fontname="Tex Gyre Pagella",fontsize=16)
ax.set_xlabel(r'$\%$', fontname="Tex Gyre Pagella",fontsize=16)

#plt.annotate(r'$K^+ = %4.3f$'% k_plus, (0.1,0.5),fontsize=16)
#plt.annotate(r'$K^- = %4.3f$'% k_minus, (0.1,0.45),fontsize=16)

ax.legend(fontsize=16,loc=2)
plt.tight_layout()
plt.savefig('rand_xoshirovect.pdf', format='pdf')
plt.show()

