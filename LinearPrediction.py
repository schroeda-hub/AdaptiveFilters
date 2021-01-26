import numpy as np
import matplotlib.pyplot as plt
from fir import fir
from fir_adaptive_lms import fir_adaptive_lms
from fir_adaptive_rls import fir_adaptive_rls
from plot_zplane import zplane
from plot_freqz import plot_freqz

time = np.linspace(0, 40*np.pi, int(1e5))
mean = 0
std = 0.1
signal = np.sin(time)
signal += 0.4 * np.cos(time * 1/3)
signal += time % 4*np.pi
noise = np.random.normal(mean, std, size=time.size)
# signal += noise

delay_samples = 20
# x: delayed signal
x = signal[delay_samples:]
signal = signal[:-1*delay_samples]
time = time[:-1*delay_samples]

fir_rls = fir_adaptive_rls(10)

b,e,y = fir_rls.train(x, signal, 0.8)
print(b[-1])

fig, axs = plt.subplots(2,1, constrained_layout = True)
l1, l2 = axs[0].plot(time, signal, '--' , time[fir_rls.N:], y)
axs[0].legend((l1, l2),('signal', 'predicted signal'))
axs[0].set_xlabel('time')
axs[0].set_ylabel('')
axs[0].axis('tight')
axs[1].plot(time[fir_rls.N:],20*np.log10(np.absolute(e)))
axs[1].set_xlabel('time')
axs[1].set_ylabel('error')
axs[1].axis('tight')
plt.show()

