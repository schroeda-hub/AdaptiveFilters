from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_freqz(b, a=1, worN=None, whole=0, plot=None):
    w, h = signal.freqz(b, a, worN, whole, plot)
    fig, ax1 = plt.subplots(1,1)
    plt.title('Digital filter frequency response')
    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()
