from pylab import *
import scipy.signal as signal
from fir import fir
import math

#Plot frequency and phase response
def mfreqz(b,a=1):
    w,h = signal.freqz(b,a)
    h_dB = 20 * math.log10 (abs(h))
    subplot(211)
    plot(w/max(w),h_dB)
    ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = unwrap(math.arctan2(imag(h),real(h)))
    plot(w/max(w),h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    subplots_adjust(hspace=0.5)

#Plot step and impulse response
def impz(b,a=1):
    l = len(b)
    impulse = repeat(0.,l); impulse[0] =1.
    x = arange(0,l)
    response = signal.lfilter(b,a,impulse)
    subplot(211)
    stem(x, response)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Impulse response')
    subplot(212)
    step = cumsum(response)
    stem(x, step)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Step response')
    subplots_adjust(hspace=0.5)
    
def _test():
    test_fir = fir([])
    lp_zeros_location = [
        -1, 
        0+1j, 0-1j,
        test_fir.angle_to_complex(170), test_fir.angle_to_complex(-170),
        test_fir.angle_to_complex(160), test_fir.angle_to_complex(-160),
        test_fir.angle_to_complex(150), test_fir.angle_to_complex(-150)
    ]
    hp_zeros_location = [
        1, 
        0+1j, 0-1j,
        test_fir.angle_to_complex(10), test_fir.angle_to_complex(-10),
        test_fir.angle_to_complex(20), test_fir.angle_to_complex(-20),
        test_fir.angle_to_complex(30), test_fir.angle_to_complex(-30)
    ]
    test_fir.set_b(test_fir.calc_from_zPlane(hp_zeros_location))  
    impz(test_fir.b)
    show()
    pass

if __name__ == '__main__':
    _test()