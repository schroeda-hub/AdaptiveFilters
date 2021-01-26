import numpy as np
import matplotlib.pyplot as plt
from fir import fir
from fir_adaptive_lms import fir_adaptive_lms
from fir_adaptive_rls import fir_adaptive_rls
from plot_zplane import zplane
from plot_freqz import plot_freqz

def angle_to_complex(angle):
    return np.exp(1j*angle*2*np.pi/360)

if __name__ == '__main__':
    mean = 0
    std = 1 
    num_samples = 2000
    adaptive_filterOrder = 15
    inputSamples = np.random.normal(mean, std, size=num_samples)
    noise = np.random.normal(mean, std*0.001, size=num_samples)
    fig, axs = plt.subplots(2,1, constrained_layout = True)
    axs[0].plot(inputSamples)
    axs[0].set_title("Input signal")
    axs[0].set_ylabel("time")
    
    zeros_location = [
        1, 
        0+1j, 0-1j,
        angle_to_complex(10), angle_to_complex(-10),
        angle_to_complex(20), angle_to_complex(-20),
        angle_to_complex(30), angle_to_complex(-30)
    ]
    zeros_location = [
        -1, 
        angle_to_complex(160), angle_to_complex(-160),
        angle_to_complex(140), angle_to_complex(-140),
        angle_to_complex(120), angle_to_complex(-120),
        angle_to_complex(100), angle_to_complex(-100),
        angle_to_complex(80), angle_to_complex(-80)
    ]
    zeros_poly = np.poly(zeros_location)
    print(zeros_poly)
    
    try:
        firObj = fir(zeros_poly)
        print("Filter order: {}".format(firObj.N))
    except:
        firObj = fir([1,-1,1,-1])
    output = firObj.filter(inputSamples)
    output = output + noise
    axs[1].plot(output)
    axs[1].set_title("Output signal")
    axs[1].set_ylabel("time")
    
    z,p,k = zplane(np.asarray(firObj.b),np.asarray(firObj.a))
    plot_freqz(np.asarray(firObj.b))
    
    adaptive_fir_lms = fir_adaptive_lms(adaptive_filterOrder)
    b_lms,e_lms = adaptive_fir_lms.train(inputSamples, output, 0.05)
    print(b_lms[-1])
    print("LMS remaining error: {}".format(np.mean(np.abs(e_lms[int(-1*num_samples/0.1):-1]))))
    
    adaptive_fir_rls = fir_adaptive_rls(adaptive_filterOrder)
    b_rls, e_rls = adaptive_fir_rls.train(inputSamples, output, 0.8)
    print(b_rls[-1])
    print("RLS remaining error: {}".format(np.mean(np.abs(e_rls[int(-1*num_samples/0.1):-1]))))
    
    fir_lms = fir(b_lms[-1])
    fir_lms_output = fir_lms.filter(inputSamples)
    fir_rls = fir(b_rls[-1])
    fir_rls_output = fir_rls.filter(inputSamples)
    fig, axs = plt.subplots(3,1, constrained_layout = True)
    axs[0].plot(inputSamples)
    axs[0].set_title("Input signal")
    axs[0].set_xlabel("time")
    l1, l2, l3 = axs[1].plot(fir_lms_output, '-', fir_rls_output, '--', output,'.-')
    axs[1].set_title("Output signals")
    axs[1].set_xlabel("time")
    axs[1].legend((l1, l2, l3),('lms_output', 'rls_output', 'fir_output'))
    l4, l5 = axs[2].plot(20*np.log10(np.square(e_lms)), '-', 20*np.log10(np.square(e_rls)), '.-')
    axs[2].legend((l4, l5),('lms_error', 'rls_error'))
    axs[2].set_title("Error during training")
    axs[2].set_ylabel("MSE [dB]")
    axs[2].set_xlabel("time")
    plt.show()
    pass
