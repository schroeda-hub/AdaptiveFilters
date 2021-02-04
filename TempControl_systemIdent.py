import pyads
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from fir_adaptive_rls import fir_adaptive_rls
from fir_adaptive_lms import fir_adaptive_lms
from fir import fir

def shift_list(list_):
    for i in range(list_.size-1,0,-1):
        list_[i] = list_[i-1]
    return list_

def get_random(mean, std):
    return np.random.normal(mean, std, size=1)

def main():
    Fs = 10 # Hz
    buffer_size = 1024
    time_duration = buffer_size * 1/Fs
    
    remote_ip = '192.168.20.157'
    remote_ads = '192.168.30.202.1.1'
    plc = pyads.Connection(remote_ads, pyads.PORT_TC3PLC1, remote_ip)
    namespace_in = "KrogstrupMBE2.temperatureController[4].temperatureSP"
    namespace_out = "KrogstrupMBE2.temperatureController[4].temperaturePV"
    time = np.arange(0, time_duration, 1/Fs)
    plc.open()
    output = np.ones(buffer_size) * plc.read_by_name(
        namespace_out,
        pyads.PLCTYPE_LREAL)
    plc.close()
    
    test_fir = fir([])
    zeros_location = [
        1, 
        0+1j, 0-1j,
        test_fir.angle_to_complex(10), test_fir.angle_to_complex(-10),
        test_fir.angle_to_complex(20), test_fir.angle_to_complex(-20),
        test_fir.angle_to_complex(30), test_fir.angle_to_complex(-30)
    ]
    test_fir.set_b(test_fir.calc_from_zPlane(zeros_location))
    
    ad_filter = fir_adaptive_rls(20)
    prediction_error = np.zeros(buffer_size)
    
    fig,ax = plt.subplots(2)
    l0, = ax[0].plot(
        time,
        prediction_error)
    ax[0].set_xlim(0,time_duration)
    ax[0].set_xlabel('time')
    ax[0].axis('tight')
    l1, = ax[1].plot(ad_filter.b)
    ax[1].set_xlabel('Filter coefficients')
    ax[1].axis('tight')
    plt.pause(0.01)
    plt.tight_layout()
    
    testADS = False
    while True:
        input_ = get_random(0, 1) # input noise
        if testADS:
            # plc.open()
            # plc.write_by_name(namespace_in, input_, pyads.PLCTYPE_LREAL)
            # output = plc.read_by_name(namespace_out, pyads.PLCTYPE_LREAL)
            # plc.close()
            pass
        else:
            output = test_fir.filter(input_) # added noise of measurement
            output += get_random(0, 0.01)
            pass
        b,e,y = ad_filter.update(x=input_, d=output)
        shift_list(prediction_error)
        prediction_error[0] = 20*np.log10(abs(e))
        # print("b: {}\n e: {}\n----------------".format(b,e))
        l0.set_ydata(prediction_error)
        ax[0].set_ylim(min(prediction_error),max(prediction_error))
        l1.set_ydata(ad_filter.b)
        ax[1].set_ylim(min(ad_filter.b),max(ad_filter.b))
        plt.pause(0.01)
        sleep(1/Fs)

if __name__ == '__main__':
    main()