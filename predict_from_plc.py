import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from fir_adaptive_rls import fir_adaptive_rls
from fir_adaptive_lms import fir_adaptive_lms

def shift_list(list_):
    for i in range(list_.size-1,0,-1):
        list_[i] = list_[i-1]
    return list_

class signal_creator_obj(object):
    def __init__(self, Fs:float):
        self.sample_number = -1
        self.Fs = Fs
        self.f = 5 # Hz
        self.noise_enable = True

    def _noise(self):
        mean = 0
        std = 1
        return np.random.normal(mean, std, size=1) * 0.01

    def get(self):
        self.sample_number += 1
        return np.sin(self.sample_number/self.Fs * 2 * np.pi * self.f) + self._noise()

def main():
    Fs = 20 # Hz
    buffer_size = 1024
    time_duration = buffer_size * 1/Fs

    ads_enable = False

    # ax[0].set_xlim(0,time_duration)
    # ax[0].set_ylim(-10,10)
    # ax[0].set_title("Signal")

    time = np.arange(0, time_duration, 1/Fs)
    signal = np.ones(buffer_size)

    if ads_enable:
        import pyads
        remote_ip = '192.168.20.157'
        remote_ads = '192.168.30.202.1.1'
        plc = pyads.Connection(remote_ads, pyads.PORT_TC3PLC1, remote_ip)
        namespace = "KrogstrupMBE2.temperatureController[4].temperaturePV"
        plc.open()
        signal *= plc.read_by_name(
            namespace,
            pyads.PLCTYPE_LREAL)
        plc.close()


    delay_samples = 15
    delay_buffer = np.ones(delay_samples) * signal[0]
    ad_filter = fir_adaptive_rls(20, p=0.6) # RLS
    # ad_filter = fir_adaptive_lms(20) # LMS
    predicted_values = np.zeros(buffer_size) + signal
    prediction_error = np.zeros(buffer_size)
    error = np.zeros(buffer_size)
    fig,ax = plt.subplots(4)
    l1, l2, l3 = ax[0].plot(
        time,
        signal,
        "-",
        time,
        predicted_values,
        "-",
        time[delay_samples:delay_samples+ad_filter.N],
        ad_filter.x)
    ax[0].set_xlim(0,time_duration)
    ax[0].legend((l1, l2, l3),('signal', 'predicted signal', 'training_data', 'error'))
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('')
    ax[0].axis('tight')
    l4, = ax[1].plot(time, prediction_error)
    ax[1].set_xlim(0,time_duration)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('prediction error [dB]')
    ax[1].axis('tight')
    l5, = ax[2].plot(time, error)
    ax[2].set_xlim(0,time_duration)
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('error')
    ax[2].axis('tight')
    l6, = ax[3].plot(ad_filter.b)
    ax[3].set_xlabel('Filter value')
    ax[3].axis('tight')
    plt.pause(0.01)
    plt.tight_layout()


    if not ads_enable:
        signal_creator = signal_creator_obj(Fs)
        signal *= 0
        delay_buffer *= 0
        predicted_values *= 0
        for i in range(int(delay_samples*0.1)):
            shift_list(signal)
            signal[0] = signal_creator.get()
        pass

    while True:
        shift_list(signal)
        # Read new value
        if ads_enable:
            plc.open()
            signal[0] = plc.read_by_name(
                namespace,
                pyads.PLCTYPE_LREAL)
            plc.close()
        else:
            signal[0] = signal_creator.get()

        shift_list(delay_buffer)
        delay_buffer[0] = signal[0]

        # update adaptive filter
        b,e,y = ad_filter.update(x=delay_buffer[-1], d=signal[0])
        shift_list(predicted_values)
        predicted_values[0] = y
        shift_list(error)
        error[0] = e
        shift_list(prediction_error)
        prediction_error[0] = 20*np.log10(abs(e))
        # print("b: {},\ne: {}\n--------".format(b,e))
        l1.set_ydata(signal)
        l2.set_ydata(predicted_values)
        l3.set_ydata(ad_filter.x)
        ax[0].set_ylim(min(predicted_values),max(predicted_values))

        l4.set_ydata(prediction_error)
        try:
            ax[1].set_ylim(min(prediction_error),max(prediction_error))
        except:
            pass

        l5.set_ydata(error)
        ax[2].set_ylim(min(error),max(error))

        l6.set_ydata(b)
        ax[3].set_ylim(min(b),max(b))
        plt.pause(0.01)
        sleep(1/Fs)

if __name__ == '__main__':
    main()