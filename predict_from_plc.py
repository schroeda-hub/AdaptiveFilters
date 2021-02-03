import pyads
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from fir_adaptive_rls import fir_adaptive_rls

def shift_list(list_):
    for i in range(list_.size-1,0,-1):
        list_[i] = list_[i-1]
    return list_
        

def main():
    Fs = 1000 # Hz
    buffer_size = 1024
    time_duration = buffer_size * 1/Fs
    # ax[0].set_xlim(0,time_duration)
    # ax[0].set_ylim(-10,10)
    # ax[0].set_title("Signal")
    
    remote_ip = '192.168.20.157'
    remote_ads = '192.168.30.202.1.1'
    plc = pyads.Connection(remote_ads, pyads.PORT_TC3PLC1, remote_ip)
    namespace = "KrogstrupMBE2.temperatureController[4].temperaturePV"
    time_vector = [np.zeros(buffer_size), np.ones(buffer_size)]
    time_vector[0] = np.arange(0, time_duration, 1/Fs)
    plc.open()
    time_vector[1] *= plc.read_by_name(
        namespace,
        pyads.PLCTYPE_LREAL)
    plc.close()
    
    
    delay_samples = 100
    delay_buffer = np.ones(delay_samples) * time_vector[1][0]
    fir_rls = fir_adaptive_rls(200)
    time = np.arange(0, time_duration, 1/Fs)
    signal = np.zeros(buffer_size)
    predicted_values = np.zeros(buffer_size) + time_vector[1]
    prediction_error = np.zeros(buffer_size)
    f,ax = plt.subplots(3)
    l1, l2, l3 = ax[0].plot(
        time,
        signal,
        '--' ,
        time,
        predicted_values,
        time[delay_samples:delay_samples+fir_rls.N],
        fir_rls.x)
    ax[0].set_xlim(0,time_duration)
    ax[0].legend((l1, l2, l3),('signal', 'predicted signal', 'training_data'))
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('')
    ax[0].axis('tight')
    l4, = ax[1].plot(time, prediction_error)
    ax[1].set_xlim(0,time_duration)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('prediction error [dB]')
    ax[1].axis('tight')
    l5, = ax[2].plot(fir_rls.b)
    ax[2].set_xlabel('Filter value')
    ax[2].axis('tight')
    plt.pause(0.01)
    plt.tight_layout()
    
    while True:
        shift_list(time_vector[1])
        plc.open()
        # Read new value
        time_vector[1][0] = plc.read_by_name(
            namespace,
            pyads.PLCTYPE_LREAL)
        plc.close()
        
        shift_list(delay_buffer)
        delay_buffer[0] = time_vector[1][0]
        
        # update adaptive filter
        b,e,y = fir_rls.update(x=delay_buffer[-1], d=time_vector[1][0])
        shift_list(predicted_values)
        predicted_values[0] = y
        shift_list(prediction_error)
        prediction_error[0] = 20*np.log10(abs(e))
        # print("b: {},\ne: {}\n--------".format(b,e))
        l1.set_ydata(time_vector[1])
        l2.set_ydata(predicted_values)
        l3.set_xdata(time[delay_samples-1:delay_samples-1+fir_rls.N])
        l3.set_ydata(fir_rls.x)
        ax[0].set_ylim(min(predicted_values),max(predicted_values))

        l4.set_ydata(prediction_error)
        ax[1].set_ylim(min(prediction_error),max(prediction_error))
        
        l5.set_ydata(b)
        ax[2].set_ylim(min(b),max(b))
        plt.pause(0.01)
        sleep(1/Fs)            

if __name__ == '__main__':
    main()