import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wavio # save np.array to wav
from fir_adaptive_rls import fir_adaptive_rls
from fir_adaptive_lms import fir_adaptive_lms

np.seterr('raise')

def shift_list(list_, numbers=1):
    list_ = list_[:-1*numbers]
    prefix = np.zeros(numbers)
    return np.concatenate((prefix,list_))

def shift_list_front(list_, numbers=1):
    list_ = list_[numbers:]
    suffix = np.zeros(numbers)
    return np.concatenate((list_,suffix))

def _test_shift_list():
    test_input = np.array([1,2,3,4,5,6,7,8,9])
    test_output = np.array([0,0,1,2,3,4,5,6,7])
    list_output = shift_list(test_input,2)
    shift_list_front(test_input,2)
    for i in range(test_output.size):
        if float(test_output[i]) != float(list_output[i]):
            raise ValueError
    pass

_test_shift_list()

class plot_container:
    def __init__(self):
        self.packets = 10
        self.Fs = 44100
        self.chunk_size = 10*1024
        self.time = np.arange(0, self.packets * self.chunk_size/self.Fs, 1/self.Fs)
        self.values = np.zeros(self.time.size)
        pass

    def update(self, values):
        self.values = shift_list_front(self.values, self.chunk_size)
        self.values[-1*self.chunk_size:] = values

class FIR_container(fir_adaptive_rls):
    def __init__(self):
        super(FIR_container, self).__init__(20,p=0.95)
        self.packets = 10
        self.Fs = 44100
        self.chunk_size = 10*1024
        self.time = np.arange(0, self.packets * self.chunk_size/self.Fs, 1/self.Fs)
        self.delay_samples = 10
        self.delay_buffer = np.zeros(self.delay_samples)
        self.predicted_values = np.zeros(self.packets * self.chunk_size)
        self.error = np.zeros(self.packets * self.chunk_size)
        pass

    def update_chunk(self, values):
        for i in range(values.size):
            self.delay_buffer = shift_list(self.delay_buffer)
            self.delay_buffer[0] = values[i]
            b,e,y = self.update(x=self.delay_buffer[-1],d=values[i])
            self.predicted_values = shift_list(self.predicted_values)
            self.predicted_values[0] = y
            self.error = shift_list(self.error)
            self.error[0] = e
        pass


class Plot():
    def __init__(self):
        self.raw_data = plot_container()
        self.filtered_data = plot_container()
        self.ad_filter = FIR_container()
        self.f,self.ax = plt.subplots(3)
        self.l1, = self.ax[0].plot(self.raw_data.time, self.raw_data.values)
        self.l2, = self.ax[1].plot(self.ad_filter.time, self.ad_filter.predicted_values)
        self.l3, = self.ax[2].plot(self.ad_filter.time, self.ad_filter.error)
        self.ax[0].set_title("Raw Audio Signal")
        self.ax[1].set_title("Predicted signal")
        self.ax[2].set_title("With reduced noise")
        plt.pause(0.01)
        plt.tight_layout()

    def plot(self):
        self.l1.set_ydata(self.raw_data.values[::-1])
        self.ax[0].set_ylim(
            min(self.raw_data.values),
            max(self.raw_data.values))
        self.l2.set_ydata(self.ad_filter.predicted_values)
        self.ax[1].set_ylim(
            min(self.ad_filter.predicted_values),
            max(self.ad_filter.predicted_values))
        self.l3.set_ydata(self.ad_filter.error)
        self.ax[2].set_ylim(
            min(self.ad_filter.error),
            max(self.ad_filter.error))
        plt.pause(0.01)
        pass

plot = Plot()

class sawtooth:
    def __init__(self):
        self.sample = 0
        self.mod_number = 1024*10

    def get(self):
        self.sample += 1
        return self.sample%self.mod_number

class signal_creator_obj(object):
    def __init__(self, Fs:float):
        self.sample_number = -1
        self.Fs = Fs
        self.f = 0.001 # Hz
        self.noise_enable = True

    def _noise(self):
        mean = 0
        std = 1
        return np.random.normal(mean, std, size=1) * 0.001

    def get(self):
        self.sample_number += 1
        self.f += 0.001/(10+1024)
        return np.sin(self.sample_number/self.Fs * 2 * np.pi * self.f) + self._noise()

def main():
    FORMAT = pyaudio.paInt16 # We use 16bit format per sample
    CHANNELS = 1
    RATE = 44100 # Fs [Hz]
    CHUNK = 10*1024 # 1024bytes of data red from a buffer
    RECORD_SECONDS = 0.1
    WAVE_OUTPUT_FILENAME = "file.wav"
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True)

    sawtooth_gen = sawtooth()
    signal_creator = signal_creator_obj(20)

    chunk_read_count = 0
    keep_going = True
    while keep_going:
        try:
            if False:
                stream_data = stream.read(CHUNK)
                audio_data = np.frombuffer(stream_data, np.int16)
            else:
                audio_data = np.zeros(CHUNK)
                for i in range(CHUNK):
                    # audio_data[i] = sawtooth_gen.get()
                    audio_data[i] = signal_creator.get()
                if chunk_read_count > 9:
                    wavio.write("RawAudio.wav",  plot.raw_data.values, RATE, sampwidth=2)
                    wavio.write("ReducedNoise.wav",  plot.ad_filter.error, RATE, sampwidth=2)
                    keep_going = False
            chunk_read_count += 1
            plot.raw_data.update(audio_data)
            plot.ad_filter.update_chunk(audio_data)
            print("chunks read: {}".format(chunk_read_count))
            plot.plot()
            pass
        except KeyboardInterrupt:
            plot.raw_data.update(audio_data)
            plot.ad_filter.update_chunk(audio_data)
            wavio.write("RawAudio.wav",  plot.raw_data.values, RATE, sampwidth=2)
            wavio.write("ReducedNoise.wav",  plot.ad_filter.error, RATE, sampwidth=2)
            keep_going = False

if __name__ == '__main__':
    main()