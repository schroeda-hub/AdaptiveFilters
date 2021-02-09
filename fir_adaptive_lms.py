import numpy as np
from plot_freqz import plot_freqz
from plot_zplane import zplane

class fir_adaptive_lms(object):
    def __init__(self, N: int, mu=0.05):
        """Init of a lms adaptive filter.

        Args:
            N (int): Order of filter
        """
        self.N = N
        self.b = np.zeros(self.N)
        self.x = np.zeros(self.N)
        self.e = 0.0
        self.y = 0.0
        self.mu = mu
        pass
        
    def calc_impz(self, b):
        """Calculate the impulse answer

        Args:
            b (list): Numerator of filter

        Returns:
            str: String with impulse answer in z-domain
        """
        impz = ""
        N = len(b)
        for i in range(N):
            impz += "+{}z^(-{})".format(b[i], i)
        return impz
    
    def plot_freqz(self):
        plot_freqz(b=self.b)
    
    def plot_zplane(self):
        a = np.zeros(self.N)
        a[0] = 1
        zplane(b=self.b, a=a)
            
    def filter(self, b, signal):
        """Filter a signal using this linear filter

        Args:
            b (list): Numerator of filter
            signal (np.array): Array of the input signal

        Returns:
            np.array: Output array of filtered input signal
        """
        output = np.zeros(signal.shape)
        N = len(b)
        for i in range(N-1, signal.size):
            fir_sum = 0
            for n in range(N):
                fir_sum += signal[i-n] * b[n]
                pass
            output[i] = fir_sum
        return output
    
    def train(self, ref_input, desired_input, mu):
        """Train adaptive fir filter LMS

        Args:
            ref_input (list/np.array): Input to fir filter
            desired_input (list/np.array): Desired output signal of the fir filter
            mu (float): Stepsize of learning (Learnrate)

        Returns:
            b: nparray with filter numerators
            e: nparray with error
        """
        # Init
        assert len(ref_input) == len(desired_input), "Input signals not equal in length" 
        iterations = len(ref_input)
        n = 0
        b = np.zeros((iterations, self.N))
        e = np.ones((desired_input.shape)) * desired_input[0]
        y = [None] * iterations
        for n in range(self.N, iterations):
            # FIR of input
            y[n] = b[n] @ ref_input[n-self.N:n]
            # Fehlerbestimmung
            e[n] = desired_input[n] - y[n]
            # LMS Koeffizienten update
            if n+1 < iterations:                
                b[n+1] = b[n] + 2*mu*e[n]*ref_input[n-self.N:n]
                pass
            pass
        pass
        return b[self.N:], e[self.N:]
    
    def _update_x(self, x:float):
        for i in range(self.x.size-1,0,-1):
            self.x[i] = self.x[i-1]
        self.x[0] = x
        
    def predict(self, x:float):
        self._update_x(x)
        self.y = self.x.T @ self.b
        return self.y

    def update(self, x:float, d:float):
        self._update_x(x)
        self.y = self.x.T @ self.b
        self.e = d - self.y
        self.b = self.b + 2 * self.mu * self.e * self.x
        return self.b, self.e, self.y
    
def _test():
    test_filter = fir_adaptive_lms(10)
    time = np.linspace(0, 4*np.pi, int(3e3))
    signal = 5*np.sin(time)
    # signal += 0.4 * np.cos(time * 3)
    # signal += time % 4*np.pi
    mean = 0
    std = 1
    noise = np.random.normal(mean, std, size=time.size)
    signal += noise
    last_signal_value = signal[-1]
    signal = signal[:-1]
    delay_samples = 1
    predicted_values = np.zeros(signal.size)
    for i in range(signal.size-delay_samples):
        b,e,y = test_filter.update(signal[i], signal[i+delay_samples])
        predicted_values[i] = y
    print("b: {},\n e: {}, y: {}".format(b,e,y))
    print(test_filter.predict(last_signal_value))
    error = np.abs(predicted_values - signal)
    import matplotlib.pyplot as plt
    plt.plot(20*np.log10(error))
    plt.show()
    plt.plot(predicted_values)
    plt.show()
    plt.plot(signal)
    plt.show()
    pass

if __name__ == '__main__':
    _test()
