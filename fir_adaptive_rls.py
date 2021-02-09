import numpy as np
from plot_freqz import plot_freqz
from plot_zplane import zplane

class fir_adaptive_rls(object):
    def __init__(self, N: int, p=0.95):
        """Init of a lms adaptive filter.

        Args:
            N (int): Order of filter
        """
        self.N = N
        self.x = np.zeros(self.N)
        self.b = np.zeros(self.N)
        self.e = 0.0
        self.z = np.zeros(self.N)
        self.y = 0.0
        self.R_inv = 1e9 * np.identity(self.N)
        self.p = p
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
    
    def train(self, ref_input, desired_input, p):
        """Train adaptive fir filter RLS

        Args:
            ref_input (list/np.array): Input to fir filter
            desired_input (list/np.array): Desired output signal of the fir filter
            p (float): Forgetfacor 0<p<=1 (typ: 0.95<p<=1)

        Returns:
            b: nparray with filter numerators
            e: nparray with error
        """
        # Init
        np.seterr(all='raise')
        assert len(ref_input) == len(desired_input), "Input signals not equal in length" 
        assert p != 0
        iterations = len(ref_input)
        R_inv = np.zeros([iterations, self.N, self.N])
        R_inv[0:self.N] = 1e9 * np.identity(self.N)
        b = np.zeros((iterations, self.N))
        e = np.zeros((desired_input.shape))
        y = np.zeros(iterations)
        z = np.zeros((iterations, self.N))
        for k in range(self.N, iterations):
            y[k] = ref_input[k-self.N:k].T @ b[k-1]
            e[k] = desired_input[k] - y[k]
            try:
                numerator = R_inv[k-1] @ ref_input[k-self.N:k]
                denominator = (p + ref_input[k-self.N:k].T @ (R_inv[k-1] @ ref_input[k-self.N:k]))
                if denominator >= 1e307:
                    print(b[k-1])
                    break
                assert denominator != 0
                z[k] =  numerator/denominator
            except OverflowError as oe:
                print("Overflow error", oe)
            b[k] = b[k-1] + e[k] * z[k]
            R_inv[k] = 1/p * (R_inv[k-1] - z[k] * ref_input[k-self.N:k].T * R_inv[k-1])
            pass
        np.seterr(all='warn')
        return b[self.N:], e[self.N:], y[self.N:]

    def _update_x(self, x:float):
        for i in range(self.x.size-1,0,-1):
            self.x[i] = self.x[i-1]
            # print("{} -> {}".format(i-1,i))
            pass
        self.x[0] = x

    def update(self, x:float, d:float):
        self._update_x(x)
        self.y = self.x.T @ self.b
        self.e = d - self.y
        try:
            numerator = self.R_inv @ self.x
            denominator = (self.p + self.x.T @ (self.R_inv @ self.x))
            if denominator >= 1e307:
                print(self.b)
                return
            assert denominator != 0
            self.z = numerator/denominator            
            pass
        except OverflowError as oe:
            print("Overflow error", oe)
            pass
        self.b = self.b + self.e*self.z
        self.R_inv = 1/self.p * (self.R_inv - self.z * self.x * self.R_inv)
        return self.b, self.e, self.y
        
    def predict(self, x:float):
        self._update_x(x)
        self.y = self.x.T @ self.b
        return self.y
    
def _test():
    test_filter = fir_adaptive_rls(10)
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
