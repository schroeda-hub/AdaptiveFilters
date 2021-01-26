import numpy as np

class fir(object):
    # G = a/b
    def __init__(self, b):
        """Init

        Args:
            b (list): numerator of a linear filter
        """
        self.b = b
        self.N = len(b)
        self.a = [0] * self.N
        self.a[1] = 1
        self.impulsAnswer = self.calc_impz()
    
    def calc_impz(self):
        """Calculate the impulse answer

        Returns:
            str: String with impulse answer in z-domain
        """
        impz = ""
        for i in range(self.N):
            impz += "+{}z^(-{})".format(self.b[i], i)
        return impz
            
    def filter(self, signal):
        """Filter a signal using this linear filter

        Args:
            signal (np.array): Array of the input signal

        Returns:
            np.array: Output array of filtered input signal
        """
        output = np.zeros(signal.shape)
        for i in range(self.N, signal.size):
            output[i] = signal[i-self.N:i] @ self.b
        return output