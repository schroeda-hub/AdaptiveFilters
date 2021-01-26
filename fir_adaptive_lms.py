import numpy as np

class fir_adaptive_lms(object):
    def __init__(self, N: int):
        """Init of a lms adaptive filter.

        Args:
            N (int): Order of filter
        """
        self.N = N
        
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
