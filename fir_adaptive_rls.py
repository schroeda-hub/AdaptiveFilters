import numpy as np

class fir_adaptive_rls(object):
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
