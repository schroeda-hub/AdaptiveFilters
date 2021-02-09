import numpy as np
from plot_freqz import plot_freqz
from plot_zplane import zplane

class fir(object):
    # G = a/b
    def __init__(self, b):
        """Init

        Args:
            b (list): numerator of a linear filter
        """
        self.set_b(b)
        
    def set_b(self, b):
        self.b = b
        self.N = len(b)
        self.impulsAnswer = self.calc_impz()
        self.x = np.zeros(self.N)
        
    def _update_x(self, x:float):
        for i in range(self.x.size-1,0,-1):
            self.x[i] = self.x[i-1]
            # print("{} -> {}".format(i-1,i))
            pass
        self.x[0] = x
        
    def plot_freqz(self):
        plot_freqz(b=self.b)
    
    def plot_zplane(self):
        a = np.zeros(self.N)
        a[0] = 1
        zplane(b=self.b, a=a)
    
    def calc_impz(self):
        """Calculate the impulse answer

        Returns:
            str: String with impulse answer in z-domain
        """
        impz = ""
        for i in range(self.N):
            impz += "+{}z^(-{})".format(self.b[i], i)
        return impz
            
    def filter_signal(self, signal):
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
    
    def filter(self, signal):
        self._update_x(signal)
        return self.x.T @ self.b
    
    def angle_to_complex(self, angle):
        return np.exp(1j*angle*2*np.pi/360)
    
    def calc_from_zPlane(self, coordinates:list):
        ret_val = np.poly(coordinates)
        return ret_val

def _test():
    test_fir = fir([])
    zeros_location = [
        1, 
        0+1j, 0-1j,
        test_fir.angle_to_complex(10), test_fir.angle_to_complex(-10),
        test_fir.angle_to_complex(20), test_fir.angle_to_complex(-20),
        test_fir.angle_to_complex(30), test_fir.angle_to_complex(-30)
    ]
    test_fir.set_b(test_fir.calc_from_zPlane(zeros_location))
    test_fir.plot_freqz()
    test_fir.plot_zplane()
    pass
    
if __name__ == '__main__':
    _test()