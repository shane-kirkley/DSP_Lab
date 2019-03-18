import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import lab04_funcs as lab4

TEST_LOW = True
TEST_HIGH = True

# returns max error between original data and synthesized data
def synth_error(data, synth):
    data = data[:synth.shape[0]]          # trim data to match synth
    diff = abs(data[:-481] - synth[481:]) # delay found experimentally (expected 512)
    return np.max(diff)


fs, data = wavfile.read(sys.argv[1])
data = data[:fs*5] # trim to 5 seconds
coeffs = lab4.pqmf(data)

# removed low bands:
if (TEST_LOW):
    err = np.zeros(12)
    for i in range(12):
        thebands = np.ones(32)
        thebands[:i] = 0
        synth = lab4.ipqmf_bands(coeffs, thebands).flatten()
        err[i] = synth_error(data, synth)

    plt.bar(range(12),err)
    plt.title("Error from ommiting low frequency sub-bands from %s" % sys.argv[1])
    plt.ylabel("Absolute Maximum Error")
    plt.xlabel("Low bands removed")
    plt.show()

# removed high bands:
if (TEST_HIGH):
    num_bands = 31 # number of bands to test removal of
    err = np.zeros(num_bands)
    for i in range(1, num_bands):
        thebands = np.zeros(32)
        thebands[:-i] = 1
        synth = lab4.ipqmf_bands(coeffs, thebands).flatten()
        err[i] = synth_error(data, synth)

    plt.bar(range(num_bands),err)
    plt.title("Error from ommiting high frequency sub-bands from %s" % sys.argv[1])
    plt.ylabel("Absolute Maximum Error")
    plt.xlabel("High bands removed")
    plt.show()

