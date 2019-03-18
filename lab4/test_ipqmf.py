import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import lab04_funcs as lab4

fs, data = wavfile.read(sys.argv[1])
data = data[:fs*5] # trim to 5 seconds

coeffs = lab4.pqmf(data)

# plt.plot(coeffs.T.flatten(), linewidth=0.5)
# plt.show()

synth = lab4.ipqmf(coeffs).flatten()

plt.plot(data, linewidth=0.5)
plt.plot(synth, linewidth=0.5)
plt.xlim(0, 1024)
plt.show()

data = data[:synth.shape[0]] # trim data to match synth
diff = abs(data[:-481] - synth[481:]) # delay found experimentally (expected 512)
err = np.max(diff)
print("Absolute max error:")
print(err)
