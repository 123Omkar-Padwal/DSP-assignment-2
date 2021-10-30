import numpy as np
import matplotlib.pyplot as plt

'''High Pass Filter'''

def highpass(coefficients, value):
    ntaps=1000
    buffer=np.zeros(ntaps)
    buffer=np.roll(buffer,1)
    buffer[0]=value
    return np.inner(buffer,coefficients)

def dofilter(coefficients, v):
    bufferlength=1000
    buffer=list(np.zeros(bufferlength-1))
    buffer.append(v)
    output = 0
    for i in range(bufferlength):
        output += coefficients[i] * buffer[(bufferlength - 1) - i]
    buffer = buffer[1:]
    return output

data=np.loadtxt('D:/Downloads/ecg-1.dat')
fs=999

yfft=np.fft.fft(data[:,1])
xf=np.linspace(0,fs,len(yfft))
plt.figure(1)
plt.plot(np.abs(yfft))
plt.show()
cutoff=500
ntaps=1000
k1=int((cutoff-500)/fs*ntaps)
k2=int((cutoff+500)/fs*ntaps)
signal=np.ones(ntaps)
signal[0:60]=0
inv_freq=np.real(np.fft.ifft(signal))
impulse_response=np.zeros(ntaps)
impulse_response[0:int(ntaps/2)]=inv_freq[int(ntaps/2):ntaps]
impulse_response[int(ntaps/2):ntaps]=inv_freq[0:int(ntaps/2)]
coeff=impulse_response*np.hamming(ntaps)
taps = np.linspace(0, fs, ntaps)

filter_result=[]
for i, value in enumerate(data[:,1]):
    filter_result.append(highpass(coeff, value))
    
filter_result1=[]
for i, value in enumerate(data[:,1]):
    filter_result1.append(highpass(coeff, value))
    
plt.figure(2)
plt.plot(filter_result)
plt.title('Filtered')
plt.show()
plt.figure(3)
plt.plot(data[:,1])
plt.show()

plt.figure(4)
plt.plot(taps, signal)
plt.title("Filter signal")
plt.show()

plt.figure(5)
plt.plot(np.abs(np.real(np.fft.fft(filter_result))))
plt.show()

plt.figure(6)
plt.plot(filter_result)
plt.title('do Filtered')
plt.show()
#duration=len(yfft)/999
#freq=1/duration
