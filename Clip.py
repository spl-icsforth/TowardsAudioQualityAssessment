
import scipy
#import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from random import shuffle
import librosa
import librosa.display
from librosa.feature import melspectrogram
import resampy

frame_length_secs = 1

def gen_split_overlap(seq, size, overlap,perm=False):  
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')
    
    if perm==True:
        shuffle(seq)
        
    for i in range(0, len(seq) - overlap, size - overlap):            
        yield seq[i:i + size]
        
def ScaleData(data):
    eps = 1e-10
    mean = np.mean(data)
    std = np.std(data)
    data = ((data-mean)/(std+eps)).astype('float32')
    return data
        
class Clip :
    def __init__(self,path,data):
        self.samples=[]
        self.Fs = sf.SoundFile(path).samplerate 
        self.duration = data["duration"]
        self.seq_generator = 0
        self.CurrentFrame = 0
        self.time_spec = 0
        self.freq_spec = 0
        self.CurrentSpectrogram = []
        
    #converts stereo to mono by adding the two (or more) channels,aditionaly
    #scales samples to [-1,1]
    def Stereo2mono(self):
        if self.samples.ndim > 1:#if chanels are more than one
            #devide every channel with it's own rms
            for ch in range(self.samples.ndim):
                rms = np.sqrt(np.mean(np.square(self.samples[:,ch]), 0))
                self.samples[:,ch] = self.samples[:,ch] * 0.05 / rms
        samples_sum = self.samples
        
        if self.samples.ndim > 1:
            samples_sum = np.sum(self.samples,axis = 1)
            self.channels = 1

        self.samples = samples_sum / np.sqrt(self.samples.ndim)
        first_nonzero = next(
            (i for i, x in enumerate(self.samples) if x),
            None) 
        self.first_nonzero = first_nonzero
        #ignore first entries that are zero (this happens a lot to UGRs for 
        # some reason and should be avoided because it biases the network)
        self.samples = self.samples[first_nonzero:]
        samples_sum = 0
            
    #handles multiple channels too
    def CalcFrameSpectrogram(self,plot = False,spec_mode = 'hz'): 
        fft_size = 2048
        hop_length = 512#fft_size//8
        self.CurrentSpectrogram = []#initialize for every new frame!!!
        if self.CurrentFrame.ndim == 1:
            #reshape so that it has dimension (dim,1)
            self.CurrentFrame = np.reshape(
                self.CurrentFrame,
                (len(self.CurrentFrame),1))
                       
        mels_keep = 119 #128 is equal to keeping all the coefficients
        for ch in range(self.CurrentFrame.shape[1]):#for every channel 
            Spectrogram = melspectrogram(
                self.CurrentFrame[:,ch],
                sr = self.Fs,
                n_fft = fft_size,
                hop_length=hop_length,power=2).astype(
                    np.float32)[0:mels_keep,:]
            self.CurrentSpectrogram.append(Spectrogram)
            
        if plot == True:
            plt.figure(figsize = (10, 4))
            librosa.display.specshow(
                librosa.power_to_db(Spectrogram,ref = np.max),
                sr=self.Fs,hop_length=hop_length,x_axis = 'time',
                y_axis=spec_mode,fmax=self.Fs/2,cmap='RdBu')
                    #plt.ylim([0, freqs[117]])
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            plt.show()
        
    def Resample(self,Fs_ref):
        if self.Fs != Fs_ref:
            self.samples = resampy.resample(self.samples, self.Fs, Fs_ref)
            self.Fs = Fs_ref
            
    def StandardizeSpec(self,ch):
        #apply zero mean unit variance per time bin 
        for ch in range(len(self.CurrentSpectrogram)): 
            for time_bin,freq_vec in enumerate(self.CurrentSpectrogram[ch].T):
                self.CurrentSpectrogram[ch][:,time_bin] = ScaleData(
                    self.CurrentSpectrogram[ch][:,time_bin])
    
    def SaveClip(self):
        sf.write("saved_clip.wav",self.samples,self.Fs)
        
    def ResetFrames(self):#reset frames for next ugr
        self.CurrentFrame=0
        self.CurrentSpectrogram=[]
        
    def CreateClipGenerator(self):   
        frame_size=int(frame_length_secs*self.Fs)

        overlap=0
        #generator of sub sequence
        self.seq_generator=gen_split_overlap(self.samples,frame_size, overlap)
        self.samples=[] 


class TestClip(Clip):
    # takes path to file directly, it's not matched with other clips (used to 
    # test new-unknown quality files)
    def __init__(self,data_path):
        with sf.SoundFile(data_path) as f:
            self.Fs=f.samplerate
            self.CurrentFrame=None
            print("This clip was initially sampled at : ",self.Fs)
            self.samples=f.read(-1)
            self.channels=self.samples.ndim
            f.close()#close file
        self.Stereo2mono()
        self.Resample(44100)
        self.CreateClipGenerator()
        
   