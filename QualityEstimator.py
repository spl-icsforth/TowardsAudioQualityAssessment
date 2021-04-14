
from Clip import TestClip
from Clip import frame_length_secs
import numpy as np
from tensorflow import keras
import datetime
import tensorflow as tf

def pure_loss(y_true, y_pred):#data only loss without regularization
    return tf.keras.backend.binary_crossentropy(y_true, y_pred)

def ConvertTo3d(array):
    return np.reshape(array,[1,array.shape[0],array.shape[1]])

def Zeropad(clip,pad_length):
   
    if clip.CurrentFrame.ndim > 1:
        #initialize first column of clip frames
        clip_frame_temp = np.empty(
            clip.CurrentFrame.shape[0] + pad_length)
        for ch in range(clip.CurrentFrame.shape[1]):
            clip_frame_temp = np.column_stack(
                (clip_frame_temp,np.pad(
                    clip.CurrentFrame[:,ch],(0,pad_length),'constant')))
        clip.CurrentFrame = clip_frame_temp[:,1:] #discard first column
        clip_frame_temp = []
    else:#if there is only one channel in this clip
        clip.CurrentFrame = np.pad(clip.CurrentFrame,(0,pad_length),'constant')


class QualityEstimator():
    def __init__(self,model_path):
        self.test_clip = None
        self.power = None
        self.n_channels = None
        self.worker_finished = False
        self.model=keras.models.load_model(
            model_path,
            custom_objects={'pure_loss': pure_loss})

        self.clip_paths = [] 
        self.quality_list = []
       
    def GetPredictions(self,test_clip_path):
         #load data
         test_clip = TestClip(test_clip_path)
         #power=np.mean(np.power(self.test_clip.samples,2))
         n_channels = test_clip.channels
         # print("channels are ",n_channels)
         predictions=[[]for i in range(n_channels)]
         # iterate frame generators
         for frame_idx,test_frame in enumerate(test_clip.seq_generator):
                test_clip.CurrentFrame_idx = frame_idx
                test_clip.CurrentFrame = test_frame#update current frames
                #zeropad if length is not the desired one
                if frame_length_secs * test_clip.Fs-len(test_frame) != 0:
                  Zeropad(
                      test_clip,
                      frame_length_secs * test_clip.Fs - len(test_frame))
                test_clip.CalcFrameSpectrogram(plot = False)
                CurrentSpec = test_clip.CurrentSpectrogram

                for ch in range(n_channels):    
                    output = self.model.predict(
                    ConvertTo3d(CurrentSpec[ch].T))
                    predictions[ch].extend(output)
                CurrentSpec = []
         test_clip.seq_generator = []
         test_clip.CurrentFrame = [] #free memory
         return predictions
     
    def GetClipScore(self,work_q):
        while not self.worker_finished:
            try:
                test_clip_path=work_q.get(timeout = 0.1)
            except:
                continue
            predictions = self.GetPredictions(test_clip_path)
            predictions = np.asanyarray(predictions)
            print(predictions)
            
            #average of scores over all frames
            quality = np.mean(predictions[0][:,1])
            time_steps = []
            for i in range(len(predictions[0])):
                #this array maps the prediction array with the corresponding 
                #time in the clip
                time_steps.append(
                    str(datetime.timedelta(seconds=i*frame_length_secs)))
            self.quality_list.append(quality)
            self.clip_paths.append(test_clip_path)
            work_q.task_done()       
        
    def Reset(self):
        self.worker_finished = False
        self.clip_paths = [] #
        self.quality_list = []
       