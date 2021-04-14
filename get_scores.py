
from QualityEstimator import QualityEstimator
import os
import numpy as np
from queue import Queue
from threading import Thread 
import time
from pathlib import Path
import argparse

def SaveInformationToText(wav_path_list,wav_quality_list):
    text_path = './results.txt'
    
    wav_names = [os.path.basename(full_path) for full_path in wav_path_list]
    results = np.zeros(
        len(wav_names), dtype=[('names', 'U107'),
                               ('scores', float)])
    results['names'] = wav_names
    results['scores'] = wav_quality_list
    np.savetxt(text_path,
               results,
               fmt = "%10.200s %10.2f", 
               delimiter = ' ', 
               header = ('WavName | Score'))

def GetClipsQuality(path_list,quality_estimator,N_threads = 4):
	work_q = Queue()#no limit at the que
	worker_list = []
	for i in range(N_threads):
	    worker = Thread(target = quality_estimator.GetClipScore,
                     args = (work_q,))
	    worker.setDaemon(True)
	    worker.start()
	    worker_list.append(worker)
	
	for i in range(len(path_list)):
	    work_q.put(path_list[i])    
	work_q.join() #block    

	quality_estimator.worker_finished = True
	for i in range(N_threads):
	    worker_list[i].join()
	#returns tuple of paths and corresponding quality
	return(quality_estimator.clip_paths,quality_estimator.quality_list) 

def main():
    parser = argparse.ArgumentParser(
        description='batch_processor', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('pathIN', help = 'folder containing audio data')    
    parser.add_argument('-t', '--nThreads', type=int, default = 4, 
                        help = 'maximum number of threads to use')    
      
    args = parser.parse_args()
    wav_folder = args.pathIN
    n_threads_to_use = args.nThreads
    print('Using ',n_threads_to_use,' threads')
    start_time = time.time()#Start counting time
    model_path = './weights.hdf5'
    quality_estimator = QualityEstimator(model_path)

    wav_path_list = [
        str(wav_path) for wav_path in Path(wav_folder).rglob('*.wav')]
    
    
    wav_path_list,wav_quality_list = GetClipsQuality(
        wav_path_list,
        quality_estimator,
        n_threads_to_use)
    # path lists should be updated because threading may change 
    # the order of paths
    
    SaveInformationToText(wav_path_list,wav_quality_list)
    elapsed_time = time.time() - start_time
    print("Elapsed time is",elapsed_time )

    
if __name__ == "__main__":
    main() 