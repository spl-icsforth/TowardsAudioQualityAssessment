# Towards Blind Quality Assessment of Concert Audio Recordings Using Deep Neural Networks
 **Nikonas Simou¹³,Yannis Mastorakis¹,Nikolaos Stefanakis¹²**
> **¹** FORTH-ICS, Heraklion, Crete, Greece, GR-70013
> **²** Hellenic Mediterranean University, Department of Music Technology and Acoustics, Rethymno, Greece, GR-74100
> **³** University of Crete, Department of Computer Science, Heraklion, Greece,GR-70013

## What it does
Based on our work which was published at [ICASSP 2020](https://ieeexplore.ieee.org/document/9053356), this script uses a pretrained CNN architecture and provides an average score over non overlaping 1-second audio frames of the given `.wav` file. This score represents the probability that the given file is professionally recorded. For more information on how the score is extracted and how it colerates with audio quality please check our paper.

## Initial setup 
1) Extract the folder containing the source files in any directory of your hard drive.
2) Install [anaconda], by downloading the installer from the official website. (e.g. Anaconda3-2020.02-Windows-x86_64.exe).
After installation, "anaconda prompt" will show up in the program list.
3) Open anaconda prompt and navigate to the folder which contains the code. (e.g. by typing "cd D:/MyDocuments/TowardsAudioQualityAssessment/")
4) In Anaconda prompt type:
    ```sh
    pip install -r requirements.txt
    ```
    in order to install all required libraries.
    This might take some time but it is important for this procedure to be completed 
    successfully.
-----
## Running the algorithm on your data
The system should be ready to run by now. You can run the tool by calling "get_scores.py.py" function.
1) Open anaconda prompt and navigate to the source code.
2) Run the code by passing as argument the path of the folder which contains the recordings to be analyzed.
For example, if the path is "D:/music/concert_recordings/", then we can simply type (in anaconda prompt):
    ```sh
    python get_scores D:/music/concert_recordings/
    ```
* The path of the folder which contains the recordings is the only mandatory input argument.
* An additional parameter is:

	* **Number of threads** to be used. For example, if we want to engage 8 threads, we can type:
        ```sh
        python get_scores.py D:/music/concert_recordings/ -t 8
        ```
	        
## OUTPUT 
A `.txt` file will be exported which contains each `.wav` file name along with the corresponding score.	        


## License
MIT 

Copyright (c) 2021 **Nikonas Simou, Yannis Mastorakis, Nikolaos Stefanakis**

--------
### How to reference
If you find any of this library useful for your research, please give cite as:
Nikonas Simou; Yannis Mastorakis, Nikolaos Stefanakis; ["Towards Blind Quality Assessment of Concert Audio Recordings Using Deep Neural Networks"](https://ieeexplore.ieee.org/document/9053356) in ICASSP 2020.


#### Important notes:
1) Tool converts recordings to 44100Hz.

2) The tool also searches for `.wav` files in all subfolders of the given directory.

3) Avoid very lengthy recordings (>30 minutes) since this may cause a memory overload and crush the script.

4) The model published here is actually trained on a slightly smaller amount of data than the one used for our publication.



 [anaconda]: <https://anaconda.org/>
