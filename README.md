# Thesis title
hand gesture & sound recognition (Internet Of Things)
### Description
The aim of this project is to control the light of the lamp using the hand gesture & sound recognition technique.
### Example of use
For example, if you show the number one to the camera with your hand, the light of the lamp will increase and if you show the number two, it will decrease.
You can also control the light of the lamp by saying light up and light down.
### Table of content
* [Sound recognition part](#sound-recognition-part)
* [Hand gesture part](#hand-gesture-part)
### Libraries needed
* [cv2](#cv2)
* [numpy](#numpy)                                   , For dealing with matrices
* [python_speech_features](#python_speech_features) , For extracting audio features
* [os](#os)                                         , For dealing with files
* [librosa](#librosa)                               , For loading and resampling wave files
* [random](#random)                                 , For shuffling data
* [matplotlib](#matplotlib)                         , For graphing things
* [tensorflow.keras](#tensorflow.keras)             , To build our neural networks and akso for converting our model to TF Lite model
* [](#)
* [](#)
* [](#)
* [](#)
* [](#)
* [](#)
* [](#)
### Hardware Requirements
Raspberry pi, camera module, some LEDs, USB microphone


# Sound Recognition
### Platform
* [Python](#Python)
* [Google Colab](#Google-Colab)
### Why colab?
You will quickly learn and use Google Colab if you know and have used Jupyter notebook before. Colab is basically a free Jupyter notebook environment running wholly in the cloud. Most importantly, Colab does not require a setup, plus the notebooks that you will create can be simultaneously edited by your team members – in a similar manner you edit documents in Google Docs. The greatest advantage is that Colab supports most popular machine learning libraries which can be easily loaded in your notebook.
### What Colab Offers You?
* [Write and execute code in Python](#Write-and-execute-code-in-Python)
* [Create/Upload/Share notebooks](#Create/Upload/Share-notebooks)
* [Import/Save notebooks from/to Google Drive](#Import/Save-notebooks-from/to-Google-Drive)
* [Import/Publish notebooks from GitHub](#Import/Publish-notebooks-from-GitHub)
* [Import external datasets ](#Import-external-datasets )
* [Integrate PyTorch, TensorFlow, Keras, OpenCV](#Integrate-PyTorch,-TensorFlow,-Keras,-OpenCV)
* [Free Cloud service with free GPU](#Free-Cloud-service-with-free-GPU)
### How do you use Colab?
To use Colaboratory, you must have a Google account and then access Colaboratory using your account. Otherwise, most of the Colaboratory features won’t work.
### How to Connect Google Colab with Google Drive
Run the following script in colab shell.
```
from google.colab import drive
drive.mount('/content/drive')
```
If you run, it gives you a link, Go to the mentioned link, Copy the authorization code of your account, finally paste the authorization code into the output shell and press enter
### How to import and export datasets in google colab?
First of all, upload your data to your google drive then, by running the following script you transfer your dataset from google drive to google colab :
```
!cp /content/drive/dataSet.zip /home/
```
After that you can make a directory in google colab by following script :
```
!mkdir /home/dataSetForVoice
```
Finally, unzip your folder by following script:
```
!unzip /home/dataSet.zip -d /home/dataSetForVoice
```
### Import labraries
```
from os import listdir  
from os.path import isdir,join   
import librosa        
import random          
import numpy as np   
import matplotlib.pyplot as plt
import python_speech_features  
```
If you get this error :  No module named 'python_speech_features', run the following script :
```
!pip install python_speech_features
```
### Dataset path and view possible targets
we'll use join to construct the path to our dataset.
```
dataset_path='/home/dataSetForVoice/dataSet'
all_targets=[name for name in listdir(dataset_path) if isdir(join(dataset_path,name))]
print(all_targets)
```   
['zero', 'up', 'wow', 'bed', 'down', 'visual', 'forward', 'happy', 'go', 'one', 'two', 'no', 'sheila', 'bird', 'yes', 'follow', 'five', 'left', 'six', 'house', 'eight', 'on', 'right', 'backward', 'cat', '_background_noise_', 'three', '_', 'seven', 'four', 'nine', 'learn', 'dog', 'stop', 'marvin', 'off', 'tree']

### Settings
Let's set a few parameters for the rest of this script, we'll want to create features for all of the target words even if we just pick out one target word later, at the end of this script we'll have a collection of features essentionally matrices that resemble images that store into an NPC file. In practice you'd want to use all the data you have avialable. however it can take hours to extract features and train using 100 thousand samples so I find it much easier to work with a random subset of data, I'll use 80%, this should only be used for initial prototypes. We also want to set aside 10% of our data for cross-validation and 10% for testing. The wav files are recorded with a 16 kilohertz sampling rate we'll be able to get our final model to run faster if we can use a lower sampling rate like 8 kilohertz. We'll set the number of mell frequency septal coefficients(num_mfcc) to 16 and the length of these mfcc's to 16
```
target_list=all_targets
feature_sets_file='all_targets_mfcc_sets.npz'
perc_keep_samples=0.8  #1.0 is keep all samples
val_ratio=0.1
test_ratio=0.1
sample_rate=8000
num_mfcc=16
len_mfcc=16
```
### Create list of filenames along with ground truth vector (y)
next we're going to create a list of all the file names with their full path. This will allow us to load each one and extract the features automatically, in addition we want to create a y array, this array holds the ground truth or actual values. Since this is a supervised learning project where we classify my signals will need the labels for the signals during the training step, we can arbitrarily assign values but they should be consistent. We'll assign number to the words in alphabetical order, zero is zero, up is one, wow is two and so on
```
filenames=[]
y=[]
for index,target in enumerate(target_list):
  print(join(dataset_path,target))
  filenames.append(listdir(join(dataset_path,target)))
  y.append(np.ones(len(filenames[index]))*index)
```
### Check ground truth Y vector
It's a collection of arrays and each array is simply the number we assigned to the target word, so there are 3728 zeroes in this first array array([0., 0., 0., ..., 0., 0., 0.]) which correspond to 3728 samples of people saying the word zero, similarly there are 3941 ones in the next array which correspond to the samples of the word up
```
print(y)
for item in y:
  print(len(item))
```
[array([0., 0., 0., ..., 0., 0., 0.]), array([1., 1., 1., ..., 1., 1., 1.]), array([2., 2., 2., ..., 2., 2., 2.]), array([3., 3., 3., ..., 3., 3., 3.]), array([4., 4., 4., ..., 4., 4., 4.]), array([5., 5., 5., ..., 5., 5., 5.]), array([6., 6., 6., ..., 6., 6., 6.]), array([7., 7., 7., ..., 7., 7., 7.]), array([8., 8., 8., ..., 8., 8., 8.]), array([9., 9., 9., ..., 9., 9., 9.]), array([10., 10., 10., ..., 10., 10., 10.]), array([11., 11., 11., ..., 11., 11., 11.]), array([12., 12., 12., ..., 12., 12., 12.]), array([13., 13., 13., ..., 13., 13., 13.]), array([14., 14., 14., ..., 14., 14., 14.]), array([15., 15., 15., ..., 15., 15., 15.]), array([16., 16., 16., ..., 16., 16., 16.]), array([17., 17., 17., ..., 17., 17., 17.]), array([18., 18., 18., ..., 18., 18., 18.]), array([19., 19., 19., ..., 19., 19., 19.]), array([20., 20., 20., ..., 20., 20., 20.]), array([21., 21., 21., ..., 21., 21., 21.]), array([22., 22., 22., ..., 22., 22., 22.]), array([23., 23., 23., ..., 23., 23., 23.]), array([24., 24., 24., ..., 24., 24., 24.]), array([25., 25., 25., ..., 25., 25., 25.]), array([], dtype=float64), array([27., 27., 27., ..., 27., 27., 27.]), array([28., 28., 28., ..., 28., 28., 28.]), array([29., 29., 29., ..., 29., 29., 29.]), array([30., 30., 30., ..., 30., 30., 30.]), array([31., 31., 31., ..., 31., 31., 31.]), array([32., 32., 32., ..., 32., 32., 32.]), array([33., 33., 33., ..., 33., 33., 33.]), array([34., 34., 34., ..., 34., 34., 34.]), array([35., 35., 35., ..., 35., 35., 35.])]
4052
3723
2123
2014
3917
1592
1557
2054
3880
...
### Flatten filename and y vectors
We then flatten these arrays, so they're just one long list rather than a collection of arrays
```
filenames=[item for sublist in filenames for item in sublist]
y=[item for sublist in y for item in sublist]
```
### Associate filenames with true output and shuffle
We'll use the python zip command to link each file name with it's associated y value, we then randomly shuffle the filenames and notice that the y values stayed a link to the individual names , we can then unzip the 2 lists to separate filenames and y but there ordering will remain.
```
filenames_y=list(zip(filenames,y))
random.shuffle(filenames_y)
filenames,y=zip(*filenames_y)
```
Finally we will take the first 10% of the data and set it aside to be our cross-validation set. This will be useful in testing the model to see how well it performs during training then we set aside another 10% as test data. We leave the rest of the data alone to be used as training data
### Break dataset apart into train, validation , and test sets
```
filenames_val=filenames[:val_set_size]
filenames_test=filenames[val_set_size:(val_set_size+test_set_size)]
filenames_train=filenames[(val_set_size+test_set_size):]
```
### Break y apart into train, validation, and test sets
```
y_orig_val=y[:val_set_size]
y_orig_test=y[val_set_size:(val_set_size+test_set_size)]
y_orig_train=y[(val_set_size+test_set_size):]
```
### Extract features
We're now ready to extract features from these wav files. Transforming audio into the mell frequency SEP coefficients seems to be very popular in machine learning for speech recognition. To calculate the MFCC's we take a small time slice of our audio waveform and compute the fast fourier transform, this gives us the amount power at each frequency of that time slice, then we apply a set of filters to that fast fourier transform spectrum, note that these filters are spaced in such a way to represent how humans perceive sound. Generally the filters are linearly spaced below one kilohertz and log rhythmically spaced above one kilohertz. We then sum up the power found in each filter to get a number representing the energy under that filter. Note that most implementations of MFCC used 26 filters for voice from there you'll want to compute the log of each value in the vector, after that we compute the discrete cosine transform of the 26 log filter bank energies. The DCT works much like the fourier transform but operates on real valued signals and does a better job of emphasizing the low frequency components, if you start with 26 elements in the filter bank energies, you should end up with 26 separate coefficients, the lower coefficients contain information about the general shape of the audio spectrum in that time slice as you go up in the coefficients you start to get into the finer details of the audio spectrum. For speech analysis you normally want to throw away the 0th element and anything after element 13, above the 13'th element is usually noise and audio artifacts that don't correlate to speech much.
![featureExtraction1](https://user-images.githubusercontent.com/50530596/123776723-d6e39100-d8cf-11eb-8b7f-ce5701452386.png)
