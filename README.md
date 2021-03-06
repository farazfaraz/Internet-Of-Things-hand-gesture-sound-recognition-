# Thesis title
hand gesture & sound recognition (Internet Of Things)
### Description
The aim of this project is to control the light of the lamp using the hand gesture & sound recognition technique.
### Example of use
For example, if you show the number one to the camera with your hand, the light of the lamp will increase and if you show the number two, it will decrease.
You can also control the light of the lamp by saying light up and light down.
### Table of content
* [Connect your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-getting-started/3)
* [Sound recognition part](#sound-recognition-part)
* [Hand Gesture Recognition using Raspberry Pi and OpenCV](#Hand Gesture Recognition using Raspberry Pi and OpenCV)
### Libraries needed
* [OpenCV](#OpenCV)                                 , is used here for digital image processing. The most common applications of Digital Image Processing are object detection, Face Recognition, and people counter.
* [cv2](#cv2)
* [numpy](#numpy)                                   , For dealing with matrices
* [python_speech_features](#python_speech_features) , For extracting audio features
* [os](#os)                                         , For dealing with files
* [librosa](#librosa)                               , For loading and resampling wave files
* [random](#random)                                 , For shuffling data
* [matplotlib](#matplotlib)                         , For graphing things
* [tensorflow.keras](#tensorflow.keras)             , To build our neural networks and akso for converting our model to TF Lite model
* [lite](#lite)                                     , To convert our model to a tensorflow lite model
* [](#)
* [](#)
* [](#)
* [](#)
* [](#)
* [](#)
### Programs needed
* [Python](#Python)
* [SD Card Formatter :](https://www.sdcard.org/downloads/formatter/)
**It is strongly recommended to use the SD Memory Card Formatter to format SD/SDHC/SDXC Cards rather than using formatting tools provided with individual operating systems.**
* [Win32DiskImager](https://sourceforge.net/projects/win32diskimager/)
* [wireless network watcher](https://www.nirsoft.net/utils/wireless_network_watcher.html)
* [Putty](https://www.putty.org/)
* [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
* [Fritzing](https://fritzing.org/download/)
* [](#)
* [](#)
* [](#)
### Hardware Requirements
Raspberry pi 3 model B, camera module, some LEDs, USB microphone

# Connect your Raspberry Pi


# Sound Recognition (Offline)
### Platform
* [Python](#Python)
* [Google Colab](#Google-Colab)
### Why colab?
You will quickly learn and use Google Colab if you know and have used Jupyter notebook before. Colab is basically a free Jupyter notebook environment running wholly in the cloud. Most importantly, Colab does not require a setup, plus the notebooks that you will create can be simultaneously edited by your team members ??? in a similar manner you edit documents in Google Docs. The greatest advantage is that Colab supports most popular machine learning libraries which can be easily loaded in your notebook.
### What Colab Offers You?
* [Write and execute code in Python](#Write-and-execute-code-in-Python)
* [Create/Upload/Share notebooks](#Create/Upload/Share-notebooks)
* [Import/Save notebooks from/to Google Drive](#Import/Save-notebooks-from/to-Google-Drive)
* [Import/Publish notebooks from GitHub](#Import/Publish-notebooks-from-GitHub)
* [Import external datasets ](#Import-external-datasets )
* [Integrate PyTorch, TensorFlow, Keras, OpenCV](#Integrate-PyTorch,-TensorFlow,-Keras,-OpenCV)
* [Free Cloud service with free GPU](#Free-Cloud-service-with-free-GPU)
### How do you use Colab?
To use Colaboratory, you must have a Google account and then access Colaboratory using your account. Otherwise, most of the Colaboratory features won???t work.
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
from tensorflow.keras import layers,models
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

##### 4052

##### 3723

##### 2123

##### 2014

##### 3917

##### 1592

##### 1557

##### 2054

##### 3880

##### ...
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

So we've computed the MFCC's for the first time slice of our audio file, notice that we've flipped the vector around, so that the first element is on the bottom and the highest element is on the top, also after some testing I found that for the model I plan to train having 16 elements seems to work the best. Note that this might not be universally true, you might be able to get good or better accuracy with a different neural network and only 12 elements. The point is to keep trying and experimenting with things to find something that works, we then slide our window over a bit on the waveform and compute the EMFCC's from that time slice, keep going until we've obtained all the EMFCC's for the whole waveform. At this point we have a two-dimentional array of MFcc values, one way to view this matrix is as an image. The X-axis here is the time, each column of pixels correspond to one of the time slices we took, the Y-axis is the MFCC's, so there should be 16 total rows, the colors give us a re;ative representation of the value of each coefficient, the bottom row or 0th coefficient is dark because ther're all large negative values compared to the rest. So we will use a neural network that classifies images
![featureExtraction2](https://user-images.githubusercontent.com/50530596/123784479-c3d4bf00-d8d7-11eb-8f00-db935ed84651.png)
### Function: Create MFCC from given path
```
def calc_mfcc(path):
  #Load wavefile
  signal, fs=librosa.load(path,sr=sample_rate) #Resamples to 8000 samples per second

  #Create NFCC's from sound clip
  mfccs=python_speech_features.base.mfcc(signal,samplerate=fs,winlen=0.256,
                                         winstep=0.050,numcep=num_mfcc,nfilt=26,
                                         nfft=2048,preemph=0.0,ceplifter=0,
                                         appendEnergy=False,winfunc=np.hanning)
  return mfccs.transpose()
```
### TEST
Construct test set by computing MFCC of each wav file we'll take the first 500 samples from the training set and display the shape of their MFCC matrices, each audio file should produce 16 sets of 16 coefficients,as you can see a few of the audio files seem to have been corrupted or not fully one second long, if we count these up and divide by 500 we can conclude that about 10% of all the audio samples have this problem.
```
prob_cnt=0
x_test=[]
y_test=[]
for index, filename in enumerate(filenames_train):
  #stop after 500
  if index>=500:
    break
  #Create path from given filename and target item
  path=join(dataset_path,target_list[int(y_orig_train[index])],filename)

  #Create MFCCs
  mfccs=calc_mfcc(path)
  if mfccs.shape[1]==len_mfcc:
    x_test.append(mfccs)
    y_test.append(y_orig_train[index])
  else:
    print('Dropped: ',index,mfccs.shape)
    prob_cnt+=1
 ```
##### Dropped:  12 (16, 13)

##### Dropped:  17 (16, 7)

##### Dropped:  27 (16, 11)

##### Dropped:  33 (16, 11)

##### Dropped:  38 (16, 14)

##### Dropped:  47 (16, 13)

##### Dropped:  51 (16, 12)

##### Dropped:  55 (16, 14)

##### Dropped:  83 (16, 14)

##### Dropped:  100 (16, 12)

##### Dropped:  108 (16, 12)

##### Dropped:  118 (16, 7)
If the sample isn't quite long enough you can append values that look like data found within the sample , something that approximates silence or white noise. You can also just drop the sample completely which is easiest thing to do, since only about 10% of the samples are problematic for this dataset I'm just going to drop any of them that don't produce exactly 16 sets of coefficients, so we write another function that does exactly that, it makes sure the file ends with .wav, calculates the MFCC's and drops the sample and corresponding label from the Y vectors if it's not long enough, we then run that function on each of our training validation and test sets.
```
#Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files,in_y):
  prob_cnt=0
  out_x=[]
  out_y=[]
  for index,filename in enumerate(in_files):
    #Create path from given filename and target item
    path=join(dataset_path,target_list[int(in_y[index])],filename)
    #Check to make sure we're reading a .wav file
    if  not path.endswith('.wav'):
      continue

    #Create MFCCs
    mfccs=calc_mfcc(path)
    #Only keep MFCCs with given length
    if mfccs.shape[1]==len_mfcc:
      out_x.append(mfccs)
      out_y.append(in_y[index])
    else:
      print('Dropped: ',index,mfccs.shape)
      prob_cnt+=1
  return out_x,out_y,prob_cnt
  ```
### Create train, validation, and test sets
```
x_train, y_train, prob=extract_features(filenames_train,y_orig_train)
print('Removed percentage: ',prob/len(y_orig_train))
x_val, y_val, prob=extract_features(filenames_val,y_orig_val)
print('Removed percentage: ',prob/len(y_orig_val))
x_test, y_test, prob=extract_features(filenames_test,y_orig_test)
print('Removed percentage: ',prob/len(y_orig_test))
```
When it's done you can see that it removed about 10% of samples from the set.
Finally we use the numpy save Z function to store these massive arrays into an NPZ file, this will allow us to load our saved features and corresponding labels in a next step when we're ready to do the actual machine learning. To load the features we just call numpy dot load and give it the location of the file.
```
np.savez(feature_sets_file,x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,x_test=x_test,y_test=y_test)
```
### load features
Up to now we downloaded the Google Speech Commands dataset, read the individual files, and converted the raw audio clips into Mel Frequency Cepstral Coefficients (MFCCs). We also split these features into training, cross validation, and test sets. Because we saved these feature sets to a file, we can read that file from disk to begin our model training.
```
dataset_path='/home/dataSetForVoice/dataSet'
all_targets=[name for name in listdir(dataset_path) if isdir(join(dataset_path,name))]
all_targets.remove('_background_noise_')
print(all_targets)
```
##### ['forward', 'bed', 'zero', '_', 'house', 'left', 'down', 'yes', 'wow', 'seven', 'follow', 'stop', 'nine', 'happy', 'on', 'learn', 'five', 'dog', 'cat', 'off', 'four', 'one', 'visual', 'two', 'bird', 'tree', 'right', 'eight', 'up', 'no', 'three', 'marvin', 'go', 'six', 'sheila', 'backward']

### Settings
```
feature_sets_path='/home/features'
feature_sets_filename='/home/all_targets_mfcc_sets100.npz'
feature_sets=np.load(join(feature_sets_path,feature_sets_filename))
print(feature_sets.files)
```
##### ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']
### Assign feature sets
```
x_train=feature_sets['x_train']
y_train=feature_sets['y_train']
x_val=feature_sets['x_val']
y_val=feature_sets['y_val']
x_test=feature_sets['x_test']
y_test=feature_sets['y_test']
```
Look at tensor dimension We can see that the first dimension is the number of samples in that set and the other 2 dimensions are the number of coefficients and the number of sets of coefficients in each sample
```
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
```
##### (77463, 16, 16)
##### (9668, 16, 16)
##### (9725, 16, 16)
Peek at labels If we look at one of the sets of labels we can see that it's a collection of numbers that correspond to the different words
```
print(y_val)
```
##### [17. 11.  2. ...  8.  9. 27.]
### convolutional neural network
A convolutional neural network usually consists of two different parts, the first is the set of convolutional layers, these layers act to automatically learn and extract features from the image, these features are then passed to a fully connected neural network that attempts to classify the image based on the features provided, the first section of the convolutional set consists of three steps, the first step is the actual convolutional operation, it consists of moving a window across the whole as a sort of filter in order to extract some features such as detecting edges, this sliding window filter is known as a kernel and performs some math operations as it samples a set of pixels from the image, in this example the window is 2 by 2 pixels, the weights used in this kernel are different for every node and are updated automatically dyring training with.
### View the dimension of our input data
```
print(x_train.shape)
```
##### (77463, 16, 16)
Tensorflow expects tensors in 4 dimensions as input to conv nets, specifically it wants sample number height, width and channel
![convolutional2](https://user-images.githubusercontent.com/50530596/124159969-0941e980-da9c-11eb-8535-078a17e0f183.png)
since conv nets need to be able to handle color images, you'll often see each sample composed of three sets of 2-dimentional arrays, one for each red, green and blue channel, however our MFCCs only have one channel per sample, but we still need to feed the conv net for dimensions, so we use the reshape function to add an extra dimension that doesn't hold any extra information
```
#CNN for TF expects (bach, height, width, channels)
# so we reshape the input tensors with a color channel of 1
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_val=x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print(x_train.shape)
```
##### (77463, 16, 16, 1)
### Build model
```
model=models.Sequential()
model.add(layers.Conv2D(32,(2,2),activation='relu',input_shape=sample_shape))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(32,(2,2),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64,(2,2),activation='relu'))
#model.add(layers.MaxPooling2D(pool_size=(2,2)))

#Classifier
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(36,activation='softmax'))
```
### Add training parameters to model
```
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
```
### Convert array of indices to 1-hot encoded numpy array
```
def convertToOneHot(size_all,size_y,y):
  y_new=np.zeros((size_all,size_y))
  for i in range(size_y):
    for j in range(size_all):
      if y[j]==i:
        y_new[j,i]=1
  return y_new
```
```
y_train_new=convertToOneHot(62021,36,y_train)
y_val_new=convertToOneHot(7757,36,y_val)
y_test_new=convertToOneHot(7737,36,y_test)
```
```
history=model.fit(x_train,y_train_new,epochs=35,batch_size=100,validation_data=(x_val,y_val_new))
```
### Evaluate model with test set
```
model.evaluate(x=x_test,y=y_test_new)
```
### Last part (Running Inference)
Because machine learning algorithms are normally computationally expensive they're usually run on large computers and servers, however our model is small and simple enough that we can run it on a Raspberry pi but first we need to convert it to a tensorflow lite model, right now our model exists as a group of numbers and commands in a file on our computer, we need to create a quick a python script that converts this model into a tensorflow light model. Tensorflow light model is stored in a special format called a flat buffer which shrinks the size of the model file and allow us to access parts of it serially without first needing to copy the whole thing to memory, we can then copy this tensor flow light model to our Raspberry pi, from there we need to add a microphone to the Raspberry pi in order to capture the audio, it will constantly be listening to everything and converting every second of captured audio to the mel frequency coefficients or MFCCs, those MFCCs will then be fed to our model, we'll use the model to make predictions based on the MFCCs and attempt to classify what was heard, because we're attempting to infer things from unseen data this process is known as inference, our model will then give us the probability it heard our wake word as opposed to anything else, if that probability is over some threshold say 50% we can assume that the wake word was heard

```
#Parameters
keras_model_filename = '/home/model4.h5'
tflite_filename = 'wake_word_stop_lite4.tflite'
# Convert model to TF Lite model
model = models.load_model(keras_model_filename)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_filename, 'wb').write(tflite_model)
```
Here's a flow chart showing how we're going to make this work in real time, we want to have a buffer that's a second long containing raw audio data from the microphone, we convert that raw audio into its MFCCs which are fed into the inference engine that contains our tensorflow light model, the inference engine spits out a number that essentially gives us the probability, it thinks the section of audio contains the word 'on', this section does not contain stop so we should see some low probability, since it doesn't meet our threshold we simply do nothing, we then slide the window our by half a second and compute the MFCCs again, to do this in real time we simply shift the second half of the buffer to the first half and put the newly captured 0.5 seconds of audio into the second half, since this window still doesn't contain the full target word we won't do anything again, we shift the window one more time and this time the MFCCs should more closely match what the model is expecting, as a result we should ideally see a probability over 0.5, because the inference engine gave us something over threshold we can trigger an action, I'm going to use USB microphone to capture the voice.

![TFlite2](https://user-images.githubusercontent.com/50530596/127682447-a04eecb3-42ca-4226-ab57-25412b2ea8e9.png)
![TFlite3](https://user-images.githubusercontent.com/50530596/127682603-cdf44fee-92cd-4edb-8354-b1a67d662ed5.png)
![TFlite4](https://user-images.githubusercontent.com/50530596/127682698-28b8f5d8-08be-4302-a8e1-3715b284cb92.png)

### Running Inference
First, copy the .tflite file over to your Raspberry Pi. You???ll want the file to be in the same directory as your code (or you???ll want to update the path in the code to point to the file).

You will want to plug in a USB microphone into your Raspberry Pi and install any necessary drivers. On the Raspberry Pi, make sure you are running Python 3 and Pip 3.
Python -m pip install sounddevice numpy scipy timeit python_speech_features . 
To install the TensorFlow Lite interpreter, you will need to point pip to the appropriate  wheel file. Go to the  ![TensorFlow Lite quickstart guide](https://www.tensorflow.org/lite/guide/python) and find the table showing the available wheel files. Copy the URL for the TensorFlow Lite package for your processor. For a Raspberry Pi running Raspbian Buster, this will likely be the ARM 32 package for Python 3.7. Install TensorFlow Lite with the following:

Python -m pip install <URL to TensorFlow Lite package>
You???ll want to connect an LED and limiting resistor (100 - 1k ??) between board pin 8 (GPIO14) and a Ground pin. See here for the ![Raspberry Pi pinout guide](https://www.raspberrypi.org/documentation/usage/gpio/). 
Add the following code to a new Python file located in the same directory as your .tflite file:
```
"""
Connect a resistor and LED to board pin 8 and run this script.
Whenever you say "stop", the LED should flash briefly
"""

import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO

from tflite_runtime.interpreter import Interpreter

# Parameters
debug_time = 1
debug_acc = 0
led_pin = 8
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = 'wake_word_stop_lite.tflite'

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO 
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    GPIO.output(led_pin, GPIO.LOW)

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)
    mfccs = mfccs.transpose()

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val1=output_data[0][0]
    val2=output_data[0][1]
    val3=output_data[0][2]
    val4=output_data[0][3]
    val5=output_data[0][4]
    val6=output_data[0][5]
    val7=output_data[0][6]
    val8=output_data[0][7]
    val9=output_data[0][8]
    val10=output_data[0][9]
    val11=output_data[0][10]
    val12=output_data[0][11]
    val13=output_data[0][12]
    val14=output_data[0][13]
    val15=output_data[0][14]
    val16=output_data[0][15]
    val17=output_data[0][16]
    val18=output_data[0][17]
    val19=output_data[0][18]
    val20=output_data[0][19]
    val21=output_data[0][20]
    val22=output_data[0][21]
    val23=output_data[0][22]
    val24=output_data[0][23]
    val25=output_data[0][24]
    val26=output_data[0][25]
    val27=output_data[0][26]
    val28=output_data[0][27]
    val29=output_data[0][28]
    val30=output_data[0][29]
    val31=output_data[0][30]
    val32=output_data[0][31]
    val33=output_data[0][32]
    val34=output_data[0][33]
    val35=output_data[0][34]
    val36=output_data[0][35]
    #print('outputData',output_data)
    changeDuty=50.0
    if val1>word_threshold:
        print('UP')
        if changeDuty>0.0 and changeDuty<=100.0:
            changeDuty=changeDuty+50.0
            p.ChangeDutyCycle(changeDuty)
            print('changeDuty',changeDuty)
        else:
            print('Duty cycle is 100.0')
        #GPIO.output(led_pin,GPIO.HIGH)
    if val2>word_threshold:
        print('unused')
    if val3>word_threshold:
        print('unused')
    if val4>word_threshold:
        print('unused')
    if val5>word_threshold:
        print('unused')
    if val6>word_threshold:
        print('unused')
    if val7>word_threshold:
        print('unused')
    if val8>word_threshold:
        print('unused')
    if val9>word_threshold:
        print('unused')
    if val10>word_threshold:
        print('unused')
    if val11>word_threshold:
        print('unused')
    if val12>word_threshold:
        print('unused')
    if val13>word_threshold:
        print('unused')
    if val14>word_threshold:
        print('unused')
    if val15>word_threshold:
        print('off')
        p.stop()
        #GPIO.output(led_pin,GPIO.LOW)
    if val16>word_threshold:
        print('three')
    if val17>word_threshold:
        print('unused')
    if val18>word_threshold:
        print('unused')
    if val19>word_threshold:
        print('unused')
    if val20>word_threshold:
        print('unused')
    if val21>word_threshold:
        print('unused')
    if val22>word_threshold:
        print('unused')
    if val23>word_threshold:
        print('unused')
    if val24>word_threshold:
        print('unused')
    if val25>word_threshold:
        print('unused')
    if val26>word_threshold:
        print('unused')
    if val27>word_threshold:
        print('unused')
    if val28>word_threshold:
        print('unused')
    if val29>word_threshold:
        print('unused')
    if val30>word_threshold:
        print('on')
        p.start(50)
    if val31>word_threshold:
        print('unused')
    if val32>word_threshold:
        print('unused')
    if val33>word_threshold:
        print('unused')
        print('down')
        if changeDuty>0.0:
            changeDuty=changeDuty-40.0
            p.ChangeDutyCycle(changeDuty)
            print('changeDuty',changeDuty)
        else:
            print('Duty cycle is 0.0')
    if val34>word_threshold:
        print('stop')
        p.stop()
    if val35>word_threshold:
        print('down')
        if changeDuty>0.0:
            changeDuty=changeDuty-40.0
            p.ChangeDutyCycle(changeDuty)
            print('changeDuty',changeDuty)
        else:
            print('Duty cycle is 0.0')
        #GPIO.output(led_pin,GPIO.HIGH)
    if val36>word_threshold:
        print('unused')
  
    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
```
# Sound Recognition (Online)
I'm going to explain you how to make your own voice recognition with Raspberry pi.
  First you have to connect to your Raspberri pi and after opening the command line (ALT+CTRL+DEL) we first need to check the python version of this Raspberry pi, on default Raspberry pi works on python 2, so now we want to make that this Raspberry pi would have python 3.7 as default and this is done by this command : sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10, if you're wondering why we want it on 3.7 it's because python 2.7 is pretty old and with python 3 is simpler to install packages. So now let's talk about programming environment, Raspberry pi has Thonny python IDE which is for simple project and it's easy to use, however we recommend using python idle ,which has more capabilities and it's more convenient, so for installing python idle you need to use this command in the terminal : sudo apt install python3 idle3. 
  Now let's install the required pachages for speech recogntion: 
* [pip install speech recognition](to use speech recognition to recognize the voice from recordings)
* [sudo apt-get install python-pyaudio python3-pyaudio](for using microphone with python we need this pachage)
* [sudo apt-get install flac](since we will be using google web speech api, we need a specific flac encoder, because the data is sent to google by this format, most linux have this encoder as default, however it's not the same on Raspberry pi, they don't have this encoder, that's why we need to install it by hand)
We already have everything we need, let's start programming now, we will be using google web speech api and this is where the fun begins, because the programming with this method is really simple, however it's not the fastest speech recognition method.
```
import speech_recognition as sr #Let's us use commands required for speech recognition
from datetime import date
#from gpiozero import LED
from time import sleep #Let's us to use delays in this program
import RPi.GPIO as GPIO

#GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8,GPIO.OUT,initial=GPIO.LOW)
p=GPIO.PWM(8,50)

changeDuty=50.0

r = sr.Recognizer()
mic = sr.Microphone() #To read sound from microphone

print("hello")

while True:
    with mic as source:
        audio = r.listen(source)    #We send all the audio to method listen where it is recognized by google which returns us only the words that were recognized
    words = r.recognize_google(audio)
    print(words)

    if words == "today": #Prints us today's date
        print(date.today())

    if words == "LED on":
        p.start(changeDuty)
        print('LED ON')

    if words == "LED off":
        p.stop()
        print('LED OFF')
    
    if words == "up":
        print('UP')
        if changeDuty>0.0 and changeDuty<=100.0:
            changeDuty=changeDuty+10.0
            p.ChangeDutyCycle(changeDuty)
            print('changeDuty',changeDuty)
        else:
            print('Duty cycle is 100.0')
    
    if words == "down":
        print('down')
        if changeDuty>0.0:
            changeDuty=changeDuty-10.0
            p.ChangeDutyCycle(changeDuty)
            print('changeDuty',changeDuty)
        else:
            print('Duty cycle is 0.0')

    if words == "goodbye":
        p.stop()
        print("...")
        sleep(1)
        print("...")
        sleep(1)
        print("...")
        sleep(1)
        print("Goodbye")
        break
 
```
For understanding you Raspberry configuration you have to write this command in the terminal : pinout
![pinout](https://user-images.githubusercontent.com/50530596/127731499-71378a4b-9358-4dd6-a91a-71002106c624.png)

# Hand Gesture Recognition using Raspberry Pi and OpenCV
### Introduction 
The essential aim of building hand gesture recognition system is to create a natural interaction between human and computer where the recognized gestures can be used for controlling a robot or conveying meaningful information. Gestures can originate from any bodily motion or state but commonly originate from the face or hand. A gesture is a spatiotemporal pattern which may be static , dynamic or both, and is a form of non-verbal communication.Gestures include motion of head, hands, fingers or other body parts Gesture Recognition collectively refers to the whole process of tracking human gestures. Gesture Recognition and more specifically hand gesture recognition can be used to enhance Human Computer Interaction (HCI) and improve the effective utilisation of the available information flow.
### Purpose
The aim of the project is to create a software that recognises pre defined hand gestures using various computer vision and machine learning algorithms.we can divide the project into three major steps which represent the major objectives in the project.
* Hand Detection and Tracking(Data Gathering)
* Feature Extraction(Training the model)
* Recognition(Gesture Detection)
###### In the first phase, we will collect the images for turn on and off , light up and down the LED, and nothing gesture. Nothing gesture is included so that Raspberry Pi doesn???t make unnecessary moves. This dataset consists of 2724 images belonging to seven classes. In the second phase, we will train the Recognizer for detecting the gestures made by the user, and in the last phase, we will use the trainer data to recognize the gesture made by the user.
### Components Required
* Raspberry Pi
* Pi Camera Module
* LED
###### we need RPi 4 and Pi camera module with OpenCV and Tensorflow installed on it. OpenCV is used here for digital image processing. The most common applications of Digital Image Processing are object detection, Face Recognition, and people counter.
### Install OpenCV 4 on Raspberry Pi 4 and Raspbian Buster
We're going to explain you how to install the opencv library onto your raspberry pi using cmake. Basically installing this labrary is alittle comprehensive and a complex process. 
##### Let???s review the hardware requirements for this tutorial:
* Raspberry Pi: This tutorial assumes you are using a Raspberry Pi 4B 1GB, 2GB or 4GB hardware.
* Operating system: These instructions only apply to Raspbian Buster.
* 32GB microSD: I recommend the high-quality SanDisk 32GB 98Mb/s cards.
Before we begin: Grab your Raspberry Pi 4 and flash BusterOS to your microSD.
###### Before starting with the installation, you need to configure few things to ensure proper installation of OpenCV.
### Step 1: Expanding File System
Why we're doing this is that, it expands the file system and there is no shortage of space on the system that we are working.
###### go to the terminal and type the following command.
```
sudo raspi-config
```
You will see a menu, select advanced options and you will see another menu, just press enter on Expand Filesystem and it will ask you for confirmation, select Yes. After that, the Pi will confirm for a reboot and then it will resize the partition.
###### Note: The mouse won't work on this screen, you need to use the keyboard.
After rebooting, your file system should have been expanded to include all available space on your micro-SD card. You can verify that the disk has been expanded by executing the following command:
```
df -h
```
I would suggest deleting both Wolfram Engine and LibreOffice to reclaim ~1GB of space on your Raspberry Pi:
```
sudo apt-get purge wolfram-engine
sudo apt-get purge libreoffice*
sudo apt-get clean
sudo apt-get autoremove
```
### Step 2: Install dependencies
The first step is to update and upgrade any existing packages. The following commands will update and upgrade any existing packages.
The way you have to do is just write the following codes into your terminal and that is :
```
sudo apt-get update && sudo apt-get upgrade
```
We then need to install some developer tools, including CMake, which helps us configure the OpenCV build process:
```
sudo apt-get install build-essential cmake pkg-config
```
Next, we need to install some image I/O packages that allow us to load various image file formats from disk. Examples of such file formats include JPEG, PNG, TIFF, etc.:
```
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
```
Just as we need image I/O packages, we also need video I/O packages. These libraries allow us to read various video file formats from disk as well as work directly with video streams:
```
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
```
The OpenCV library comes with a sub-module named highgui which is used to display images to our screen and build basic GUIs. In order to compile the highgui module, we need to install the GTK development library and prerequisites:
```
sudo apt-get install libfontconfig1-dev libcairo2-dev
sudo apt-get install libgdk-pixbuf2.0-dev libpango1.0-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev
```
Many operations inside of OpenCV (namely matrix operations) can be optimized further by installing a few extra dependencies:
```
sudo apt-get install libatlas-base-dev gfortran
```
Lastly, let???s install Python 3 header files so we can compile OpenCV with Python bindings:
```
sudo apt-get install python3-dev
```
If you???re working with a fresh install of the OS, it is possible that these versions of Python are already at the newest version (you???ll see a terminal message stating this).
### Step 3: Create your Python virtual environment and install NumPy
A Python virtual environment is an isolated development/testing/production environment on your system ??? it is fully sequestered from other environments. Best of all, you can manage the Python packages inside your your virtual environment inside with pip (Python???s package manager).
You can install pip using the following commands:
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py
sudo rm -rf ~/.cache/pip
```
Let???s install virtualenv  and virtualenvwrapper :
```
sudo pip install virtualenv virtualenvwrapper
```
Once both virtualenv  and virtualenvwrapper  have been installed, open up your ~/.bashrc  file:
```
nano ~/.bashrc
```
and append the following lines to the bottom of the file(by using arrow keys):
```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
![step3InstallOpencv](https://user-images.githubusercontent.com/50530596/128686170-2f60dc9c-1878-4bcb-9841-40bbec086b45.png)
###### Save and exit via ctrl + x , y , enter .
From there, reload your ~/.bashrc  file to apply the changes to your current bash session:
```
source ~/.bashrc
```
Next, create your Python 3 virtual environment:
```
 mkvirtualenv cv -p python3
```
###### Here we are creating a Python virtual environment named cv  using Python 3.
If you have a Raspberry Pi Camera Module attached to your RPi, you should install the PiCamera API now as well:
```
pip install "picamera[array]"
```
Step 4: Compile OpenCV 4 from source
This method gives you the full install of OpenCV 4, including patented (???Non-free???) algorithms. It will take 1-4 hours depending on the processor in your Raspberry Pi.
Let???s go ahead and download the OpenCV source code for both the opencv and opencv_contrib repositories, followed by unarchiving them:
```
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.1.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.1.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.1.1 opencv
mv opencv_contrib-4.1.1 opencv_contrib
```
##### Increasing your SWAP space
Before you start the compile you must increase your SWAP space. Increasing the SWAP will enable you to compile OpenCV with all four cores of the Raspberry Pi (and without the compile hanging due to memory exhausting). Open up your /etc/dphys-swapfile  file:
```
sudo nano /etc/dphys-swapfile
```
and then edit the CONF_SWAPSIZE  variable:
```
# set size to absolute value, leaving empty (default) then uses computed value
#   you most likely don't want this, unless you have an special disk situation
# CONF_SWAPSIZE=100
CONF_SWAPSIZE=2048
```
Notice that we're increasing the swap from 100MB to 2048MB. This is critical to compiling OpenCV with multiple cores on Raspbian Buster.
Save and exit via ctrl + x , y , enter . If you do not increase SWAP it???s very likely that your Pi will hang during the compile.
##### From there, restart the swap service:
```
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
```
#### Compile and install OpenCV 4 on Raspbian Buster
We???re now ready to compile and install the full, optimized OpenCV library on the Raspberry Pi 4.
Ensure you are in the cv  virtual environment using the workon  command:
```
workon cv
```
Then, go ahead and install NumPy (an OpenCV dependency) into the Python virtual environment:
```
pip install numpy
```
And from there configure your build:
```
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
    -D BUILD_EXAMPLES=OFF ..
```
Now that we???ve prepared for our OpenCV 4 compilation, it is time to launch the compile process using all four cores:
```
make -j4
```
###### In case your code is not compiling at 100% and it gets an error like above, try using "make -j1". Also if it gets stuck at 63% or in the middle somewhere, delete all the files from the build folder and rebuild again with proper steps.
Assuming OpenCV compiled without error, you can install your optimized version of OpenCV on your Raspberry Pi:
```
sudo make install
sudo ldconfig
```
#### Reset your SWAP
Don???t forget to go back to your /etc/dphys-swapfile  file and:
* Reset CONF_SWAPSIZE  to 100MB.
* Restart the swap service.
#### Sym-link your OpenCV 4 on the Raspberry Pi
Symbolic links are a way of pointing from one directory to a file or folder elsewhere on your system. For this sub-step, we will sym-link the cv2.so  bindings into your cv  virtual environment.

Let???s proceed to create our sym-link. Be sure to use ???tab-completion??? for all paths below (rather than copying these commands blindly):
```
cd /usr/local/lib/python3.7/site-packages/cv2/python-3.7
sudo mv cv2.cpython-37m-arm-linux-gnueabihf.so cv2.so
cd ~/.virtualenvs/cv/lib/python3.7/site-packages/
ln -s /usr/local/lib/python3.7/site-packages/cv2/python-3.7/cv2.so cv2.so
```
### Step 5: Testing your OpenCV 4 Raspberry Pi BusterOS install
As a quick sanity check, access the cv  virtual environment, fire up a Python shell, and try to import the OpenCV library:
```
cd ~
workon cv
python
>>> import cv2
>>> cv2.__version__
'4.1.1'
>>>
```

### Install tensorflow
First we have to activate our virtual environment that we have created for opencv by using following command :
```
workon cv
```
Install dependencies :
```
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
sudo pip3 install pip --upgrade
sudo pip3 install keras_applications==1.0.8 --no-deps
sudo pip3 install keras_preprocessing==1.1.0 --no-deps
sudo pip3 install numpy==1.20.3
sudo pip3 install h5py==3.1.0
sudo pip3 install pybind11
pip3 install -U --user six wheel mock
```
###### If you get an error by doing the last command, delete --user.
Next, you have to get the wheel file
Depends on the version of your python and the model of your Raspberry pi you have to select the right one, go to this ![link](https://github.com/lhelontra/tensorflow-on-arm/releases) and select: 
```
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
```
Uninstall any version of tensorflow that we already have, if you don't already have tensorflow installed, don't worry about the following command: 
```
sudo pip3 uninstall tensorflow
```
Then install tensorflow :
```
sudo -H pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl
```
###### Restart the terminal.
#### Testing your tensorflow Raspberry Pi BusterOS install
```
workon cv
python
import tensorflow as tf
tf.__version__
```
### Hand Detection and Tracking(Data Gathering)
This step deals with detection of hand in the frame and tracking it through the video, our objective in this step is to create a robust system that can detect and track hands of different skin colours in varying light conditions with different but simple background. 
  
### Feature Extraction(Training the model)
This step deals with extracting important features that represent important characteristics of the gesture throughout the video and then storing these features. Our objective in this step is to find features that represent shape, motion, size, reflectivity and other important properties.
### Recognition(Gesture Detection)
This step deals with recognising and classifying the performed gesture.It has two phases , the training phase which involves training the system on datasets and the classification phase which involves classifying the performed gestures, our objective in this step is to obtain classification with high accuracy within minimum time.
### Project Scope
The scope of this project is to build a real time gesture classification system that can automatically detect gestures in natural lighting condition. In order to accomplish this objective, a real time gesture based system is developed to identify gestures. This system will work as one of futuristic of IOT with user interface. Its create method to recognize hand gesture based on different parameters. The main priority of this system is to simple, easy and user friendly without making any special hardware.
### Method to design
The design of hand gesture recognition system is broadly divided into two phase.
* The first phase is the preprocessing phase.
* The second phase is the classification phase.
### preprocessing phase
The efficiency of the Classification phase entirely depends on the preprocessing phase. The main purpose of the pre-processing stage is to:
* Extract the only hand gesture from an image.
* Remove the noises (if present) and unwanted region.
* Process the extracted image to form a binary image and
* Extract the distinguishable significant features from the processed image,
to form a feature set for classification. In this step I have used background elimination, average method and thresholding. Infact in this step we have maked a method namely collectData.py using background elimination method.
##### Import libraries
```
import cv2
import imutils
import numpy as np
import os
# global variables
bg = None
```
To find the running average over the background :
```
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)
```
To segment the region of hand in the image :
```
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    cv2.imshow("different image", diff)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
```
##### Main function
Create the directory structure
```
if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/train")
        os.makedirs("data/test")
        os.makedirs("data/validation")
        os.makedirs("data/train/0")
        os.makedirs("data/train/1")
        os.makedirs("data/train/2")
        os.makedirs("data/train/3")
        os.makedirs("data/train/4")
        os.makedirs("data/train/5")
        os.makedirs("data/train/nothing")
        os.makedirs("data/test/0")
        os.makedirs("data/test/1")
        os.makedirs("data/test/2")
        os.makedirs("data/test/3")
        os.makedirs("data/test/4")
        os.makedirs("data/test/5")
        os.makedirs("data/test/nothing")
        os.makedirs("data/validation/0")
        os.makedirs("data/validation/1")
        os.makedirs("data/validation/2")
        os.makedirs("data/validation/3")
        os.makedirs("data/validation/4")
        os.makedirs("data/validation/5")
        os.makedirs("data/validation/nothing")
```
Here we have to select train, test or validation directory for gathering the data.
```
mode = 'train'
directory = 'data/'+mode+'/'
```
initialize weight for running average
```
aWeight = 0.5
```
get the reference to the webcam
```
camera = cv2.VideoCapture(0)
```
region of interest (ROI) coordinates
```
top, right, bottom, left = 10, 350, 225, 590
```
initialize number of frames
```
num_frames = 0
```
keep looping, until interrupted
```
while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # Getting count of existing images
        count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'nothing': len(os.listdir(directory+"/nothing"))}
        # Printing the count in each set to the screen
        cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "NOTHING : "+str(count['nothing']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        
        # observe the keypress by the user  
        #thresholded = cv2.resize(thresholded, (64, 64)) 
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('0'):
            cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', cv2.resize(thresholded, (64, 64)))
        if interrupt & 0xFF == ord('1'):
            cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('2'):
            cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('3'):
            cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('4'):
            cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('5'):
            cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('n'):
            cv2.imwrite(directory+'nothing/'+str(count['nothing'])+'.jpg', cv2.resize(thresholded, (64, 64)) )

# free up memory
camera.release()
cv2.destroyAllWindows() 
```
![handGesture1](https://user-images.githubusercontent.com/50530596/129381647-4533f895-1c4e-411a-8402-2541729c8524.png)
### Classification phase (Feature Extraction)
 for classification, there are lots of efficient algorithms that are already used. They are Gradient, PCA (Principal Component Analysis) and SVM (Support Vector Machine).
 I have focuses on SVM, which is a machine learning algorithm, from which I have used CNN. For this end we have used google colaboratory for training the model. First of all we have to mount our google drive, because we have uploaded our dataset into google drive
```
from google.colab import drive
drive.mount('/content/drive')
```
By the following command we transfer our dataset from google drive to google colaboratory and then we unzip it.                         
```
!cp /content/drive/MyDrive/Thesis/HandGesture/dataSet/train.zip /home/
!mkdir /home/Dataset
!unzip /home/train.zip -d /home/Dataset   
                           
!cp /content/drive/MyDrive/Thesis/HandGesture/dataSet/validData.zip /home/
!unzip /home/validData.zip -d /home/Dataset                           
```
### Import libraries
                           
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, shutil,glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from ImageDataAugmentor.image_data_augmentor import *
import albumentations                           
```
```
train_dir='/home/Dataset/train'
validation_dir='/home/Dataset/validData'
```
### For seeing how many classes we have                           
```
import torchvision.datasets as datasets
valid_ds = datasets.ImageFolder(validation_dir)
valid_ds.classes
```
### Make the model                          
```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same',strides=1, input_shape=(64, 64, 1),name = "Conv1_layer"))
model.add(layers.MaxPooling2D((2, 2),name = "Max1_layer"))
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same',strides=1,name = "Conv2_layer"))
model.add(layers.MaxPooling2D((2, 2),name = "Max2_layer"))
model.add(layers.Conv2D(64, (5, 5), activation='relu',padding='same',strides=1,name = "Conv3_layer"))
model.add(layers.MaxPooling2D((2, 2),name = "Max3_layer"))
model.add(layers.Conv2D(128, (5, 5), activation='relu',padding='same',strides=1,name = "Conv4_layer"))
model.add(layers.MaxPooling2D((2, 2),name = "Max4_layer"))
model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu',name = "Dense1_layer"))
model.add(layers.Dense(7, activation='softmax',name = "Dense2_layer_softmax"))
tf.keras.utils.plot_model(model, to_file="model.png")
```
### Compile the model                           
```
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='categorical_crossentropy', metrics=['acc'])
```
```
train_datagen = ImageDataAugmentor(rescale=1./255,
        #augment = AUGMENTATIONS, preprocess_input=None)
        
validation_datagen = ImageDataAugmentor(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 64*64
        target_size=(64, 64), batch_size=10, color_mode='grayscale', class_mode='categorical',shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(64, 64), batch_size=6, color_mode='grayscale', class_mode='categorical', shuffle=True)
```
### Train the model                          
```
history = model.fit(train_generator, steps_per_epoch=221, epochs=50, validation_data=validation_generator, validation_steps=85 )
```
### Saving the model                          
```
model.save('/content/drive/MyDrive/Thesis/Models/handGesture50Epoch.h5')
```
```
# Saving the model
model_json = model.to_json()
with open("/content/drive/MyDrive/Thesis/Models/handGesture50Epoch.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('/content/drive/MyDrive/Thesis/Models/handGesture50Epoch.h5')
```
### Recognition(Gesture Detection) phase
We use the model saved in the previous part for prediction I mean gesture detection, in the following you can see the code.                           
```
# organize imports
import cv2
import imutils
import numpy as np
import os
from keras.models import model_from_json
import operator
import RPi.GPIO as GPIO
import timeit
# global variables
bg = None
changeDuty=50.0
#GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8,GPIO.OUT,initial=GPIO.LOW)
p=GPIO.PWM(8,50)
#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)
#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    cv2.imshow("different image", diff)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    
#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # Loading the model
    json_file = open("handGesture50Epoch.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("handGesture50Epoch.h5")
    print("Loaded model from disk")
    # Category dictionary
    categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE', 6: 'NOTHING'}

    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                
                thresholded = cv2.resize(thresholded, (64, 64)) 
                result = loaded_model.predict(thresholded.reshape(1, 64, 64, 1))
                prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'NOTHING': result[0][6]}
                # Sorting based on top prediction
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
                # Displaying the predictions
                cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
                cv2.imshow("Frame", frame)
                if prediction[0][0]=='ONE':
                    print('ON')
                    p.start(50)
                if prediction[0][0]=='TWO':
                    print('OFF')
                    p.stop()
                if prediction[0][0]=='THREE':
                    print('UP')
                    if changeDuty>=0.0 and changeDuty<100.0:
                        changeDuty=changeDuty+10.0
                        p.ChangeDutyCycle(changeDuty)
                        print('changeDuty',changeDuty)
                    else:
                        print('Duty cycle is 100.0')
                if prediction[0][0]=='FOUR':
                    print('DOWN')
                    if changeDuty>10.0:
                        changeDuty=changeDuty-10.0
                        p.ChangeDutyCycle(changeDuty)
                        print('changeDuty',changeDuty)
                    else:
                        print('Duty cycle is 0.0')

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        
        # observe the keypress by the user  
        #thresholded = cv2.resize(thresholded, (64, 64)) 
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        
# free up memory
camera.release()
cv2.destroyAllWindows()
```
### Setting Up UART Serial Communication between 2 Raspberry Pis
##### Prerequisites
* Two Raspberry Pi boards
* 3 jumper wires (female-female) : to connect GPIO pins between Raspberry Pis.
##### Wiring
Connect jumper wires between two Raspberry Pi boards. Rx pin on one Raspberry Pi should be connected to Tx pin on the other Raspberry Pi. In our case :
* connect pin number 6(GND) of Raspberri pi 3 to pin number 6(GND) of Raspberry pi 4
* connect pin number 8(UART-TX) of Raspberri pi 3 to pin number 10(UART-RX) of Raspberry pi 4
* connect pin number 10(UART-RX) of Raspberri pi 3 to pin number 8(UART-TX) of Raspberry pi 4
You have to apply the following steps in both Raspberry pi's.
##### Enabling UART
we can make use of the raspi-config tool. This tool will allow us to easily disable the serial input/output interface that is enabled by default
```
sudo raspi-config
```
This command will load up the Raspberry Pi configuration screen. Use the arrow keys to go down and select ???5 Interfacing Options???. Once this option has been selected, you can press Enter. With the next screen you will want to use the arrow keys again to select ???P6 Serial???, press Enter once highlighted to select this option. You will now be prompted as to whether you want the login shell to be accessible over serial, select No with your arrow keys and press Enter to proceed. Immediately after you will be asked if you want to make use of the Serial Port Hardware, make sure that you select Yes with your arrow keys and press Enter to proceed. Once the Raspberry Pi has made the changes, you should see the following text appear on your screen.
###### The serial login shell is disabled , The serial interface is enabled.
Next, reset your Raspberry pi by following command.
```
sudo reboot
```
Let???s now check to make sure that everything has been changed correctly by running the following command on your Raspberry Pi.
```
dmesg | grep tty
```
Here you want to make sure the following message is not displayed in the output, if it is not there then you can skip onto the next section. Otherwise, start over from beginning. These messages indicate that Serial Login is still enabled for that interface.
* [ttyS0] enabled
* [ttyAMA0] enabled
##### Programming the Raspberry Pi for Serial Writing
In our case , the Raspberry pi 3 that have USB microphone, sends some message to the Raspberry pi 4 that have camera module.
* If it detects the word 'on' , sends number 1 to Raspberry pi 4.
* If it detects the word 'off' , sends number 2 to Raspberry pi 4.
* If it detects the word 'up' , sends number 3 to Raspberry pi 4.
* If it detects the word 'down' , sends number 4 to Raspberry pi 4.
###### Then we add the following codes to our main code related to the voice recognition part.
```
import serial
import time

s=0

ser = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1)
if val1>word_threshold:
        print('UP')
        s=3
        ser.write(str.encode("%d\n"%s))
        time.sleep(1)
if val15>word_threshold:
        print('off')
        s=2
        ser.write(str.encode("%d\n"%s))
        time.sleep(1)
if val30>word_threshold:
        print('on')
        s=1
        ser.write(str.encode("%d\n"%s))
        time.sleep(1)
if val35>word_threshold:
        print('down')
        s=4
        ser.write(str.encode("%d\n"%s))
        time.sleep(1)
```
##### Programming the Raspberry Pi for Serial Reading
We connect the LED to Raspberry pi 4 then we add the following codes to our main code related to the hand gesture part.
```
import time
import serial

ser = serial.Serial(
        port='/dev/ttyS0',
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1)
while(True):
        x=ser.readline()
        if num_frames < 30:
            run_avg(gray, aWeight)
        elif x==b'1\n' or x==b'2\n' or x==b'3\n' or x==b'4\n':
            if x==b'1\n':
                print('ON')
                p.start(50)
            if x==b'2\n':
                print('OFF')
                p.stop()            
            if x==b'3\n':
                print('UP')
                if changeDuty>=0.0 and changeDuty<100.0:
                    changeDuty=changeDuty+10.0
                    p.ChangeDutyCycle(changeDuty)
                    print('changeDuty',changeDuty)
                else:
                    print('Duty cycle is 100.0')
            if x==b'4\n':
                print('DOWN')
                if changeDuty>10.0:
                    changeDuty=changeDuty-10.0
                    p.ChangeDutyCycle(changeDuty)
                    print('changeDuty',changeDuty)
                else:
                    print('Duty cycle is 0.0')
            
        else:
             #Like before
```
