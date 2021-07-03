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
### Last part
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



