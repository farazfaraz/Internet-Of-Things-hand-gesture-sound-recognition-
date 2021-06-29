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
Finally unzip your folder by following script:
```
!unzip /home/dataSet.zip -d /home/dataSetForVoice
```

