# Web-Crawler

### Purpose

```
Package exists to provide the functionalities of a web crarwler's profile extractor. 
Given a training set from the 898_data directory, the extractor trains a model which 
then, given a professor's blank homepage, is able to determine the key features of 
the professor (for example, professor name, location, affiliation, etc.). All such 
parameters are provided by the paper: https://keg.cs.tsinghua.edu.cn/jietang/publications/TKDD11-Tang-et-al-web-user-profiling.pdf
```

### Deployment/Usage

```
To deploy the code, simply clone the repository. All the data to train the model 
resides within the 898_data directory, and the files will be used in two situations, 
1: if the model is being run for the first time, it will run all the files in the 
folder and train the model based on that, or 2: If the model has been run before, 
the model will assess the cached model weights from a previous run and use them 
as the model, making the process faster. Once you've run the model with 
'python3 main.py', and the model has been trained, the executed file will 
assess whatver file is in test.html and provide a profile construction for that 
html page. Place whatever html prediction you would like in that file. 
```

### Requirements and Dependencies

```
from PyDictionary import PyDictionary
import time
import os
import numpy as np
import sys
import requests
import pandas as pd
import itertools
from sklearn.model_selection import KFold

The above imports match the required packages for the project, and most come default with the standard 
python packages. pandas is one of the packages required, which can be downloaded with the pip command. sys, 
time, os, numpy, PyDictionary, itertools, and requests can be done in a similar manner. Sklearn additionally is required 
for model testing purposes, and can be run with 'pip install -U scikit-learn'. This project assumes you have 
the most up to date python version downloaded.

```

### System Architecture

```
The primary classes include the web crawler class, and the model class. In the main function, we call both 
classes in order to perform the actions specified in the purpose clause above. The model class is split into 
multiple parts, including: 1. All probability calculations
for the specific formulas provided in the standard CRF model (p_theta, p_theta sum, etc.), 2. Optimizer functions, 
which can be used for training weights from the feature functions, as well as actually train the data from the 
provided directory, 3. The Feature functions, which are a set of functions that accept the following parameters: 
(sentence, word_label, prev_word_label, i), and outputs a truth value based upon the input parameters, and finally
4. Testing, which has a set of functions designed to function with KFold cross validation to ensure that the model
is tested correctly on the data directory. The web crawler class provides an interface to read in websites to html pages.
```

### Codebase Organization 

```
As covered in the file list, each of the files provided have interlinking functionalitites, but main.py serves as the 
primary 'brain' of the project. It handles the classification model, entry and exit points, and calls the data required. 
The supporting data is held by 898_data, which holds labeled data useful for training and testing, and unlabeled_webpages,
which serves as the container for HTML pages with to tagging on them whatsoever. For more details, see the section below.
```

### File List 

```
The repo is organized into a standard github repo, titled Web-Crawler. This repo contains 6 total entities, 4 files 
and 2 directories. main.py serves as the main entry point through which the user can enter information about training 
data, testing files, and professor html webpages. trained_model.pickle may or may not be present, depending on 
whether or not the repo has been run by the user already. It contains a trained set of weights for use by the classifier. 
urls_sample.txt contains a sample list of urls through which the user can find professor webpages, and thus use their 
htmls for testing purposes. The README.md is this file, and the 898_data directory contains the entire set of 
training data which is used by main.py. All files can be found in this repository. The unlabeled_webpages directory 
serves as the container for all pure html webpages of professors which can be used for demos, testing, etc.
```

### Description 

```
The code is organized into a few sections within the overall CRF_MODEL class, as specified earlier in the system architecture
question. Beginning from the main function, as soon as the module is run, the first task completed is the training of the 
model. If a model file with valid weights already exists, then the model is loaded from the file and the training is 
completed. Otherwise, the model will begin training the weights, which have been randomly initialized from 0.5 to 10, 
decimal, inclusive. It does so by iterating through each weight, and the respective feature function, and calling the formula 
specified by the following link's gradient descent section: 
https://medium.com/data-science-in-your-pocket/named-entity-recognition-ner-using-conditional-random-fields-in-nlp-3660df22e95c. 
Once the weights have been fully trained, then we move on to the predicting. We have a few options at this point, including
accuracy testing, page testing, or sentence testing. Accuracy testing involves calling the KFOLD_cross_validation() function, 
which will print out the overall accuracy of the model given the KFold technique. Page testing invloves calling 
find_page_labels(), which outputs a set of labels given an html page, specifically those regarding the descriptors of the
professor. Sentence testing can be invoked by calling test_sentence(), which outputs a classification set for a sample 
sentence, more used for minute testing. All the protocols/algorithms used by the model, includeing p_theta, gradient 
descent, and the testing formula can be found on the medium page specified earlier.
```

### Limitations and Improvements

```
In terms of current issues, the current problems consist of speed. Training the model takes on average 15-20 minutes, most likely 
due to the large amount of calculations required to perform the training for even one set of weights, and the large amount of 
files present in the 898_data directory. In order to remedy this currently, The weights are stored in a file for later usage, 
but retraining the weights would require a large amount of time. Additionally, predictions take 5+ minutes, as the prediction 
for one page must check each sentence against all possible label combinations with respect to p_theta. This is a permutation
function, so in certain cases can amount to multiple seconds of checking for just one sentence. Finally, ony issue that can 
always be corrected is the accuracy. Because the dataset is quite diverse, overfitting danger is low. We can add more feature 
functions in order to correct this, with focus on english morphology, and even adding facial recognition software to detect 
whether certain images contain faces or not. More examples of feature functions can be found in the description section of this 
README, through the medium article.
```

