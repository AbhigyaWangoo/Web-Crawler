# Web-Crawler

###### Purpose // What’s the purpose of this package.

```
Package exists to provide the functionalities of a web crarwler's profile extractor. Given a training set from the 898_data directory, the extractor trains a model which then, given a professor's blank homepage, is able to determine the key features of the professor (for example, professor name, location, affiliation, etc.). All such parameters are provided by the paper: https://keg.cs.tsinghua.edu.cn/jietang/publications/TKDD11-Tang-et-al-web-user-profiling.pdf
```

###### Deployment // How to deploy/install the code and how to use it.

```
To deploy the code, simply clone the repository. All the data to train the model resides within the 898_data directory, and the files will be used in two situations, 1: if the model is being run for the first time, it will run all the files in the folder and train the model based on that, or 2: If the model has been run before, the model will assess the cached model weights from a previous run and use them as the model, making the process faster. Once you've run the model with 'python3 main.py', and the model has been trained, the executed file will assess whatver file is in test.html and provide a profile construction for that html page. Place whatever html prediction you would like in that file. 
```

###### Usage // How to use it?



###### Requirements and Dependencies // What are the requirements for running the code. What packages/components does it depend on.

###### System Architecture // The components and how they relate to each other. Give corresponding classes or modules/functions.

###### Codebase Organization // How is the code repo organized. How does it correspond to the system architecture.

###### File List // list and link to the files. For each file, give a one-sentence description for it does.

###### Description // Explain how the code works – workflows, algorithms, or protocols.

###### Limitations and Improvements // What are pending issues to address.
