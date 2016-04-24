### Code for naacl-2015 paper: Deep Multilingual Correlation for Improved Word Embeddings 
Author: Ang Lu
Based on work of Manaal Faruqui and Weiran Wang

###Requirement:
Python 2.7
Matlab from shell

###Data:
The original word embeddings and aligned word embeddings have been store as .mat format, the header of English embeddings is in head.txt. The original embeddings can be found from Manaal's homepage: http://www.cs.cmu.edu/~mfaruqui .

Before applying DCCA, please follows the instruction https://github.com/mfaruqui/crosslingual-cca to align two original embeddings and get the subset to train the model. Then save them into matlab format using makemat.m. The alignment file is en-de, which may be different from orignial alignment file.

###Usage:

sh alltask.sh [128] [128] 0.0001 0.0001 3000 0.0001 0.99

###Reference:
```
@inproceedings{lu2015deep,
  title={Deep multilingual correlation for improved word embeddings},
  author={Lu, Ang and Wang, Weiran and Bansal, Mohit and Gimpel, Kevin and Livescu, Karen}
}
```


