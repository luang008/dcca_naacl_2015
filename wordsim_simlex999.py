import sys
import numpy
import gzip
import math

from numpy.linalg import norm
from ranking import spearmans_rho
from ranking import assign_ranks

#Calculates the cosime sim between two numpy arrays
def cosine_sim(vec1, vec2):  
  return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

#Read all the word vectors and normalize them
def read_word_vectors(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
    
  for lineNum, line in enumerate(fileObject):
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
  
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

if __name__=='__main__':

  wordVectorFile = sys.argv[1]
  wordVectors = read_word_vectors(wordVectorFile)
  FILES = ['SIMLEX999.txt']
  result_line = ''	
  for i, FILE in enumerate(FILES):
    		
    manualDict, autoDict = ({}, {})
    notFound, totalSize = (0, 0)
    for line in open(FILE,'r'):
      line = line.strip().lower()
      word1, word2, val = line.split()
      if word1 in wordVectors and word2 in wordVectors:
        manualDict[(word1, word2)] = float(val)
        autoDict[(word1, word2)] = cosine_sim(wordVectors[word1], wordVectors[word2])
      else:
        notFound += 1
        totalSize += 1
        

    result_line = result_line + "%.4f" % spearmans_rho(assign_ranks(manualDict), assign_ranks(autoDict)) + ' '# + "%15s" % str(notFound) + ' '+ "%15s" % str(totalSize) + ' '
  result_line = result_line[:-1]  
  print result_line 	
