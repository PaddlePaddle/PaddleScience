# Python function to manipulate OpenFOAM files
# Developer: Jian-Xun Wang (jwang33@nd.edu)

###############################################################################

# system import
import numpy as np
import numpy.matlib
import sys # Add extra path/directory
import os
import os.path as ospt
import shutil
import subprocess # Call the command line
from subprocess import call
import matplotlib.pyplot as plt # For plotting
import re
import tempfile
import pdb
from matplotlib import pyplot as plt
# local import
from PIL import Image
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
import multiprocessing
from functools import partial
import time
import multiprocessing
from functools import partial

import scipy.sparse as sp

global unitTest 
unitTest = False;






def readVectorFromFile(UFile):
	""" 
	Arg: 
	tauFile: The directory path of OpenFOAM vector file (e.g., velocity)

	Regurn: 
	vector: Matrix of vector    
	"""
	resMid = extractVector(UFile)
	fout = open('Utemp', 'w');
	glob_pattern = resMid.group()
	glob_pattern = re.sub(r'\(', '', glob_pattern)
	glob_pattern = re.sub(r'\)', '', glob_pattern)
	fout.write(glob_pattern)
	fout.close();
	vector = np.loadtxt('Utemp')
	return vector





	
def readScalarFromFile(fileName):    
	""" 

	Arg: 
	fileName: The file name of OpenFOAM scalar field

	Regurn: 
	a vector of scalar field    
	"""
	resMid = extractScalar(fileName)
	
	# write it in Tautemp 
	fout = open('temp.txt', 'w')
	glob_patternx = resMid.group()
	glob_patternx = re.sub(r'\(', '', glob_patternx)
	glob_patternx = re.sub(r'\)', '', glob_patternx)
	fout.write(glob_patternx)
	fout.close();
	scalarVec = np.loadtxt('temp.txt')
	return scalarVec


################################################ Regular Expression ##################################################### 


def extractVector(vectorFile):
	""" Function is using regular expression select Vector value out
	
	Args:
	UFile: The directory path of file: U

	Returns:
	resMid: the U as (Ux1,Uy1,Uz1);(Ux2,Uy2,Uz2);........
	"""

	fin = open(vectorFile, 'r')  # need consider directory
	line = fin.read() # line is U file to read
	fin.close()
	### select U as (X X X)pattern (Using regular expression)
	patternMid = re.compile(r"""
	(
	\(                                                   # match(
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	\)                                                   # match )
	\n                                                   # match next line
	)+                                                   # search greedly
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)
	return resMid    
	
def extractScalar(scalarFile):
	""" subFunction of readTurbStressFromFile
		Using regular expression to select scalar value out 
	
	Args:
	scalarFile: The directory path of file of scalar

	Returns:
	resMid: scalar selected;
			you need use resMid.group() to see the content.
	"""
	fin = open(scalarFile, 'r')  # need consider directory
	line = fin.read() # line is k file to read
	fin.close()
	### select k as ()pattern (Using regular expression)
	patternMid = re.compile(r"""
		\(                                                   # match"("
		\n                                                   # match next line
		(
		[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
		\n                                                   # match next line
		)+                                                   # search greedly
		\)                                                   # match")"
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)

	return resMid

