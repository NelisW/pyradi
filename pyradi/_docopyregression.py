# -*- coding: utf-8 -*-


"""This is a lazy regression testing attempt, but it sort of works.

This script requires the regression data at https://github.com/NelisW/pyradi-data,
on the same level as the pyradi repo clone:

    ..
    +-pyradi  [https://github.com/NelisW/pyradi]
      +-.git
      |-setup.py (this file)
      | ...
      +-pyradi
        + ... all the pyradi files

    +-pyradi-docs [https://github.com/NelisW/pyradi-data] (this is an optional clone) 
      +-.git
      +-_build

    +-pyradi-data [https://github.com/NelisW/pyradi-docs] (this is an optional clone) 
      +-.git
      +-regression
      +-images

The test suite probably does not have 100% coverage.
It might grow in coverage with time. For now, it tests basic operation.
If you find some gaps in coverage please let me know.

The following file types are not hash stable and are not used for comparison:
svg, eps, pdf.
The following files are too big are not used for comparison:
hdf5.

The procedure is as follows:
1. Manually: activate conda environment containing the version you want to test with.
2. Manually: remove all clutter files, old results, etc. (python _cleanClutter.py)
3. Run this script and wait for the results.

The algorithm used in the code below is as follows:
1. Get the list of scripts to be tested from the regression folder. 
   Only scripts with subfolders in regression folder are executed.
2. Run the scripts identified in (1) above.
3. Calculate a hash numbers for the local and regression versions of the file 
4. Compare the local file hash and the regression file hash and decide pass/fail.
   If one of the two files missing, indicate such.
5. At the end, write list of passed/failed/missing to the regression test result file.
   The regression result file is a text file with extension regrtxt. The files are 
   committed to the repository.

Notes:
1. File compares are done by comparing the hashes for the respective files.
   This means that this method does a binary compare: \r\n vs \n text files fail.
   Also any minute difference in png files will also result in a fail. 
   Some manual investigation is required on failed files.
2. Only files present in the regression folders are used for comparison. Files not
   present in these folders are ignored.
3. Each script's regression folder must have subfolders where the subfolder is the
   name of the environment used when the tests were excuted. Currently py27 and py35.
4. Comparisons are marked thus:
   + passed: successful binary comparison between local and regression reference.
   - failed: unsuccessful binary comparison between local and regression reference.
   ? missing: there is no local file to match a regression reference file.
5. The regression set is built on Windows 7.

image comparison taken from here:
http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images

"""

import subprocess
import time
import os
import os.path
import pyradi.ryfiles as ryfiles
import sys
import hashlib

import sys
import scipy
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

pathToRegressionData = '../../pyradi-data/regression'

#################################################################################
# to detect text files:
# http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
is_binary_string = lambda bytes: bool(bytes.translate(None, textchars))

#################################################################################
def hash_file(filename):
    """"This function returns the SHA-1 hash
    of the file contents for the filename passed into it
    """

    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    # make a hash object
    h = hashlib.sha1()

    # open file for reading in binary mode
    with open(filename,'rb') as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only BUF_SIZE bytes at a time
            chunk = file.read(BUF_SIZE)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()


#################################################################################
def runTask(task,sleeptime):
    p=subprocess.Popen(task)          
    while  p.poll() == None:
        time.sleep(sleeptime)


#################################################################################
def getcondaenvis():
    envis = []

    # find the current environment
    out = subprocess.check_output('conda info --envs')
    envi = ''
    for line in out.split(b'\r\n'):
        if not b'#' in line and len(line)>2:
            # print(line)
            if b'*' in line:
                cenvi = line.split()[0].decode('utf-8')
            if not b'root' in line:
                envis.append(line.split()[0].decode('utf-8'))
    return envis,cenvi


#################################################################################
def getscriptnamesregression():
    # get a list of all folders in regression set
    flist = ryfiles.listFiles(pathToRegressionData, patterns='*', recurse=0, return_folders=1)
    ryscripts = []
    for fli in flist:
        if not os.path.isfile(fli):
            ryscripts.append(fli)
    return ryscripts

#################################################################################
# http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

#################################################################################
# http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
def to_grayscale(arr):
    """If arr is a color image (3D array), convert it to grayscale (2D array).
    """
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

#################################################################################
# http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
def imagedifference(file1, file2):
    # read images as 2D arrays (convert to grayscale for simplicity)
    img1 = to_grayscale(imread(file1).astype(float))
    img2 = to_grayscale(imread(file2).astype(float))
    if not img1.shape == img2.shape:
        # resize
        img2 = scipy.misc.imresize(img2,img1.shape)
    # compare
    inorm = compare_images(img1, img2)

    return inorm 

#################################################################################
# http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
def compare_images(img1, img2, method='zeronorm'):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    if method in 'zeronorm':
        inorm = norm(diff.ravel(), 0)  # Zero norm
    else:
        inorm = sum(abs(diff))  # Manhattan norm
    #normalise by image size
    inorm /= float(img1.size)
    return inorm


#################################################################################
def runscripts(ryscripts,fout=None, similarity=0.01):
    # run tests for each script and compare against its own ref set
    passed = {}
    failed = {}
    missing = {}
    for ryscript in ryscripts:
        # get the script name for this run
        script = os.path.basename(ryscript) 

        #build a commandline string to execute and write stdout to file
        task = 'python {}.py >{}-{}.txt'.format(script,script,cenvi)
        print('\n{}'.format(task))
        out = subprocess.check_output(task)
        with open('{}-{}.txt'.format(script,cenvi),'wb') as foutlocal:
            foutlocal.write(out)

        for envi in envis:
            if envi in cenvi: # replace with True to cross check
                if envi in cenvi:
                    strenvi = ''
                else:
                    strenvi = ' in {}'.format(envi)

                #get the list of files in the regression folder
                targfolder = os.path.join(pathToRegressionData,script.split('.')[0],envi)
                print('Regression result files in folder: {}'.format(targfolder))

                rlist = ryfiles.listFiles(targfolder, patterns='*.*', recurse=0, return_folders=0)
                for fregres in rlist:
                    flocal = os.path.basename(fregres)

                    if os.path.exists(flocal):
                        hflocal = hash_file(flocal)

                        if os.path.exists(fregres):
                            hfregres = hash_file(fregres)
                        
                            if hflocal==hfregres:
                                result = '+'
                                passed['{}{}'.format(flocal,strenvi)] = 1
                            else:
                                # test for text string compare
                                if not is_binary_string(open(fregres, 'rb').read(1024)):
                                    if cmp_lines(fregres, flocal):
                                        passed['{}{}'.format(flocal,strenvi)] = 1
                                    else:
                                        failed['{}{}'.format(flocal,strenvi)] = 1
                                else:
                                    # hashes and text content mismatch, these are probably bitmaps or binary files
                                    imgdiff = imagedifference(flocal, fregres)

                                    if imgdiff < similarity:
                                        result = '+'
                                        passed['{}{} -- {:.5f}'.format(flocal,strenvi,imgdiff)] = 1
                                    else:
                                        result = '-'
                                        failed['{}{} -- {:.5f}'.format(flocal,strenvi,imgdiff)] = 1
                        # else:
                        #     result = '?'
                        #     missing['{}{}'.format(flocal,strenvi)] = 1
                    else:
                        result = '?'
                        missing['{}{}'.format(flocal,strenvi)] = 1

                    print('  {}  {} vs {}'.format(result,fregres,flocal))

    if fout is not None:
        fout.write('\n\nPassed:\n')
        for key in sorted(list(passed.keys())):
            fout.write('   + {}\n'.format(key))

        fout.write('\n\nFailed:\n')
        for key in sorted(list(failed.keys())):
            fout.write('   - {}\n'.format(key))

        fout.write('\n\nMissing:\n')
        for key in sorted(list(missing.keys())):
            fout.write('   ? {}\n'.format(key))

    return passed, failed, missing


########################################################################
def cmp_lines(path_1, path_2):
    l1 = l2 = ' '
    with open(path_1, 'rb') as f1:
        with open(path_2, 'rb') as f2:
            while len(l1)>0 and len(l2)>0:
                l1 = f1.readline()
                l2 = f2.readline()
                if l1.strip() != l2.strip():
                    return False
    return True


#################################################################################
def compareenvresults(envis,ryscripts,fout=None, similarity=0.01):
    """Compare files with the same names in each of environments
    """

    fdict = {}
    fnames = {}
    for ryscript in ryscripts:
        # get the script name for this run
        script = os.path.basename(ryscript)
        fdict[script] = {}

        for envi in envis:
            fdict[script][envi] = {}
            # capture the files in the environment
            paths = ryfiles.listFiles(os.path.join(ryscript,envi), patterns='*', 
                recurse=0, return_folders=0)
            for path in paths:
                fname = os.path.basename(path)
                fnames[fname] = 1
                if  os.path.isfile(path):
                    fdict[script][envi][fname] = hash_file(path)
                else:
                    fdict[script][envi][fname] = None

    # finished collecting, now compare
    sameHash = {}
    sameTxt = {}
    sameImg = {}
    diffEnv = {}
    for ryscript in ryscripts:
        script = os.path.basename(ryscript) 
        for fname in fnames:
            hsh0 = None
            if script in fdict.keys():
                if envis[0] in fdict[script].keys():
                    if fname in fdict[script][envis[0]].keys():
                        hsh0 = fdict[script][envis[0]][fname]
            hsh1 = None
            if script in fdict.keys():
                if envis[1] in fdict[script].keys():
                    if fname in fdict[script][envis[1]].keys():
                        hsh1 = fdict[script][envis[1]][fname]

            if hsh0 is not None and hsh1 is not None:
                if hsh0 == hsh1:
                    sameHash['{}/{}'.format(script,fname)] = 1
                else:
                    path0 = os.path.join(pathToRegressionData,script,envis[0],fname)
                    path1 = os.path.join(pathToRegressionData,script,envis[1],fname)
                    # do text compare if text file
                    if not is_binary_string(open(path0, 'rb').read(1024)):
                        if cmp_lines(path0, path1):
                            sameTxt['{}/{}'.format(script,fname)] = 1
                        else:
                            diffEnv['{}/{}'.format(script,fname)] = 1
                    else:
                        # hashes and text content mismatch, these are probably bitmaps or binary files
                        imgdiff = imagedifference(path0, path1)

                        if imgdiff < similarity:
                            sameImg['{}/{}'.format(script,fname)] = imgdiff
                        else:
                            print('{} {}'.format(path0, imgdiff))
                            diffEnv['{}/{}'.format(script,fname)] = 1

    if fout is not None:
        fout.write('\n\nHash the same in both environments:\n')
        for key in sorted(list(sameHash.keys())):
            fout.write('   + {}\n'.format(key))

        fout.write('\n\nText the same in both environments:\n')
        for key in sorted(list(sameTxt.keys())):
            fout.write('   + {}\n'.format(key))

        fout.write('\n\nImages are similar in both environments with normalised error less than {}:\n'.format(similarity))
        for key in sorted(list(sameImg.keys())):
            fout.write('   + {}   {:.5f}\n'.format(key,sameImg[key]))

        fout.write('\n\nDifferent between environments:\n')
        for key in sorted(list(diffEnv.keys())):
            fout.write('   - {}   {:.5f}\n'.format(key,diffEnv[key]))


    return sameHash, sameTxt, diffEnv



########################################################################

runScripts = True
compareEnvResults = True

# get conda environments
envis,cenvi = getcondaenvis()
print('Currently working in environment {}'.format(cenvi))
print('All environments: {}'.format(envis))

# open disk file for writing
if sys.version_info[0] > 2:
    fout = open('regression-results-{}.regrtxt'.format(cenvi),'wt', encoding='utf-8') 
else:
    fout = open('regression-results-{}.regrtxt'.format(cenvi),'wt') 

now = time.strftime("%c")
fout.write('Environment: {}\n'.format(cenvi))
fout.write('Python: {}\n'.format(sys.version))
fout.write('Current date & time {}\n'.format(time.strftime('%c')))
fout.write('------------------------------------\n')


# get the names of the folders in regression - these are script names
ryscripts = getscriptnamesregression()

# run the scripts and collect test results
if runScripts:
    passed, failed, missing = runscripts(ryscripts, fout)
    # passed, failed, missing = runscripts(ryscripts=['ryptw'], fout)


if compareEnvResults:
    sameHash, sameTxt, diffEnv = compareenvresults(envis,ryscripts,fout)


fout.close()
