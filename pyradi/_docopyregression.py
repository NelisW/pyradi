# -*- coding: utf-8 -*-


"""This is a lazy regression testing attempt, but it sort of works.

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



"""

import subprocess
import time
import os
import os.path
import pyradi.ryfiles as ryfiles
import sys
import hashlib

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

def hash_file(filename):
   """"This function returns the SHA-1 hash
   of the file contents for the filename passed into it
   """

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

def runTask(task,sleeptime):
    p=subprocess.Popen(task)          
    while  p.poll() == None:
        time.sleep(sleeptime)

passed = {}
failed = {}
missing = {}
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

print('Currently working in environment {}'.format(cenvi))
print('All environments: {}'.format(envis))

# get a list of all folders in regression set
flist = ryfiles.listFiles('./regressiondata', patterns='*', recurse=0, return_folders=1)
ryscripts = []
for fli in flist:
    if not os.path.isfile(fli):
        ryscripts.append(fli)

for ryscript in ryscripts:
    # get the script name for this run
    script = ryscript.split(os.sep)[1] 

    # if script in 'ry3dnoise':
    if True: 

        #build a commandline string to execute and write stdout to file
        task = 'python {}.py >{}-{}.txt'.format(script,script,cenvi)
        print('\n{}'.format(task))
        out = subprocess.check_output(task)
        with open('{}-{}.txt'.format(script,cenvi),'wb') as fout:
            fout.write(out)

        for envi in envis:
            if envi in cenvi: # replace with True to cross check
                if envi in cenvi:
                    strenvi = ''
                else:
                    strenvi = ' in {}'.format(envi)

                #get the list of files in the regression folder
                targfolder = os.path.join(ryscript.split('.')[0],envi)
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
                                result = '-'
                                failed['{}{}'.format(flocal,strenvi)] = 1
                        # else:
                        #     result = '?'
                        #     missing['{}{}'.format(flocal,strenvi)] = 1
                    else:
                        result = '?'
                        missing['{}{}'.format(flocal,strenvi)] = 1

                    print('  {}  {} vs {}'.format(result,fregres,flocal))

# write the result file to disk
if sys.version_info[0] > 2:
    fout = open('regression-results-{}.regrtxt'.format(cenvi),'wt', encoding='utf-8') 
else:
    fout = open('regression-results-{}.regrtxt'.format(cenvi),'wt') 

now = time.strftime("%c")
fout.write('Environment: {}\n'.format(cenvi))
fout.write('Python: {}\n'.format(sys.version))
fout.write('Current date & time {}\n'.format(time.strftime('%c')))
fout.write('------------------------------------\n')

fout.write('\n\nPassed:\n')
for key in sorted(list(passed.keys())):
    fout.write('   + {}\n'.format(key))

fout.write('\n\nFailed:\n')
for key in sorted(list(failed.keys())):
    fout.write('   - {}\n'.format(key))

fout.write('\n\nMissing:\n')
for key in sorted(list(missing.keys())):
    fout.write('   ? {}\n'.format(key))




fout.close()
