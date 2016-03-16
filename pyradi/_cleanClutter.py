#clean a precisely defined set of files from a precisely defined directory set.
#the specified files are deleted, after the user has been prompted

import os
import os.path, fnmatch
import sys


def listFiles(root, patterns='*', recurse=1, return_folders=0):
    """lists the files in a directory and subdirectories (from Python Cookbook)
    """
    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    filenames = []
    filertn = []

    if sys.version_info[0] < 3:

        # Collect input and output arguments into one bunch
        class Bunch(object):
            def __init__(self, **kwds): self.__dict__.update(kwds)
        arg = Bunch(recurse=recurse, pattern_list=pattern_list,
            return_folders=return_folders, results=[])

        def visit(arg, dirname, files):
            # Append to arg.results all relevant files (and perhaps folders)
            for name in files:
                fullname = os.path.normpath(os.path.join(dirname, name))
                if arg.return_folders or os.path.isfile(fullname):
                    for pattern in arg.pattern_list:
                        if fnmatch.fnmatch(name, pattern):
                            arg.results.append(fullname)
                            break
            # Block recursion if recursion was disallowed
            if not arg.recurse: files[:]=[]
        os.path.walk(root, visit, arg)
        return arg.results

    else:
        for dirpath,dirnames,files in os.walk(root):
            if dirpath==root or recurse:
                for filen in files:
                    filenames.append(os.path.abspath(os.path.join(os.getcwd(),dirpath,filen)))
                if return_folders:
                    for dirn in dirnames:
                        filenames.append(os.path.abspath(os.path.join(os.getcwd(),dirpath,dirn)))
        for name in filenames:
            if return_folders or os.path.isfile(name):
                for pattern in pattern_list:
                    if fnmatch.fnmatch(name, pattern):
                        filertn.append(name)
                        break

    return filertn


def QueryDelete(recurse, dir, patn, promptUser=True, dryrun=False):
    """Delete files in specified directory.

    recurse: defines if the search must be recursively 0=not, 1=recursive
    dir: specifies the path
    patn: specifies the file patterns to erase
    promptUser: if true the user is first asked to confirm
    dryrun: if true nothing is erased

    the user is promted before the files are deleted
    """
    thefiles = listFiles(dir, patn,recurse)
    if thefiles is not None:
        if len(thefiles)>0:
            if promptUser==True:
                for filename in thefiles:
                    print(filename)
            if promptUser:
                if sys.version_info[0] < 3:
                    instr = raw_input("Delete these files? (y/n)")
                else:
                    instr = input("Delete these files? (y/n)")
            else:
                instr = 'y'
            if instr=='y' and not dryrun:
                for filename in thefiles:
                    if os.path.exists(filename):
                        os.remove(filename)


################################################################
import unittest
import tempfile
import shutil

class cleanClutterTest(unittest.TestCase):
    def __init__(self):
        self.paths = []
        self.tdirs = []

    def setUp(self):
        # print "Creating some random files"
        self.exts = ['txt', 'bin', 'tmp', 'bot','pot']
        self.dirs = ['.','cleancluttertmp1','cleancluttertmp2']
        stdir = tempfile.mkdtemp()
        for dir in self.dirs:
            self.tdirs.append(os.path.join(stdir,dir))
            if dir == '.':
                self.root = self.tdirs[-1]
            # print(self.tdirs[-1])
            if not os.path.exists(self.tdirs[-1]):
                os.makedirs(self.tdirs[-1])
        self.totalFiles = 53
        for i in range(0,self.totalFiles):
            (handle, path) = tempfile.mkstemp(prefix='cleanclutter',suffix='.'+self.exts[i%len(self.exts)],
                dir=os.path.join('.',self.tdirs[i%len(self.tdirs)]))
            os.close(handle)
            self.paths.append(path)
        # print(self.root)
        #count the number of files after creation
        self.cntDirs, self.cntAll = self.countFiles()

    def countFiles(self):
        """count the number of files in total and directories, per type.
        """
        cntDirs = {}
        cntAll = {}
        #count all files of given type everywhere in all the dirs
        for ftype in self.exts:
            cntAll[ftype] = len(listFiles(self.root, patterns='*.{}'.format(ftype), recurse=1, return_folders=0))
        #count files of given type per different directory
        for dir in self.dirs:
            cntDirs[dir] = {}
            for ftype in self.exts:
                cntDirs[dir][ftype] = len(listFiles(os.path.join(self.root,dir),
                    patterns='*.{}'.format(ftype), recurse=0, return_folders=0))
        return cntDirs, cntAll

    def tearDown(self):
        # print "Cleaning up"
        for filename in self.paths:
            # print(filename)
            os.remove(filename) if os.path.exists(filename) else None
        for dirname in self.tdirs:
            # print(dirname)
            shutil.rmtree(dirname) if os.path.exists(dirname) else None


    def test_listFiles(self):
        """Count the number of files in total and count the number of files per
        directory: the totals per type should agree.
        The sum of all of these must also be equal to the number of files created.
        """
        #assert that both evaluations tally up the same
        self.chkDirs = {keyt: sum([self.cntDirs[keyd][keyt] for keyd in self.dirs]) for keyt in self.exts}
        self.assertTrue(self.chkDirs==self.cntAll)
        #assert that the total counted agrees with the number created
        self.assertTrue(self.totalFiles==sum([self.cntAll[key] for key in self.cntAll.keys()]))

    def test_QueryDelete(self):
        """Delete given types and in given directories and count after deletion.
        """
        import time
        #first test dry run functionality
        if False:
            QueryDelete(1,self.root,'*.*',promptUser=False,dryrun=True)
            cntDirs01, cntAll01 = self.countFiles()
            #assert that the total counted agrees with the number created
            self.assertTrue(self.totalFiles==sum([cntAll01[key] for key in cntAll01.keys()]))

        #delete all files of one type in one dir only
        ftype = self.exts[0]
        dir = self.dirs[-1]
        dpath = os.path.join(self.root,dir)
        QueryDelete(0,dpath,'*.{}'.format(ftype),promptUser=False)
        time.sleep(0.3)
        #do equivalent in our counter dicts
        self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

        #delete all files of one type in all the dirs
        ftype = self.exts[1]
        dpath = os.path.join(self.root,'.')
        QueryDelete(1,dpath,'*.{}'.format(ftype),promptUser=False)
        time.sleep(0.3)
        #do equivalent in our counter dicts
        for dir in self.cntDirs.keys():
            self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

        #delete all file types in one dir
        dir = self.dirs[1]
        dpath = os.path.join(self.root,dir)
        QueryDelete(0,dpath,'*.*',promptUser=False)
        time.sleep(0.3)
        #do equivalent in our counter dicts
        for ftype in self.exts:
            self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

        #delete all files in all dirs
        dpath = os.path.join(self.root,'.')
        QueryDelete(1,dpath,'*.*',promptUser=False)
        time.sleep(0.3)
        #do equivalent in our counter dicts
        for ftype in self.exts:
            for dir in self.cntDirs.keys():
                self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    # #we take the conservative approach and do not do blanket erase,
    # #rather do it by type, asking the user first
    QueryDelete(0,'.', '*.eps;*.png;*.jpg;*.pdf;*.txt;*.tiff;*.dat;*.lut')
    QueryDelete(0,'.', 'tape7-*.txt;arr*.txt;Traje*.txt;trian*.txt;vertex*.txt;*.testing')
    QueryDelete(0,'.', 'arr*.txt;colourcoordinates.*;tar;*.svg' )
    QueryDelete(0,'.', '*.hdf5')

    # tt = cleanClutterTest()
    # tt.setUp()
    # tt.test_listFiles()
    # tt.test_QueryDelete()
    # tt.tearDown()
