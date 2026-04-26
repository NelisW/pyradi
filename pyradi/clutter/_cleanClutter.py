#clean a precisely defined set of files from a precisely defined directory set.
#the specified files are deleted, after the user has been prompted

import os
import os.path, fnmatch
import sys


def listFiles(root, patterns='*', recurse=1, return_folders=0, useRegex=False):
    """Lists the files/directories meeting specific requirement

        Returns a list of file paths to files in a file system, searching a 
        directory structure along the specified path, looking for files 
        that matches the glob pattern. If specified, the search will continue 
        into sub-directories.  A list of matching names is returned. The 
        function supports a local or network reachable filesystem, but not URLs.

    Args:
        | root (string): directory root from where the search must take place
        | patterns (string): glob/regex pattern for filename matching. Multiple pattens 
          may be present, each one separated by ;
        | recurse (unt): flag to indicate if subdirectories must also be searched (optional)
        | return_folders (int): flag to indicate if folder names must also be returned (optional)
        | useRegex (bool): flag to indicate if patterns areregular expression strings (optional)

    Returns:
        | A list with matching file/directory names

    Raises:
        | No exception is raised.
    """
    if useRegex:
        import re
        
    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    filenames = []
    filertn = []

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
                if useRegex:
                    import re
                    regex = re.compile(pattern)
                    if regex.search(name):
                        filertn.append(name)
                        break
                else:
                    if fnmatch.fnmatch(name, pattern):
                        filertn.append(name)
                        break
    return filertn


def QueryDelete(recurse, tdir, patn, promptUser=True, dryrun=False):
    """Delete files in specified directory.

    recurse: defines if the search must be recursively 0=not, 1=recursive
    tdir: specifies the path
    patn: specifies the file patterns to erase
    promptUser: if true the user is first asked to confirm
    dryrun: if true nothing is erased

    the user is promted before the files are deleted
    """
    thefiles = listFiles(tdir, patn, recurse)
    if thefiles is not None:
        if len(thefiles) > 0:
            if promptUser == True:
                for filename in thefiles:
                    print(filename)
            if promptUser:
                instr = input("Delete these files? (y/n)")
            else:
                instr = 'y'
            if instr == 'y' and not dryrun:
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
        print('Creating some random files')
        self.exts = ['txt', 'bin', 'tmp', 'bot','pot']
        self.dirs = ['.','cleancluttertmp1','cleancluttertmp2']
        print('with extensions: {}'.format(self.exts))
        print('in directories: {}\n'.format(self.dirs))
        stdir = tempfile.mkdtemp()
        for dir in self.dirs:
            self.tdirs.append(os.path.join(stdir,dir))
            if dir == '.':
                self.root = self.tdirs[-1]
            if not os.path.exists(self.tdirs[-1]):
                os.makedirs(self.tdirs[-1])
        self.totalFiles = 53
        for i in range(0,self.totalFiles):
            (handle, path) = tempfile.mkstemp(prefix='cleanclutter',suffix='.'+self.exts[i%len(self.exts)],
                dir=os.path.join('.',self.tdirs[i%len(self.tdirs)]))
            os.close(handle)
            self.paths.append(path)
        self.cntDirs, self.cntAll = self.countFiles()
        print('{} files in {} directories'.format(self.cntAll, self.cntDirs))

    def countFiles(self):
        """count the number of files in total and directories, per type."""
        cntDirs = {}
        cntAll = {}
        for ftype in self.exts:
            cntAll[ftype] = len(listFiles(self.root, patterns='*.{}'.format(ftype), recurse=1, return_folders=0))
        for dir in self.dirs:
            cntDirs[dir] = {}
            for ftype in self.exts:
                cntDirs[dir][ftype] = len(listFiles(os.path.join(self.root,dir),
                    patterns='*.{}'.format(ftype), recurse=0, return_folders=0))
        return cntDirs, cntAll

    def tearDown(self):
        print('Cleaning up.')
        for filename in self.paths:
            os.remove(filename) if os.path.exists(filename) else None
        for dirname in self.tdirs:
            shutil.rmtree(dirname) if os.path.exists(dirname) else None

    def test_listFiles(self):
        """Count the number of files in total and count the number of files per
        directory: the totals per type should agree."""
        print('Assert that the correct number of files in appropriate directories exist')
        self.chkDirs = {keyt: sum([self.cntDirs[keyd][keyt] for keyd in self.dirs]) for keyt in self.exts}
        self.assertTrue(self.chkDirs==self.cntAll)
        self.assertTrue(self.totalFiles==sum([self.cntAll[key] for key in self.cntAll.keys()]))

    def test_QueryDelete(self):
        """Delete given types and in given directories and count after deletion."""
        import time
        print('Delete various combinations of types and dirs and check the count.')
        ftype = self.exts[0]
        dir = self.dirs[-1]
        dpath = os.path.join(self.root,dir)
        QueryDelete(0,dpath,'*.{}'.format(ftype),promptUser=False)
        time.sleep(0.3)
        self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

        ftype = self.exts[1]
        dpath = os.path.join(self.root,'.')
        QueryDelete(1,dpath,'*.{}'.format(ftype),promptUser=False)
        time.sleep(0.3)
        for dir in self.cntDirs.keys():
            self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

        dir = self.dirs[1]
        dpath = os.path.join(self.root,dir)
        QueryDelete(0,dpath,'*.*',promptUser=False)
        time.sleep(0.3)
        for ftype in self.exts:
            self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)

        dpath = os.path.join(self.root,'.')
        QueryDelete(1,dpath,'*.*',promptUser=False)
        time.sleep(0.3)
        for ftype in self.exts:
            for dir in self.cntDirs.keys():
                self.cntDirs[dir][ftype] = 0
        cntDirs, cntAll = self.countFiles()
        self.assertTrue(self.cntDirs==cntDirs)
        print('Completed testing.')

def doUnitTest():
    print('Doing unit test...')
    tt = cleanClutterTest()
    tt.setUp()
    tt.test_listFiles()
    tt.test_QueryDelete()
    tt.tearDown()

################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    # Script lives in clutter/ — resolve the pyradi package directory (one level up)
    _pyradi = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'clutter')
    print(f'Cleaning up in {_pyradi}')

    doUnit = False

    if doUnit:
        doUnitTest()
    else:
        # We take the conservative approach and do not do blanket erase,
        # rather do it by type, asking the user first.
        # All paths are relative to the pyradi package directory (one level up from clutter/).
        QueryDelete(0, _pyradi, '*.eps;*.png;*.jpg;*.pdf;*.txt;*.tiff;*.dat;*.lut')
        QueryDelete(0, _pyradi, 'tape7-*.txt;arr*.txt;Traje*.txt;trian*.txt;vertex*.txt;*.testing')
        QueryDelete(0, _pyradi, 'arr*.txt;colourcoordinates.*;tar;*.svg')
        QueryDelete(0, _pyradi, '*.hdf5')
        QueryDelete(0, _pyradi, 'sample*.double')

        QueryDelete(0, os.path.join(_pyradi, 'examples'), '*.eps')
