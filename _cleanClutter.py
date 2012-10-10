#clean a precisely defined set of files from a precisely defined directory set.
#the specified files are deleted, after the user has been prompted


import os.path, fnmatch

#lists the files in a directory and subdirectories (from Python Cookbook)
def listFiles(root, patterns='*', recurse=1, return_folders=0):
    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    # Collect input and output arguments into one bunch
    class Bunch:
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

#function to delete files in specified directory
#  first parameter defines if the search must be recursively 0=not, 1=recursive
#  second parameter specifies the path
#  third parameter specifies the file patterns to erase
#  the user is promted before the files are deleted
def QueryDelete(recurse,dir,patn):
    thefiles = listFiles(dir, patn,recurse)
    if len(thefiles)>0:
        for filename in thefiles:
            print filename
        instr=raw_input("Delete these files? (y/n)")
        if instr=='y':
            for filename in thefiles:
                os.remove(filename)



#we take the conservative approach and do not do blanket erase,
#rather do it by type, asking the user first
QueryDelete(0,'.', '*.eps;*.png')






