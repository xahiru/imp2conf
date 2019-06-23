from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import os
# import itertools

def replace(file_path, pattern):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(pattern+line)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)    
        # with open(os.path.expanduser(file_name)) as f:
        #     raw_ratings = [self.reader.parse_line(line) for line in
        #                    itertools.islice(f, self.reader.skip_lines, None)]

# replace('/Users/xahiru/.surprise_data/Netflix-Dataset/training_set/training_set/mv_0000001test.txt','1,')
# teste
def rendirfiles(path):
    # i = 0
    for filename in os.listdir(path): 
        # dst ="mv_" + str(i) + ".txt"
        # src = path+'/'+filename 
        # dst = path +'/'+ dst 
        replace(path+'/'+filename, str(int(filename[3:-4]))+',')
        # print(filename[3:-4])
          
        # rename() function will 
        # rename all the files 
        # os.rename(src, dst) 
        # i += 1

def removefirstline(path):
    for filename in os.listdir(path):
        with open(path+'/'+filename, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(path+'/'+filename, 'w') as fout:
            fout.writelines(data[1:])

file_path = os.path.expanduser('~/.surprise_data/Netflix-Dataset/training_set/training_set')
# removefirstline(file_path)
rendirfiles(file_path)


