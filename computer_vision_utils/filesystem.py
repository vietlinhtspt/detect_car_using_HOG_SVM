import os 
import sys
from os.path import splitext
from os.path import join
def get_file_list_recursively(root_path):
    file_list = []
    for cur_dir, cur_subdirs, cur_files in os.walk(root_path):
        for file in cur_files:
            f_name, f_ext = splitext(file)
            if f_ext:
                file_list.append(join(cur_dir, file))
                sys.stdout.write('\r[{}] - found {:06d} files...'.format(root_path, len(file_list)))
                sys.stdout.flush()
        

    sys.stdout.write('Get file in ' + root_path + ' done.\n')

    return file_list

if __name__ == "__main__":
    list_dir = get_file_list_recursively("../data")
    print(len(list_dir))
    