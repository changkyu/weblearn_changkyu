import os
import shutil
import numpy
import matplotlib.image as mpimg

def RenameImages(dir_src, dir_dst, ext):
    
    if not os.path.isdir(dir_dst):
        os.makedirs(dir_dst)
        
    filenames = os.listdir(dir_src)
    idx = 0
    for filename in filenames:
        len_filename = len(filename)         
        if ('-001.' + ext) == filename[(len_filename-5-len(ext)):]:
            print filename + 'is skipped because it is duplicated.'
            continue
        
        try:
            image = mpimg.imread(dir_src + '/' + filename)
        except Exception:
            print 'Cannot open ' + filename
            continue
        
        height_image = numpy.size(image,0)
        width_image  = numpy.size(image,1)
        if( height_image < 255 or width_image < 255 ):
            print filename + 'is skipped because it is too small.'
            continue
        
        filename_new = ('%06d.' + ext) % idx
        shutil.copyfile(dir_src + '/' + filename, dir_dst + '/' + filename_new)
        idx = idx + 1
        
def main():
    RenameImages('/z/home/changkyu/dataset/www.google.com/archive/car_1', '/z/home/changkyu/dataset/www.google.com/car_1_rename' , 'jpg')

if __name__ == '__main__':
    main()