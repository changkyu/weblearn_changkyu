import os
from os.path import expanduser

################# liblinear library ##################
PATH_HOME=expanduser("~")
EXEC_LIBLINEAR_TRAIN = PATH_HOME + '/lib/liblinear-weights-1.94/train'
EXEC_LIBLINEAR_TEST  = PATH_HOME + '/lib/liblinear-weights-1.94/predict'
FLAG_LIBLINEAR_QUITE = '-q'

__all__ = ['Train_liblinear']

def Train_liblinear(filepath_train, filepath_model, c, filepath_weights=None, filepath_classweights=None):
            
    # Logical regression (for possibility prediction)
    options_train = '-s 0' 
    
    # Class Weight
    if filepath_classweights!=None:
        
        file_classweight = open(filepath_classweights,'r')
        idx_class = 0
        for line in file_classweight:
            line = line.strip()
            for number in line.split():
                options_train += ' -w' + str(idx_class) + ' ' + number
                idx_class = idx_class + 1
        file_classweight.close()
    
    # Weight
    if filepath_weights!=None:
        options_train += ' -W '  + filepath_weights
        
    # Parameter c
    options_train += ' -c '  + str(c) + ' '                         

    # Train
    os.system(EXEC_LIBLINEAR_TRAIN + ' ' + options_train + ' ' + filepath_train + ' ' + filepath_model)
