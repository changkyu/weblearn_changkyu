import os
from os.path import expanduser
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
from collections import namedtuple
import time
import scipy.stats
import scipy.ndimage

SIZE_IMAGE     = 227#256

# Path Setting
PATH_HOME=expanduser("~")

PATH_WEBLEARN          = PATH_HOME+"/projects/web_learn/software"
PATH_WEBLEARN_SAVE     = PATH_WEBLEARN+'/save'
PATH_WEBLEARN_RESULTS  = PATH_WEBLEARN+'/results'

FILEPATH_TEST_TEMP         = PATH_WEBLEARN_SAVE+"/temp.test"
FILEPATH_RESULT_TEMP       = PATH_WEBLEARN_SAVE+"/temp.result"

PATH_FIG_SAVE              = PATH_WEBLEARN_RESULTS + '/fig'
FILEPATH_FMT_FIG_SAVE      = PATH_FIG_SAVE +"/%s/%s.fig.png" #/fig/{FLAG}/{NAME_IMAGE}.fig.png

PATH_SAVE_FEATURES         = PATH_WEBLEARN_SAVE+"/features"

FILEPATH_FMT_LABELS_SAVE   = PATH_SAVE_FEATURES + "/%s/%s.%s.labels.npy"          # {NAME_DATASET}/{NAME_IMAGE}.{NAME_FEATURE}
FILEPATH_FMT_FEATURES_SAVE = PATH_SAVE_FEATURES + "/%s/%s.%s.feats.npy"           # {NAME_DATASET}/{NAME_IMAGE}.{NAME_FEATURE}
FILEPATH_FMT_WEIGHTS_SAVE  = PATH_SAVE_FEATURES + "/%s/%s.%s.sigma%d.weights.npy" # {NAME_DATASET}/{NAME_IMAGE}.{NAME_FEATURE}.{SIGMA}

PATH_SAVE_RESULTS          = PATH_WEBLEARN_SAVE+"/res"
FILEPATH_FMT_PREDICTS_SAVE = PATH_SAVE_RESULTS  + "/%s/%s.pred.npy"

FILEPATH_FMT_TRAIN         = PATH_WEBLEARN_SAVE + "/%s.%s.wnd%d.train"
FILEPATH_FMT_WEIGHT        = PATH_WEBLEARN_SAVE + '/%s.%s.wnd%d.sigma%d.weight'
FILEPATH_FMT_CLASSWEIGHT   = PATH_WEBLEARN_SAVE + '/%s.%s.wnd%d.sigma%d.classweight'
FILEPATH_FMT_MODEL         = PATH_WEBLEARN_SAVE + '/%s.%s.wnd%d.sigma%d.c%f.model'

FILEPATH_FMT_TEST          = PATH_WEBLEARN_SAVE + "/%s.%s.wnd%d.test"

FILEPATH_DEFAULT_LOG       = PATH_WEBLEARN + '/log/default.log'
FILEPATH_FMT_LOG           = PATH_WEBLEARN + '/log/%s.log'

# liblinear library
EXEC_LIBLINEAR_TRAIN = PATH_HOME + '/lib/liblinear-weights-1.94/train'
EXEC_LIBLINEAR_TEST  = PATH_HOME + '/lib/liblinear-weights-1.94/predict'
FLAG_LIBLINEAR_QUITE = '-q'

# Training Options
FLAG_WEIGHTS_ONOFF = True
FLAG_SIGMA_ONOFF = False

RANGE_SIGMA = [SIZE_IMAGE, SIZE_IMAGE/2, SIZE_IMAGE/4, SIZE_IMAGE/8] #SIZE_IMAGE/16
RANGE_C     = [0.0001, 0.001, 0.01, 0.1, 1, 10]

BBOX_COVERALL = 'coverall'
BBOX_COVEREX1 = 'coverex1'

# Decaf
sys.path.append(PATH_WEBLEARN+"/3rdparty/decaf-release")
sys.path.append(PATH_WEBLEARN+"/3rdparty/decaf-release/decaf/scripts")

from imagenet import DecafNet
from decaf.util import transform

FeatVecInfo = namedtuple("FeatVecInfo", "name width_feature height_feature dims_feature")
fvi_conv1_neuron  = FeatVecInfo('conv1_neuron_cudanet_out', 55, 55, 96)
fvi_rnorm1        = FeatVecInfo('rnorm1_cudanet_out', 27, 27, 96)
fvi_conv2_neuron  = FeatVecInfo('conv2_neuron_cudanet_out', 27, 27, 256)
fvi_rnorm2        = FeatVecInfo('rnorm2_cudanet_out', 13, 13, 256)
fvi_conv3_neuron  = FeatVecInfo('conv3_neuron_cudanet_out', 13, 13, 384)
fvi_conv4_neuron  = FeatVecInfo('conv4_neuron_cudanet_out', 13, 13, 384)
fvi_conv5_neuron  = FeatVecInfo('conv5_neuron_cudanet_out', 13, 13, 256)
fvi_pool5         = FeatVecInfo('pool5_cudanet_out', 6, 6, 256)
RANGE_FVI = [fvi_conv1_neuron, fvi_rnorm1, fvi_conv2_neuron, fvi_rnorm2, fvi_conv3_neuron, fvi_conv4_neuron, fvi_conv5_neuron, fvi_pool5]


PATH_DATASET          = PATH_HOME     + "/dataset"
PATH_IMAGENET         = PATH_DATASET  + "/imagenet"
PATH_FMT_IMAGES_TRAIN = PATH_IMAGENET + "/images/%s/train"
PATH_FMT_IMAGES_TEST  = PATH_IMAGENET + "/images/%s/test"
PATH_FMT_ANNOTATION   = PATH_IMAGENET + "/Annotation/%s" 


DatasetInfo = namedtuple("DatasetInfo", "name path_images_train path_images_test path_annotations ext_annotation")
dataset_imagenet_n02958343 = DatasetInfo('n02958343', 
                                         PATH_HOME + '/dataset/imagenet/images/n02958343/train', 
                                         PATH_HOME + '/dataset/imagenet/images/n02958343/test',
                                         PATH_HOME + '/dataset/imagenet/Annotation/n02958343',
                                         'xml'                                                        )

dataset_INRIA_cars         = DatasetInfo('INRIAcars',
                                         None, 
                                         PATH_HOME + '/dataset/INRIA/cars/images',
                                         None, #PATH_HOME + '/dataset/INRIA/cars/objects',
                                         None)#'pgm.objects'                                  )

dataset_google_car_1       = DatasetInfo('google',
                                         None, 
                                         PATH_HOME + '/dataset/www.google.com/car_1_rename/',
                                         None,
                                         None)

class TargetDetector(object):
    def __init__(self, 
                 feature_vector_info=fvi_conv4_neuron, 
                 size_window=1, 
                 range_sigma=RANGE_SIGMA, 
                 range_c=RANGE_C, 
                 resize_image=SIZE_IMAGE,
                 type_bbox=BBOX_COVERALL ):
        
        if not os.path.isdir(PATH_WEBLEARN_SAVE):
            os.makedirs(PATH_WEBLEARN_SAVE)
        if not os.path.isdir(PATH_WEBLEARN_RESULTS):
            os.makedirs(PATH_WEBLEARN_RESULTS)
        if not os.path.isdir(PATH_FIG_SAVE):
            os.makedirs(PATH_FIG_SAVE)
        if not os.path.isdir(PATH_SAVE_FEATURES):
            os.makedirs(PATH_SAVE_FEATURES)
        if not os.path.isdir(PATH_SAVE_RESULTS):
            os.makedirs(PATH_SAVE_RESULTS)
        
        # Log
        self.file_log=None

        # DecafNet
        self.decaf = DecafNet()

        # Model Variables        
        self.range_sigma  = range_sigma
        self.n_sigma      = len(range_sigma)
        self.range_c      = range_c
        
        self.size_image   = resize_image
        self.width_image  = self.size_image
        self.height_image = self.size_image 
                
        # Normal pdf for training weights
        self.normal_dist = scipy.stats.norm(0, 1)
        self.normal_pdf_zero    = self.normal_dist.pdf(0)
        self.normal_pdf_zero_sq = self.normal_pdf_zero*self.normal_pdf_zero
        
        # Initialize Feature Vector Info
        self.SetFeatureVectorInfo(feature_vector_info)
        self.SetWindowSize(size_window)
        self.SetBBoxType(type_bbox)

    def SetBBoxType(self, type_bbox):
        self.type_bbox    = type_bbox
        
    def SetFeatureVectorInfo(self, feature_vector_info):
        
        self.fvi = feature_vector_info
         
        self.width_patch  = int(math.floor(self.size_image/float(feature_vector_info.width_feature )))
        self.height_patch = int(math.floor(self.size_image/float(feature_vector_info.height_feature)))
        
        self.offset_x     = (self.size_image - (self.width_patch  * feature_vector_info.width_feature ))/2
        self.offset_y     = (self.size_image - (self.height_patch * feature_vector_info.height_feature))/2
        
    def SetWindowSize(self, size_window):

        self.row_up    = int(math.floor((size_window-1)/float(2)))
        self.row_down  = int(math.ceil( (size_window-1)/float(2)))
        self.col_left  = int(math.floor((size_window-1)/float(2)))
        self.col_right = int(math.ceil( (size_window-1)/float(2)))
        
        self.size_window = size_window
        
    def SetTrainDataset(self, dataset):
        
        filenames_images_valid = GetValidFilenames(dataset.path_images_train, dataset.path_annotations, dataset.ext_annotation)
        
        self.train_name_dataset     = dataset.name
        self.train_path_images      = dataset.path_images_train
        self.train_path_annotation  = dataset.path_annotations
        self.train_filenames_images = filenames_images_valid        
        
        if not os.path.isdir(PATH_SAVE_FEATURES + '/' + self.train_name_dataset):
            os.makedirs(PATH_SAVE_FEATURES + '/' + self.train_name_dataset)
        
    def SetTestDataset(self, dataset):
        
        self.test_name_dataset     = dataset.name
        
    def GetFeature_windowsize(self, row, col, labels=None, feats=None, weights=None):
        
        label  = None
        feat   = None
        weight = None
        
        if self.size_window == 1:
            if labels!=None:
                label  = labels[row][col]
            if feats!=None:
                feat   = feats[row][col]
            if weights!=None:
                weight = weights[row][col]
                        
        else:        
            label = 1
            feat = np.zeros((0,))
            weight= 0
            
            rows = np.size(feats,0)
            cols = np.size(feats,1)
            
            for r in range(row-self.row_up, row+self.row_down+1):
                for c in range(col-self.col_left, col+self.col_right+1):
                    
                    # padding with a reflect way
                    if r < 0:
                        r = -r
                    if c < 0:
                        c = -c
                    if r >= rows:
                        r = (rows-1)*2 - r
                    if c >= cols:
                        c = (cols-1)*2 - c
                    
                    if feats!=None:
                        feat  = np.concatenate( (feat,feats[r][c]), axis=0)
                    if labels!=None:
                        label = label * labels[r][c] # label=1 when all patches have (label==1)
                    if weights!=None:                    
                        weight = weight + weights[r][c]
                        
            weight = weight / (self.size_window*self.size_window)
        
        return (label, feat, weight)
        
    def GetModelFilepath(self, sigma, c):
        
        filepath_model = FILEPATH_FMT_MODEL  % (self.train_name_dataset, self.fvi.name, self.size_window, int(sigma), c)

        # if it is needed to train 
        if not os.path.isfile(filepath_model):
        
            self.LOG('Train Model...')
            
            filepath_train        = FILEPATH_FMT_TRAIN       % (self.train_name_dataset, self.fvi.name, self.size_window)
            filepath_weights      = FILEPATH_FMT_WEIGHT      % (self.train_name_dataset, self.fvi.name, self.size_window, int(sigma))
            filepath_classweights = FILEPATH_FMT_CLASSWEIGHT % (self.train_name_dataset, self.fvi.name, self.size_window, int(sigma))                                        
            
            if not os.path.isfile(filepath_train):
                self.ExtractFeatures_path(self.train_path_images, 
                                          self.train_filenames_images, 
                                          self.train_path_annotation, 
                                          sigma, 
                                          True,
                                          filepath_train,
                                          filepath_weights,
                                          filepath_classweights        )
            
            self.Train(filepath_train, filepath_model, c, filepath_weights, filepath_classweights)            
        else:
            self.LOG('Load Model...')
            
        pt_ext = filepath_model.rfind('.')
        pt_dir = filepath_model.rfind('/')
        filename_model = filepath_model[(pt_dir+1):pt_ext]
        if not os.path.isdir(PATH_SAVE_RESULTS + '/' + filename_model):
            os.makedirs(PATH_SAVE_RESULTS + '/' + filename_model)
            
        return filepath_model
                        
    def PredictBBox(self, 
                    filepath_image,
                    filepath_model,
                    filepath_annotation=None, 
                    load_pred_history=True,
                    is_save_fig=False,
                    filepath_fig=None         ): 
        """
            Input: 
                vis: visualization
                save_fig: save result figure
                path_save_fig: path to save figures
        """        
        
        exist_annotation = filepath_annotation!=None and os.path.isfile(filepath_annotation)
        
        # Prediction from Image
        pred = self.Predict_image(filepath_image, filepath_model, is_quite=True, load_pred_history=load_pred_history)
        rows = np.size(pred,0)
        cols = np.size(pred,1)        
        pred_gaussian = scipy.ndimage.filters.gaussian_filter(pred, sigma=min(rows,cols)/13)        
        
        # Get Bounding Box from the prediction
        bbox_pred          = self.GetBBox_pred(pred)
        bbox_pred_gaussian = self.GetBBox_pred(pred_gaussian)
        
        if exist_annotation or is_save_fig:            
            image = mpimg.imread(filepath_image)
        
        # Compute Overlapping Ratio between Ground Truth and Prediction 
        if exist_annotation:
            bbox_ann = self.GetBBox_annotation(filepath_annotation, image.shape[1], image.shape[0], SIZE_IMAGE)
            overlap_ratio          = round( self.GetOverlapRatio(bbox_ann, bbox_pred), 2)
            overlap_ratio_gaussian = round( self.GetOverlapRatio(bbox_ann, bbox_pred_gaussian), 2)
        
        # Visualization    
        if is_save_fig==True:                
            
            # Draw Image
            image = transform.scale_and_extract(transform.as_rgb(image), 256)
            image = DecafNet.oversample(image, True)[0]            
            
            plt.subplot(221)
            plt.imshow(image)
            
            plt.subplot(222)
            plt.imshow(image)
            
            plt.xlim([0,image.shape[1]])
            plt.ylim([image.shape[0],0])

            # Draw Prediction Bounding Box
            self.Draw_bbox(bbox_pred, 'r')
            self.Draw_bbox(bbox_pred_gaussian, 'y')
            
            # Draw Ground Truth Bounding Box           
            if exist_annotation:
                self.Draw_bbox( bbox_ann, (0,1,0))
                
                plt.subplot(223)
                plt.text(1, -1, str(overlap_ratio) + '%')
                
                plt.subplot(224)
                plt.text(1, -1, str(overlap_ratio_gaussian) + '%')
        
            # Draw Potential map
            plt.subplot(223)
            plt.imshow(pred)
            
            cmin = (bbox_pred[0] - self.offset_x)/self.width_patch
            cmax = (bbox_pred[2] - self.offset_x + 1)/self.width_patch - 1
            rmin = (bbox_pred[1] - self.offset_y)/self.height_patch
            rmax = (bbox_pred[3] - self.offset_y + 1)/self.height_patch - 1            
            self.Draw_bbox([cmin, rmin, cmax, rmax], 'w')
            
            plt.subplot(224)
            plt.imshow(pred_gaussian)
            
            cmin = (bbox_pred_gaussian[0] - self.offset_x)/self.width_patch
            cmax = (bbox_pred_gaussian[2] - self.offset_x + 1)/self.width_patch - 1
            rmin = (bbox_pred_gaussian[1] - self.offset_y)/self.height_patch
            rmax = (bbox_pred_gaussian[3] - self.offset_y + 1)/self.height_patch - 1            
            self.Draw_bbox([cmin, rmin, cmax, rmax], 'w')
            
            # Save figure in a file
            if filepath_fig==None:       
                                
                pt_ext = filepath_model.rfind('.')
                pt_dir = filepath_model.rfind('/')                
                name_model = filepath_model[(pt_dir+1):pt_ext]
                if not os.path.isdir(PATH_FIG_SAVE + '/' + name_model + '/' + self.test_name_dataset):
                    os.makedirs(PATH_FIG_SAVE + '/' + name_model + '/' + self.test_name_dataset)
                
                pt_ext = filepath_image.rfind('.')
                pt_dir = filepath_image.rfind('/')
                filepath_fig=(FILEPATH_FMT_FIG_SAVE % (name_model + '/' + self.test_name_dataset,filepath_image[(pt_dir+1):pt_ext]))
                
            plt.savefig(filepath_fig)
        
        if exist_annotation:    
            return (overlap_ratio, overlap_ratio_gaussian)
        else:
            return (None, None)
                    
    def Predict_image(self, filepath_image, filepath_model, is_quite=False, load_pred_history=False):
        """
            Input:
                image: matplotlib image
                filepath_model: filepath for model
                fvi: feature vector information
                is_quite: don't show testing output
            Output:
                pred: numpy array of prediction results (n_instance x 1)

        """        
        
        pt_ext = filepath_image.rfind('.')
        pt_dir = filepath_image.rfind('/')
        filename_image = filepath_image[(pt_dir+1):pt_ext]
        
        pt_ext = filepath_model.rfind('.')
        pt_dir = filepath_model.rfind('/')
        filename_model = filepath_model[(pt_dir+1):pt_ext]        
                
        filepath_pred_save = FILEPATH_FMT_PREDICTS_SAVE % (filename_model, filename_image)
        
        if load_pred_history==True and os.path.isfile(filepath_pred_save ):
            pred_reshape = np.load(filepath_pred_save)
        else:        
            # Get middle level of feature vector            
            (_, feats, _) = self.ExtractFeatures_image(filepath_image)
            
            rows       = np.size(feats,0)
            cols       = np.size(feats,1)
            dims_feats = np.size(feats,2)
            
            # Write an Input File
            file_test = open(FILEPATH_TEST_TEMP, 'w')        
            str_tmp = ''        
            for r in range(0,rows):
                for c in range(0,cols):
                    
                    str_tmp = '-1'
                    for idx_feat in range(0,dims_feats):
                        str_tmp += ' ' + str(idx_feat+1) + ':' + str(feats[r][c][idx_feat])
                    str_tmp += '\r\n'
                    file_test.write(str_tmp)
            
            file_test.close()
            
            # Test Options
            
            str_options = '-b 1' # estimate probability
            
            if is_quite:
                str_options = str_options + ' ' + FLAG_LIBLINEAR_QUITE
            
            
            os.system(EXEC_LIBLINEAR_TEST  + ' ' + str_options + ' ' + FILEPATH_TEST_TEMP  + ' ' + filepath_model + ' ' + FILEPATH_RESULT_TEMP)
            
            # Read an Output File
            pred = np.genfromtxt(FILEPATH_RESULT_TEMP,skiprows=1)
                    
            pred_reshape = np.zeros((rows,cols))
            idx=0
            for r in range(0,rows):
                for c in range(0,cols):
                    pred_reshape[r][c] = pred[idx][2]
                    idx = idx + 1
            
            np.save(filepath_pred_save, pred_reshape)
                
        return np.flipud(pred_reshape)
        
    def Train(self, filepath_train, filepath_model, c, filepath_weights=None, filepath_classweights=None, is_print_accuracy=False):
                
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
           
        # Accuracy
        if( is_print_accuracy==True ):
            os.system(EXEC_LIBLINEAR_TEST + ' ' + filepath_train + ' ' + filepath_model + ' ' + FILEPATH_RESULT_TEMP)            
            file_result = open(FILEPATH_RESULT_TEMP,'r')
            pred = np.fromfile(file_result,dtype=int,sep='\r\n')
            file_result.close()
            self.LOG('Train Accuracy (' + filepath_model + '): ')
            self.LOG(str( round( (1 - np.sum( np.abs( np.subtract( self.labels , pred ) ) ) / self.n_feats) * 100 ,2) ) + '%' )
        
    def ExtractFeatures_image(self, 
                              filepath_image,
                              filepath_annotation=None,
                              sigma=None):
        
        labels  = None
        feats   = None
        weights = None
        
        if os.path.isfile(filepath_image):
            
            pt_ext = filepath_image.rfind('.')
            pt_dir = filepath_image.rfind('/')
            filename_image = filepath_image[(pt_dir+1):pt_ext]
            
            filepath_features_save = FILEPATH_FMT_FEATURES_SAVE % (self.train_name_dataset, filename_image, self.fvi.name)
            if not os.path.isfile(filepath_features_save):
                # Read Image
                image = mpimg.imread(filepath_image)
                    
                # Get middle level of feature vector            
                self.decaf.classify(image, True)
                feats   = self.decaf.feature(self.fvi.name)[0]
                
                # Save Feats
                np.save(filepath_features_save, feats )                
            else:
                # Load Feats
                feats   = np.load(filepath_features_save)
                
            rows_feats = np.size(feats,0)
            cols_feats = np.size(feats,1)
            dims_feats = np.size(feats,2)
            
            if (sigma!=None) and (filepath_annotation!=None):
                
                filepath_labels_save  = FILEPATH_FMT_LABELS_SAVE  % (self.train_name_dataset, filename_image, self.fvi.name)
                filepath_weights_save = FILEPATH_FMT_WEIGHTS_SAVE % (self.train_name_dataset, filename_image, self.fvi.name, sigma)
                
                if ((not os.path.isfile(filepath_labels_save)) or 
                    (not os.path.isfile(filepath_weights_save))   ):            
                    
                    # Read Annotation File
                    if os.path.isfile(filepath_annotation):
                        image = mpimg.imread(filepath_image)
                        bbox    = self.GetBBox_annotation(filepath_annotation, image.shape[1], image.shape[0], self.size_image)
                        cx_bbox = (bbox[0] + bbox[2]) / 2
                        cy_bbox = (bbox[1] + bbox[3]) / 2
                        
                        # Count positive / negative training patches
                        labels  = np.zeros( (rows_feats,cols_feats) )
                        weights = np.ones(  (rows_feats,cols_feats) )
                        
                        for r in range(0,rows_feats):
                            ymin = self.offset_y + self.height_patch * r                        
                            ymax = ymin + self.height_patch - 1
                            cy = (ymin + ymax) / 2
                            
                            if bbox[1] <= ymin and ymax <= bbox[3]:
                                for c in range(0,cols_feats):
                                    xmin = self.offset_x + self.width_patch*c                                
                                    xmax = xmin + self.width_patch - 1
                                    cx = (xmin + xmax) / 2                            
                                                                    
                                    if bbox[0] <= xmin and xmax <= bbox[2]:
                                        labels[r][c]  = 1
                                        weights[r][c] = (self.normal_dist.pdf( (cx - cx_bbox)/sigma ) 
                                                                  * self.normal_dist.pdf( (cy - cy_bbox)/sigma ) / self.normal_pdf_zero_sq)
                
                        np.save(filepath_labels_save,  labels  )
                        np.save(filepath_weights_save, weights )
                    else:
                        self.LOG('[Error] ExtractFeatures_image... Cannot Read Annotation File: ' + filepath_annotation)
                else:
                    labels  = np.load(filepath_labels_save)
                    weights = np.load(filepath_weights_save)
                    
                # Concatenate feature vectors according to window size
                shape_ret  = (rows_feats,cols_feats)
                retlabels  = np.zeros( (rows_feats,cols_feats)  )
                retfeats   = np.zeros( (rows_feats,cols_feats) + (dims_feats*self.size_window*self.size_window,) )
                retweights = np.zeros( (rows_feats,cols_feats) ) 
                
                for r in range(0, rows_feats):
                    for c in range(0,cols_feats):
                        (retlabels[r][c], 
                         retfeats[r][c], 
                         retweights[r][c]) = self.GetFeature_windowsize(r, c, labels, feats, weights)
        
                return (retlabels, retfeats, retweights)
            else:
                shape_ret  = (rows_feats,cols_feats)
                retfeats   = np.zeros( (rows_feats,cols_feats) + (dims_feats*self.size_window*self.size_window,) )
                for r in range(0, rows_feats):
                    for c in range(0,cols_feats):
                        (_, 
                         retfeats[r][c], 
                         _              ) = self.GetFeature_windowsize(r, c, None, feats, None)
        
                return (None, retfeats, None)
        else:
            self.LOG('[Error] ExtractFeatures_image... Cannot Read Image File: ' + filepath_image)
            
    def ExtractFeatures_path(self, 
                             path_images, 
                             filenames_images=None,
                             path_annotation=None, 
                             sigma=None,
                             write_as_file=False,
                             filepath_feats=None,
                             filepath_weights=None,
                             filepath_classweights=None    ):

        if filenames_images==None:
            filenames_images = os.listdir(path_images)

        # For efficient memory usage, count the number of negative and positive patches first
        n_images = 0
        filenames_images_valid = []
        for filename_img in filenames_images:
            
            filepath_image = path_images + "/" + filename_img
            if (not os.path.isfile(filepath_image)):
                print 'Skip... there does not exist: ' + filepath_image
                continue
            
            if (path_annotation!=None):
                pt_ext = filename_img.rfind('.')
                filepath_annotation = path_annotation + "/" + filename_img[0:pt_ext] + '.xml'
                if (not os.path.isfile(filepath_annotation)):
                    print 'Skip... there does not exist: ' + filepath_annotation
                    continue
            
            filenames_images_valid.append(filename_img)
            n_images = n_images + 1
            
        # Allocate array space
        if write_as_file==False:
            feats   = np.zeros((n_images, self.fvi.height_feature, self.fvi.width_feature, self.fvi.dims_feature)) 
            labels  = np.zeros((n_images, ))
            weights = np.zeros((n_images, ))    
        
        fmt = '%' + str(len(str(n_images))) + 'd / ' + str(n_images) # print progress                            

        # Get feature vectors for each patches
        idx = 0
        w_pos = 0;
        w_neg = 0;
        for filename_img in filenames_images_valid:
            
            filepath_image          = path_images     + "/" + filename_img
            if (path_annotation!=None):
                filepath_annotation = path_annotation + "/" + filename_img[0:filename_img.rfind('.')] + '.xml'
            else:
                filepath_annotation = None
                        
            if write_as_file==False:
                (labels[idx], feats[idx], weights[idx]) = self.ExtractFeatures_image(filepath_image, filepath_annotation, sigma)
            else:
                (labels, feats, weights) = self.ExtractFeatures_image(filepath_image, filepath_annotation, sigma)
                self.WriteFeatureAsFile(labels, feats, filepath_feats, weights, filepath_weights, True)
                
                rows_weights = np.size(weights,0)
                cols_weights = np.size(weights,1)
                for r in range(0,rows_weights):
                    for c in range(0,cols_weights):
                        if (labels[r][c] == 1):
                            w_pos += weights[r][c]
                        else:
                            w_neg += weights[r][c]
            
            print fmt % (idx+1)
            idx = idx + 1
        
        if write_as_file==True:    
            w_all = w_pos + w_neg                
            file_classweight = open(filepath_classweights, 'w')
            file_classweight.write(str(1 - w_neg/w_all) + ' ' + str(1 - w_pos/w_all))
            file_classweight.close()
        else:
            return (labels, feats, weights)
    
    def WriteClassweightsAsFile(self, weights=None, filepath_weights=None):
    
        w_pos = 0;
        w_neg = 0;
            
        if weights!=None:
        
            n_images     = np.size(weights,0)
            rows_weights = np.size(weights,1)
            cols_weights = np.size(wieghts,2)
            
            for i in range(0,n_images):
                for r in range(0,rows_weights):
                    for c in range(0,cols_weights):
                        if (labels[i][r][c] == 1):
                            w_pos += weights[i][r][c]
                        else:
                            w_neg += weights[i][r][c]
                    
            w_all = w_pos + w_neg
            
        elif filepath_weights!=None:
            
            
            
            rows_weights = np.size(weights,0)
            cols_weights = np.size(weights,1)
                                
            for r in range(0,rows_weights):
                for c in range(0,cols_weights):
                    if (labels[r][c] == 1):
                        w_pos += weights[r][c]
                    else:
                        w_neg += weights[r][c]
                
            w_all = w_pos + w_neg
            
            file_classweight = open(filepath_classweights, option)
            file_classweight.write(str(1 - w_neg/w_all) + ' ' + str(1 - w_pos/w_all))
            file_classweight.close()
    
    def WriteFeatureAsFile(self,
                           labels=None, feats=None,  filepath_feats=None, 
                           weights=None,             filepath_weights=None, 
                           is_append=True):
        if is_append==True:
            option = 'a'
        else:
            option = 'w'
        
        if labels!=None and feats!=None and filepath_feats!=None:
            
            file_feat = open(filepath_feats, option)            
            
            b_multi = len( np.shape(labels) )>2
            if b_multi:
                
                n_images   = np.size(feats,0)
                rows_feats = np.size(feats,1)
                cols_feats = np.size(feats,2)
                dim_feats  = np.size(feats,3)
                
                for i in range(0,n_images):
                    for r in range(0,rows_feats):
                        for c in range(0,cols_feats):
                            str_tmp = str(int(labels[i][r][c]))            
                            for idx_feat in range(0,dim_feats):
                                str_tmp += ' ' + str(idx_feat+1) + ':' + str(feats[i][r][c][idx_feat])
                            str_tmp += '\r\n'
                            file_feat.write(str_tmp)
            else:
                
                rows_feats = np.size(feats,0)
                cols_feats = np.size(feats,1)
                dim_feats  = np.size(feats,2)
                            
                for r in range(0,rows_feats):
                    for c in range(0,cols_feats):
                        str_tmp = str(int(labels[r][c]))            
                        for idx_feat in range(0,dim_feats):
                            str_tmp += ' ' + str(idx_feat+1) + ':' + str(feats[r][c][idx_feat])
                        str_tmp += '\r\n'
                        file_feat.write(str_tmp)
                        
            file_feat.close()            
                
        if weights!=None and filepath_weights!=None:
            
            file_weight = open(filepath_weights, option)
                            
            b_multi = len( np.shape(weights) )>2
            if b_multi:
                
                n_images     = np.size(weights,0)
                rows_weights = np.size(weights,1)
                cols_weights = np.size(wieghts,2)
                
                for i in range(0,n_images):
                    str_tmp = ''                
                    for r in range(0,rows_weights):
                        for c in range(0,cols_weights):
                            str_tmp += str(weights[i][r][c]) + '\r\n'            
                    file_weight.write(str_tmp)
                
            else:
                rows_weights = np.size(weights,0)
                cols_weights = np.size(weights,1)
    
                str_tmp = ''
                for r in range(0,rows_weights):
                    for c in range(0,cols_weights):
                        str_tmp += str(weights[r][c]) + '\r\n'            
                file_weight.write(str_tmp)

            file_weight.close()

    def GetOverlapRatio(self, bbox_A, bbox_B):
        """
            Get Overlapping Ratio between two bounding boxes
            
            Input: two bounding boxes
            
            Output: overlapping ratio                 
        """
        xmin_comm = max(bbox_A[0], bbox_B[0])
        ymin_comm = max(bbox_A[1], bbox_B[1])
        xmax_comm = min(bbox_A[2], bbox_B[2])
        ymax_comm = min(bbox_A[3], bbox_B[3])
        
        area_A = (bbox_A[2]-bbox_A[0]) * (bbox_A[3]-bbox_A[1])
        area_B = (bbox_B[2]-bbox_B[0]) * (bbox_B[3]-bbox_B[1])
        area_comm = (xmax_comm-xmin_comm) * (ymax_comm-ymin_comm)
        
        return area_comm / float(max(area_A, area_B))

    def GetBBox_pred(self, pred, is_save_fig=False):
        
        rows = np.size(pred,0)
        cols = np.size(pred,1)

        rmin = rows-1
        rmax = 0
        cmin = cols-1
        cmax = 0
        
        if (self.type_bbox==BBOX_COVEREX1):
            for r in range(0,rows):            
                for c in range(0,cols):
                    if (pred[r][c] > 0.5):
                        if ( (c==0        or pred[r][c-1] <= 0.5) and # left                     
                             (r==0        or pred[r-1][c] <= 0.5) and # up
                             (c==(cols-1) or pred[r][c+1] <= 0.5) and # right
                             (r==(rows-1) or pred[r+1][c] <= 0.5)     # down
                        ):
                            color = 'w'                            
                        else:      
                            color = 'y'
                            if r < rmin:
                                rmin = r
                            if r > rmax:
                                rmax = r
                            if c < cmin:
                                cmin = c
                            if c > cmax:
                                cmax = c

                        # Draw [X]
                        if is_save_fig==True:
                            ymin = self.offset_y + r*self.height_patch
                            ymax = self.offset_y + (r+1)*self.height_patch - 1
                            xmin = self.offset_x + c*self.width_patch
                            xmax = self.offset_x + (c+1)*self.width_patch  - 1
                            plt.plot([xmin,xmax],[ymin,ymax],color)
                            plt.plot([xmax,xmin],[ymin,ymax],color) 
                
        elif (self.type_bbox==BBOX_COVERALL):
            for r in range(0,rows):
                for c in range(0,cols):
                    if (pred[r][c] > 0.5):
                        if r < rmin:
                            rmin = r
                        if r > rmax:
                            rmax = r                        
                        if c < cmin:
                            cmin = c
                        if c > cmax:
                            cmax = c
                            
                        # Draw [X]                                    
                        if is_save_fig==True:
                            ymin = self.offset_y + r*self.height_patch
                            ymax = self.offset_y + (r+1)*self.height_patch - 1
                            xmin = self.offset_x + c*self.width_patch
                            xmax = self.offset_x + (c+1)*self.width_patch  - 1
                            plt.plot([xmin,xmax],[ymin,ymax],'y')
                            plt.plot([xmax,xmin],[ymin,ymax],'y') 
                
        ymin = self.offset_y + rmin*self.height_patch
        ymax = self.offset_y + (rmax+1)*self.height_patch - 1
        xmin = self.offset_x + cmin*self.width_patch
        xmax = self.offset_x + (cmax+1)*self.width_patch  - 1
        
        return [xmin,ymin,xmax,ymax]
        

    def GetBBox_annotation(self, filepath_annotation, width_org, height_org, width_reshape, height_reshape=None):
        """
        Get ground truth bound box from annotation xml file
        
        Input:
            filepath_annotation: file path
            width_image:  the width of original image
            height_image: the height of original image
        Output:
            scores: a numpy vector of size 1000 containing the
                predicted scores for the 1000 classes.
        """
        pt_ext = filepath_annotation.rfind('.')
        ext_annotation = filepath_annotation[(pt_ext+1):]
        if ext_annotation=='xml':
            
            # read Annotation
            annotation = ET.parse(filepath_annotation).getroot()

            # Take just the first object annotation
            e_object = annotation.find('object')
            e_bndbox = e_object.find('bndbox')
            xmin = int(e_bndbox.find('xmin').text)
            xmax = int(e_bndbox.find('xmax').text)
            ymin = int(e_bndbox.find('ymin').text)            
            ymax = int(e_bndbox.find('ymax').text)
        
        elif ext_annotation=='objects':
            
            file_annotation = open(filepath_annotation,'r')
            idx_class = 0
            for line in file_annotation:
                line = line.strip()
                bbox = line.split()
                bbox = bbox[1:]
                break                    
            file_annotation.close()
            
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0])+int(bbox[2])
            ymax = int(bbox[1])+int(bbox[3])
            
        else:
            self.LOG('[Error] Invalid Annotation Type')
            
        # Adjust bbox for the Decaf's fixed size image
        # Note that Decaf's transformation function returns a cropped image instead of distortion
        if not height_reshape:
            height_reshape = width_reshape
        
        ratio    = max(width_reshape/float(width_org) , height_reshape/float(height_org))            
        x_offset = -(width_org *ratio - width_reshape) / 2
        y_offset = -(height_org*ratio - height_reshape) / 2
        return np.array([ max(x_offset + xmin*ratio,0), 
                          max(y_offset + ymin*ratio,0),  
                          min(x_offset + xmax*ratio,SIZE_IMAGE-1),  
                          min(y_offset + ymax*ratio,SIZE_IMAGE-1)  ])
        
    def Draw_bbox(self, bbox, color='y', linewidth=2):
        """
        Draw bounding box on figure
            Input: 
                bbox: bounding box (xmin, ymin, xmax, ymax)
                color: color of boudning box
                axes: canvas
        """
        plt.vlines(bbox[0], bbox[1], bbox[3], color, linewidth=linewidth)
        plt.vlines(bbox[2], bbox[1], bbox[3], color, linewidth=linewidth)
        plt.hlines(bbox[1], bbox[0], bbox[2], color, linewidth=linewidth)
        plt.hlines(bbox[3], bbox[0], bbox[2], color, linewidth=linewidth)
        
    def LOG(self, str_log, filepath_log=None, with_newline=True):
        print str_log
        if filepath_log==None:
            if self.file_log==None:
                self.file_log = open(FILEPATH_DEFAULT_LOG,'w')
        else:
            if self.file_log!=None:
                self.file_log.close()
            self.file_log = open(filepath_log,'w')
            
        self.file_log.write( str_log )
        if with_newline:
            self.file_log.write( '\r\n' )
                        
    def __exit__(self, type, value, traceback):
        if self.file_log!=None:
            self.file_log.close()
            self.file_log=None
    
def GetValidFilenames(path_images, path_annotations, ext_annotation):

    filenames_images_valid = []
    filenames_images = os.listdir(path_images)
    for filename_img in filenames_images:
        
        filepath_image = path_images + "/" + filename_img
        if (not os.path.isfile(filepath_image)):
            print 'Skip... there does not exist: ' + filepath_image
            continue
        
        pt_ext = filename_img.rfind('.')            
        filepath_annotation = path_annotations + "/" + filename_img[0:pt_ext] + '.' + ext_annotation
        if (not os.path.isfile(filepath_annotation)):
            print 'Skip... there does not exist: ' + filepath_annotation
            continue
        
        filenames_images_valid.append(filename_img)

    return filenames_images_valid
    
tic = 0
toc = 0
def TicToc(msg):    
    global tic
    global toc
    
    toc = time.clock()    
    t = (toc - tic)    
    tic = time.clock()
    print msg + ' Time: ' + str(t)
    
def main():
    
    load_pred_history = True
    is_save_fig = False
    
    # Setting for training
    td = TargetDetector()
    td.SetTrainDataset( dataset_imagenet_n02958343 )
    
    for dataset in (dataset_imagenet_n02958343, dataset_google_car_1,dataset_INRIA_cars,):
                
        # Setting for testing
        td.SetTestDataset(dataset)    
        path_test = dataset.path_images_test
        if dataset.path_annotations!=None:    
            filenames_test = GetValidFilenames(path_test, dataset.path_annotations, dataset.ext_annotation)
        else:    
            filenames_test = os.listdir(path_test)
        if not os.path.isdir(PATH_WEBLEARN_RESULTS + '/fig'):
            os.makedirs(PATH_WEBLEARN_RESULTS + '/fig')
        
        for type_bbox in (BBOX_COVERALL,):#(BBOX_COVERALL, BBOX_COVEREX1):
            td.SetBBoxType(type_bbox)
            for sigma in (RANGE_SIGMA[0],): #RANGE_SIGMA:
                for c in (0.001,): #RANGE_C:
                    for fvi in RANGE_FVI: #(fvi_rnorm1,fvi_conv4_neuron,):#
                        td.SetFeatureVectorInfo(fvi)        
                        for size_window in range(1,6):
                        
                            """
                            if fvi.name == fvi_rnorm1.name :
                                size_window=3
                            elif fvi.name == fvi_conv4_neuron.name :
                                size_window=2
                            """
                            
                            td.SetWindowSize(size_window)
                            filepath_log = FILEPATH_FMT_LOG % (dataset.name + '.' + fvi.name + '.wnd' + str(size_window) + '.sigma' + str(int(sigma)) + '.c' + str(c) + '.bbox' + type_bbox)        
                            td.LOG('------------------------',filepath_log)
                            td.LOG('feat  : ' + fvi.name)
                            td.LOG('bbox  : ' + type_bbox            )
                            td.LOG('sigma : ' + str(sigma)           )
                            td.LOG('c     : ' + str(c)               )
                            td.LOG('wnd   : ' + str(size_window)     )
                            td.LOG('Load Pred History : ' + str(load_pred_history))
                        
                            # Load Model
                            filepath_model = td.GetModelFilepath(sigma, c)
                            
                            n_images = len(filenames_test)
                            fmt = '%' + str(len(str(n_images))) + 'd / ' + str(n_images) # print progress
                            scores          = np.zeros((n_images,1))                        
                            scores_gaussian = np.zeros((n_images,1))
                
                            idx = 0
                            for filename in filenames_test:
                                
                                filepath_image = dataset.path_images_test + "/" + filename
                                
                                if dataset.path_annotations!=None:
                                    pt_ext = filename.rfind('.')
                                    filepath_annotation = dataset.path_annotations + ('/' + filename[0:pt_ext] + '.' + dataset.ext_annotation)
                                    if not os.path.isfile(filepath_annotation):
                                        continue
                                else:
                                    filepath_annotation = None
                                
                                plt.close()
                                plt.figure()
                            
                                (score, score_gaussian) = td.PredictBBox(   filepath_image,
                                                                            filepath_model,
                                                                            filepath_annotation, 
                                                                            load_pred_history,
                                                                            is_save_fig               )                            
                            
                                if( score != None ):
                                    scores[idx] = score
                                    scores_gaussian[idx] = score_gaussian
                                
                                idx = idx + 1                            
                                print 'test ' + (fmt % idx)
                            
                            td.LOG('# of image: ' + str(idx))                
                            td.LOG('Test Accuracy: ' + '(before:' + str( round(np.sum(scores)/float(idx) * 100,2) ) + '), ' 
                                                     + '(gaussian:' + str( round(np.sum(scores_gaussian)/float(idx) * 100,2) ) + ')'
                                                     )
        
if __name__ == '__main__':
    main()

