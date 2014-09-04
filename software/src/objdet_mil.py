import ObjectDector

class objdet_mil(ObjectDector):
    
    def __init__(self, 
                 dataset_train,
                 feature_vector_info=fvi_conv4_neuron, 
                 size_window=1,
                 sigma=277,
                 c=0.001,
                 resize_image=SIZE_IMAGE,
                 type_bbox=BBOX_COVERALL ):
        
        ObjectDector.__init__(self)
        