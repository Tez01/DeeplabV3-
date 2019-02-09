import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input,Activation,Concatenate,Add,Dropout,BatchNormalization,Conv2D,DepthwiseConv2D,ZeroPadding2D,AveragePooling2D
from keras import backend as K
WEIGHTS_PATH_X="https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"

class BilinearUpsampling(Layer):
    def __init__(self,upsampling=(2,2),output_size=None,data_format=None,**kwargs):
        super(BilinearUpsampling,self).__init__(**kwargs)
        self.data_format=K.normalize_data_format(data_format)
        self.input_spec=InputSpec(ndim=4)

        if  output_size:
            self.output_size=conv_utils.normalize_tuple(output_size,2,'output_size')
            self.upsampling=none
        else:
            self.output_size=none
            seld.upsampling=conv_utils.normalize_tuple(upsampling,2,'upsampling')
    def compute_output_shape(self,input_shape):
        if self.upsampling:
            height=self.upsampling[0]*input_shape[1] if input_shape[1]\
            is not None else None
            width= self.upsampling[1]* input_shape[2] if input_shape[2] is not \
            None else None
        else:
            height=self.output_size[0]
            width=self.output_size[1]
        return(input_shape[0],height,width,input_shape[3])

    def call(self,inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs,(inputs.shape[1]*self.upsampling[0],inputs.shape[2]*self.upsampling[1],align_corners=True))
        else:
            return K.tf.image.resize_bilinear(inputs,(self.output_size[0],self.output_size[1]),align_corners=True)

    def get_config(self):
        config={'upsampling':self.upsampling,'output_size':self.output_size,'data_format':self.data_format}
        base_config=super(BilinearUpsampling,self).get_config()
        return dict(list(base_config.items())+list(config.items()))

            
