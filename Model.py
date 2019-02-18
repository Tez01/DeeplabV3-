import numpy as np

from keras.models import Model
from keras import layers
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers import Input,Activation,Concatenate,Add,Dropout,BatchNormalization,Conv2D,DepthwiseConv2D,ZeroPadding2D,AveragePooling2D
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
WEIGHTS_PATH_X="https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"

class BilinearUpsampling(Layer):
    def __init__(self,upsampling=(2,2),output_size=None,data_format=None,**kwargs):
        super(BilinearUpsampling,self).__init__(**kwargs)
        self.data_format=K.normalize_data_format(data_format)
        self.input_spec=InputSpec(ndim=4)

        if  output_size:
            self.output_size=conv_utils.normalize_tuple(output_size,2,'output_size')
            self.upsampling=None
        else:
            self.output_size=None
            self.upsampling=conv_utils.normalize_tuple(upsampling,2,'upsampling')
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
            return K.tf.image.resize_bilinear(inputs,(inputs.shape[1]*self.upsampling[0],inputs.shape[2]*self.upsampling[1]),align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs,(self.output_size[0],self.output_size[1]),align_corners=True)

    def get_config(self):
        config={'upsampling':self.upsampling,'output_size':self.output_size,'data_format':self.data_format}
        base_config=super(BilinearUpsampling,self).get_config()
        return dict(list(base_config.items())+list(config.items()))

def SepConv_BN(x,filters,prefix,stride=1,kernel_size=3,rate=1,depth_activation=False,epsilon = 1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    x= DepthwiseConv2D((kernel_size,kernel_size),strides=(stride,stride),dilation_rate=(rate,rate),padding=depth_padding,use_bias=False,name=prefix+'_depthwise')(x)
    x=BatchNormalization(name=prefix+'_depthwise_BN',epsilon=epsilon)(x)
    if depth_activation:
        x= Activation('relu')(x)
    x=Conv2D(filters,(1,1),padding='same',use_bias=False,name=prefix+'_pointwise')(x)
    x=BatchNormalization(name=prefix+'_pointwise_BN',epsilon=epsilon)(x)
    if depth_activation:
        x=Activation('relu')(x)

    return x

def _conv2d_same(x,filters,prefix,stride=1, kernel_size=3, rate =1):
    if stride==1:
        return Conv2D(filters,(kernel_size,kernel_size),strides=(stride,stride),
        padding='same',use_bias=False,dilation_rate=(rate,rate),name=prefix)(x)

    else:
        kernel_size_effective=kernel_size+(kernel_size-1)*(rate-1)
        pad_total=kernel_size_effective-1
        pad_beg=pad_total//2
        pad_end=pad_total-pad_beg
        x=ZeroPadding2D((pad_beg,pad_end))(x)
        return Conv2D(filters,(kernel_size,kernel_size),strides=(stride,stride),padding='valid',use_bias=False,dilation_rate=(rate,rate),name=prefix)(x)

def _xception_block(inputs,depth_list,prefix,skip_connection_type,stride,rate=1,depth_activation=False,return_skip=False):
    residual= inputs
    for i in range(3):
        residual = SepConv_BN(residual,depth_list[i],prefix + '_separable_conv{}'.format(i+1), stride=stride if i==2 else 1, rate = rate, depth_activation=depth_activation)

        if i==1:
            skip= residual
    if skip_connection_type=='conv':
        shortcut= _conv2d_same(inputs, depth_list[-1],prefix+'_shortcut',kernel_size=1,stride=stride)

        shortcut= BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs= layers.add ([residual,shortcut])
    elif skip_connection_type=='sum':
        outputs=layers.add([residual,inputs])

    elif skip_connection_type=='none':
        outputs=residual
    if return_skip:
        return outputs,skip
    else:
        return outputs

def relu6(x):
    return K.relu(x,max_value=6)

def _make_divisible(v, divisor, min_value=None):  # Used in filter making
    if min_value is None:
        min_value=divisor
    new_v = max(min_value,int (v+divisor/2)//divisor*divisor)
    if new_v<0.9*v:
        new_v+= divisor
    return new_v

'''MULTINET_______
def _inverted_res_block(inputs):
    in_channels= inputs._keras_shape[-1]
    pointwise_conv_filters=int(filters*alpha) #Alpha?
    pointwise_filters=_make_divisible(pointwise_conv_filters,8)
    x=inputs
    prefix='expanded_conv_{}_'.format(block_id)
    if block_id:'''

def DeepLabv3(weights='pascal_voc',input_tensor=None,input_shape=(512,512,3),classes=21,backbone='xception',OS=16,alpha=1):
    if not (weights in {'pascal_voc',None}):
        raise ValueError('the weights argument should be either' 'None(random initialization or pascal_voc''(pre-trained on PASCAL VOC))')
    if not ( backbone in {'xception','mobilenetv2'}):
        raise ValueError('The DeeplabV3+ model is only available with'
        'the Tensorflow backend')
    if K.backend()!='tensorflow':
        raise RuntimeError('The DeeplabV3+ model is only available with Tensorflow backend.')
    if input_tensor is None:
        img_input=Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input= Input(tensor=input_tensor,shape=input_shape)
        else:
            img_input=input_tensor
    if backbone=='xception':
        if OS==8:
            entry_block3_stride=1#???????
            middle_block_rate=2
            exit_block_rates=(2,4)
            atrous_rates=(12,24,36)
        else:
            entry_block3_stride=2
            middle_block_rate=1
            exit_block_rate=(1,2)
            atrous_rates=(6,12,18)

        x=Conv2D(32,(3,3),strides=(2,2),name='entry_flow_conv1_1',use_bias=False,padding='same')(img_input)
        x=BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x=Activation('relu')(x)
        x=_conv2d_same(x,64,'entry_flow_conv1_2',kernel_size=3,stride=1)
        x=BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x=Activation('relu')(x)

        x=_xception_block(x,[128,128,128],'entry_flow_block1',skip_connection_type='conv',stride=2,depth_activation=False)
        x,skip1=_xception_block(x,[256,256,256],'entry_flow_block2',skip_connection_type='conv',stride=2,depth_activation=False,return_skip=True)
        x=_xception_block(x,[728,728,728],'entry_flow_block3',skip_connection_type='conv',stride=entry_block3_stride,depth_activation=False)
        for i in range(16):
            x=_xception_block(x,[728,728,728],'middle_flow_unit_{}'.format(i+1),skip_connection_type='sum',stride=1,rate=middle_block_rate,depth_activation=False)
        x=_xception_block(x,[728,1024,1024],'exit_flow_block1',skip_connection_type='conv',stride=1,rate=exit_block_rate[0],depth_activation=False)
        x=_xception_block(x,[1536,1536,2048],'exit_flow_block2',skip_connection_type='none',stride=1,rate=exit_block_rate[1],depth_activation=True)
#End of feature extractor for Xception

    b4= AveragePooling2D(pool_size=(int(np.ceil(input_shape[0]/OS)),int(np.ceil(input_shape[1]/OS))))(x)
    b4=Conv2D(256,(1,1),padding='same',use_bias=False, name='image_pooling')(b4)
    b4=BatchNormalization(name='image_pooling_BN',epsilon=1e-5)(b4)
    b4=Activation('relu')(b4)
    b4= BilinearUpsampling((int(np.ceil(input_shape[0]/OS)),int(np.ceil(input_shape[1]/OS))))(b4)

 # simple 1x1
    b0=Conv2D(256,(1,1),padding='same',use_bias=False,name='aspp0')(x)
    b0=BatchNormalization(name='aspp0_BN',epsilon=1e-5)(b0)
    b0=Activation('relu',name='aspp0_activation')(b0)

    if backbone=='xception':
        # rate = 6
        b1= SepConv_BN(x,256,'aspp1',rate=atrous_rates[0],depth_activation=True,epsilon=1e-5)

        # rate=12
        b2= SepConv_BN(x,256,'aspp2',rate=atrous_rates[1],depth_activation=True,epsilon=1e-5)          # why depth activation=True

        # rate= 18
        b3 =SepConv_BN(x,256,'aspp3',rate=atrous_rates[2],depth_activation=True,epsilon= 1e-5)

        x= Concatenate()([b4,b0,b1,b2,b3])

    else:
        x= Concatenate()([b4,b0])

    x = Conv2D(256,(1,1),padding='same',use_bias=False,name='concat_projection')(x)
    x= Activation('relu')(x)
    x= Dropout(0.1)(x)

    # Deeplab v3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        x= BilinearUpsampling(output_size=(int(np.ceil(input_shape[0]/4)),int(np.ceil(input_shape[1]/4))))(x)
        dec_skip1= Conv2D(48,(1,1),padding='same',use_bias=False,name='feature_projection0')(skip1)

        x= Concatenate()([x,dec_skip1])
        x= SepConv_BN(x,256,'decoder_conv0',depth_activation=True,epsilon=1e-5)

        x= SepConv_BN(x,256,'decoder_conv1',depth_activation=True,epsilon=1e-5)
    if classes== 21:
        last_layer_name='logits_semantic'

    else:
        last_layer_name= 'custom_logits_semantic'

    x= Conv2D(classes, (1,1), padding= 'same',name=last_layer_name)(x)
    x= BilinearUpsampling(output_size=(input_shape[0],input_shape[1]))(x)

    ''' ensure that the model takes into account any potenial predecessors of input_tensor.'''

    if input_tensor is not None:
        inputs=get_source_inputs(input_tensor)

    else:
        inputs= img_input

    model = Model(inputs,x,name='deeplabv3plus')


    if weights=='pascal_voc':
        if backbone== 'xception':
            weight_path= get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',WEIGHTS_PATH_X,cache_subdir='models')


        model.load_weights(weight_path, by_name = True)

    return model
def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='tf')
