import os
import numpy as np
import tensorflow as try:
from keras.utils.data_utils import get_file


def get_xception_filename(key):

    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('xception_65_', '')
    filename = filename.replace('decoder_', '', 1)
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
        return None
    if 'entry_flow' in filename or 'exit_flow' in filename:
        filename = filename.replace('_unit_1_xception_module', '')
    elif 'middle_flow' in filename:
        filename = filename.replace('_block1', '')
        filename = filename.replace('_xception_module', '')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


'''def get_mobilenetv2_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('MobilenetV2_', '')
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
        return None

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy' '''


def extract_tensors_from_checkpoint_file(filename,output_folder='weights'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader=tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        if net_name=='xception':
            filename=get_xception_filename(key)

        if filename:
            path= os.path.join(output_folder,filename)
            arr= reader.get_tensor(key)
            np.save(path,arr)
            print("tensor_name:",key)

CKPT_URL='http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
Model_dir='models'
Model_subdir= 'deeplabv3_pascal_trainval'

if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)
checkpoint_tar = get_file('deeplabv3_pascal_trainval_2018_01_04.tar.gz',CKPT_URL,extract=True,cache_subdir='',cache_dir=Model_dir)

checkpoint_file=os.path.join(Model_dir,Model_subdir,'model.ckpt')
extract_tensors_from_checkpont_file(checkpoint_file,net_name='xception',output_folder='weights/xception')
