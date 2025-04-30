'''
Data loader for the Chesapeake Bay Watershed "patches" data set

Author: Andrew H. Fagg  2022-04-26
  Original

Andrew H. Fagg 2024-03
  Update to support diffusion examples
        

'''
import numpy as np

import tensorflow as tf


def load_single_file(fname:str):
    '''
    Load one patches file of the Chesapeake Watershed data set and return the imagery and pixel labels
    
    For the returned imagery, the features are scaled to the 0...1 range
    
    Pixel labels are 8-bit integers representing the following classes:
    0: No class (should generally not be seen in this data set, if at all)
    1: water
    2: tree canopy / forest
    3: low vegetation / field
    4: barren land
    5: impervious (other)
    6: impervious (road) 
    
    :param fname: Absolute file name
    
    :return: Tuple of image (256x256x24) and labels (256x256).  The outputs are proper TF Tensors
    '''
    # Load data
    fname = fname.numpy().decode('utf-8')
    
    dat = np.load(fname)
    
    # Extract the example and place it into standard TF format
    dat = dat['arr_0']
    dat = np.transpose(dat, [0, 2, 3, 1])
    
    # Labels: 8th element
    outs = dat[0, :, :, 8]
    outs = outs.astype(np.int8)
    
    # 15 = no data case; set these to weight 0; all others are weight 1.0
    #weights = np.logical_not(np.equal(outs, 15)) * 1.0
    
    # Set all class 15 to class 0
    outs[outs == 15] = 0
    outs_np = outs
    
    # Image data
    images = dat[0, :, :, 0:8]/255.0
    
    # Landsat data
    # Unclear what the max is over the data set, but at least this gets us into the ballpark
    landsat = dat[0, :, :, 10:28]/4000.0
    
    # Full set of input channels
    ins_np = np.concatenate([images, landsat], axis=2)
    
    # Some basic checks
    assert not np.isnan(ins_np).any(), "File contains NaNs (%s)"%(fname)
    assert np.min(outs_np) >= 0, "Labels out of bounds (%s, %d)"%(fname, np.min(outs_np))
    assert np.max(outs_np) < 7, "Labels out of bounds (%s, %d)"%(fname, np.max(outs_np))
    
    # Translate from numpy to TF Tensors
    ins = tf.convert_to_tensor(ins_np, tf.float32)

    outs = tf.convert_to_tensor(outs, tf.int8)

    return ins, outs

def load_single_file_wrapper(fname:str, image_size:[int]=(256, 256, 26))->[tf.Tensor]:
    '''
    :param fname: Absolute file name for the patch file
    :image_size: The expected size of the patch (needs to match the true size
            of (256, 256, 26) right now)
    :return: Input-output tuple

    This call structure is necessary:
    - The py_function wrapper allows these operations to be mapped to the GPU, but
       it does not properly preserve the shape of the returned tensors (which do
       have a defined shape at time off loading).
    - This function then adds the shape information back into the TF Tensor
    
    '''
    ins, outs = tf.py_function(
        func=load_single_file,
        inp=[fname],
        Tout=(tf.float32, tf.int8)
    )

    # Add proper shape information to the TF Tensor
    ins = tf.ensure_shape(ins, image_size)
    outs = tf.ensure_shape(outs, image_size[0:2])

    return ins, outs

@tf.autograph.experimental.do_not_convert
def load_single_image_class_pair(fname:str, patch_size:int = 32) -> [tf.Tensor]:
    '''
    NOTE: DO NOT USE

    Load one patches file of the Chesapeake Watershed data set and return the imagery and pixel labels
    
    For the returned imagery, the features are scaled to the 0...1 range
    
    Pixel labels are 8-bit integers representing the following classes:
    0: No class (should generally not be seen in this data set, if at all)
    1: water
    2: tree canopy / forest
    3: low vegetation / field
    4: barren land
    5: impervious (other)
    6: impervious (road) 
    
    :param fname: Absolute file name
    
    :return: Tuple of image (256x256x24) and labels (256x256).  The outputs are proper TF Tensors
    '''
    nclasses = 7
    
    # Load data
    fname = fname.numpy().decode('utf-8')
    
    dat = np.load(fname)
    
    # Extract the example and place it into standard TF format
    dat = dat['arr_0']
    dat = np.transpose(dat, [0, 2, 3, 1])
    
    # Labels: 8th element
    outs = dat[0, :, :, 8]
    outs = np.int_(outs)
    
    # Set all class 15 to class 0
    np.equal(outs, 15, where=0)

    # Cleck class number
    assert np.min(outs) >= 0, "Labels out of bounds (%s, %d)"%(fname, np.min(outs))
    assert np.max(outs) < nclasses, "Labels out of bounds (%s, %d)"%(fname, np.max(outs))

    # One-hot encode class image
    outs = tf.one_hot(outs, nclasses).numpy()
    # Cut down to patch_size
    #print('PATCH', patch_size)
    #outs = tf.image.crop_to_bounding_box(outs,
    #tf.constant(0, tf.dtypes.int32), tf.constant(0, tf.dtypes.int32),
    #tf.constant(patch_size, tf.dtypes.int32),
    #tf.constant(patch_size, tf.dtypes.int32))
    outs = outs[:patch_size, :patch_size]
    
    # Image data
    image = dat[0, :, :, 0:3]/255.0
    #image = tf.image.crop_to_bounding_box(image, #0, 0, patch_size, patch_size)
    #tf.constant(0, tf.dtypes.int32), tf.constant(0, tf.dtypes.int32),
    #tf.constant(patch_size, tf.dtypes.int32),
    #tf.constant(patch_size, tf.dtypes.int32))
    image = image[:patch_size, :patch_size, :]

    # Some basic checks
    assert not np.isnan(image).any(), "File contains NaNs (%s)"%(fname)
    
    # Translate from numpy to TF Tensors
    return tf.cast(image, tf.float32), tf.cast(outs, tf.float32)


### @tf.autograph.experimental.do_not_convert
def create_single_dataset(base_dir:str='/home/fagg/datasets/radiant_earth/pa',
                          full_sat:bool=True,
                          patch_size:[int]=None,
                          partition:str='train',
                          fold:int=0,
                          filt:str='*',
                          cache_path:str=None,
                          repeat:bool=False,
                          shuffle:bool=None,
                          batch_size:int=8,
                          prefetch:int=2,
                          num_parallel_calls:int=4):
    '''
    Files are located in <base_dir>/<partition>/F<fold>/
    
    Create a TF Dataset that can be used for training and evaluating a model
    :param base_dir: Location of the dataset partitions 
    :param partition: Partition subdirectory of base_dir that contains the folds to be loaded (possibilities are "train" and "valid")
    :param fold: Fold to load (0 ... 9)
    :param filt: Regular expression filter for the files within the fold directory. 
                    '*' means use all files
                    '*0' means use all files ending in zero
                    '*[012]' means use all files ending zero, one or two
    :param cache_path: None -> no cache; '' -> cache to RAM; path (including file name) -> cache to file
    :param repeat: repeat data set indefinitely (default = False)
    :param shuffle: shuffle buffer size (default = None -> dont shuffle) 
    :param batch_size: Size of the batches to be produced by this data set
    :param prefetch: Number of batches to prefetch in parallel with training
    :param num_parallel_calls: Number of threads to use for the data loading process
    
    :return: TF Dataset that emits tuples (ins, outs)
                ins is a TF Tensor of shape batch_size x 256 x 256 x 24
                outs is a TF Tensor of shape batch_size x 256 x 256
    '''
    if num_parallel_calls == -1:
        num_parallel_calls = tf.data.AUTOTUNE
    
    
    # Full list of files in the dataset
    data = tf.data.Dataset.list_files('%s/%s/F%d/%s.npz'%(base_dir, partition, fold, filt), shuffle=False)
    
    # Load each file
    #  - py_function allows eager execution
    #  - we must declare here the return types of the Dataset
    if full_sat:
        # We are using this for HW 4
        data = data.map(load_single_file_wrapper, num_parallel_calls=num_parallel_calls)
    else:
        data = data.map(lambda x: tf.py_function(func=load_single_image_class_pair,
                                                 inp=[x, patch_size], Tout=(tf.float32, tf.float32)), 
                        num_parallel_calls=num_parallel_calls)

    # Caching
    if cache_path is not None:
        if cache_path == '':
            # Cache to RAM
            data = data.cache()
        else:
            # Cache to file
            data = data.cache(cache_path)
            
    # Repeat the sequence if it runs out (use if steps_per_epoch is not None)
    if repeat:
        data = data.repeat()

    # Shuffle
    if shuffle is not None:
        data = data.shuffle(shuffle)
        
    # Batch the individual elements
    if batch_size is not None:
        data = data.batch(batch_size)
    
    # Buffer multiple batches
    if prefetch is not None:
        data = data.prefetch(prefetch)
    
    return data

@tf.autograph.experimental.do_not_convert
def create_datasets(base_dir:str='/home/fagg/datasets/radiant_earth/pa',
                    full_sat:bool=True,
                    patch_size:int=None,
                    fold:int=0,
                    train_filt:str='*[012345678]',
                    cache_dir:str=None,
                    repeat_train:bool=False,
                    shuffle_train:int=None,
                    batch_size:int=8,
                    prefetch:int=2,
                    num_parallel_calls:int=4):
    '''
    Files are located in <base_dir>/<partition>/F<fold>/
    
    Create TF Datasets for training, validation and testing
    
    :param base_dir: Location of the dataset partitions 
    :param fold: Fold to load (0 ... 9)
    :param train_filt: Regular expression filter for the files within the fold directory. 
                    '*' means use all files
                    '*0' means use all files ending in zero
                    '*[012]' means use all files ending zero, one or two
                    You should not need to change from default unless you want to use a smaller training set
    :param cache_dir: None -> no cache; directory path -> cache to 3 files (on the supercomputer, this is LSCRATCH)
    :param repeat_train: repeat data set indefinitely (default = False) (validation and test sets are not repeated)
    :param shuffle_train: shuffle buffer size (default = None -> dont shuffle) (validation and test sets are not shuffled)
    :param batch_size: Size of the batches to be produced by this data set 
    :param prefetch: Number of batches to prefetch in parallel with training 
    :param num_parallel_calls: Number of threads to use for the data loading process (-1 = use AutoTune)
    
    :return: TF Datasets for training, validation and testing; and the number of classes
                ins is a TF Tensor of shape batch_size x 256 x 256 x 3
                outs is a TF Tensor of shape batch_size x 256 x 256 x 7
    '''
    ds_train = create_single_dataset(base_dir=base_dir,
                                     full_sat=full_sat,
                                     patch_size=patch_size,
                                     partition='train',
                                     fold=fold,
                                     filt=train_filt,
                                     cache_path = None if cache_dir is None else '' if cache_dir=='' else '%s/train_f_%d_'%(cache_dir, fold),
                                     repeat=repeat_train,
                                     shuffle=shuffle_train,
                                     batch_size=batch_size,
                                     prefetch=prefetch,
                                     num_parallel_calls=num_parallel_calls)
    
    ds_valid = create_single_dataset(base_dir=base_dir,
                                     full_sat=full_sat,
                                     patch_size=patch_size,
                                     partition='train',
                                     fold=fold,
                                     filt='*9',
                                     cache_path = None if cache_dir is None else '' if cache_dir=='' else '%s/validation_f_%d_'%(cache_dir, fold),
                                     repeat=False,
                                     shuffle=None,
                                     batch_size=batch_size,
                                     prefetch=prefetch,
                                     num_parallel_calls=num_parallel_calls)

    ds_test = create_single_dataset(base_dir=base_dir,
                                     full_sat=full_sat,
                                     patch_size=patch_size,
                                     partition='valid',
                                     fold=fold,
                                     filt='*',
                                     cache_path = None if cache_dir is None else '' if cache_dir=='' else '%s/test_f_%d_'%(cache_dir, fold),
                                     repeat=False,
                                     shuffle=None,
                                     batch_size=batch_size,
                                     prefetch=prefetch,
                                     num_parallel_calls=num_parallel_calls)

    return ds_train, ds_valid, ds_test, 7


@tf.autograph.experimental.do_not_convert
def create_diffusion_example(I:tf.float32, L:tf.float32, patch_size:int, alpha:tf.Tensor, t:tf.Tensor)->[tf.Tensor]:
    #nalpha:tf.Tensor, generator:tf.random.Generator)
    '''
    Given an input image and a label image, produce a single example for training a diffusion network

    :param I: Input image
    :param L: Label image
    :param patch_size: size of each image on each side
    :param alpha: TF tensor of alpha blending values
    :param t: Time step to generate the solution for

    :return: Label image, Time image, blended image, and the raw noise.  All are patch_size x patch_size x ?
    '''
    # Sample noise
    noise = tf.random.normal(shape=(patch_size, patch_size, 3), mean=0, stddev=1.0, dtype=tf.dtypes.float32)

    # AHF: this tensorflow method seems to have a bug that will result in generating the wrong shape from
    #   from time to time.  t is now handed into this method
    # Sample time step: int
    # t = tf.random.uniform(shape=(1,), minval=0, maxval=nalpha, dtype=tf.dtypes.int32)
    # assert alpha.shape == (50,) and t.shape == (1,), "SHAPES DON'T MATCH %s %s"%(str(alpha.shape), str(t.shape))

    # Pull out the alpha that corresponds to the chosen time
    a = tf.gather_nd(alpha, t)
    
    # Scale pixels to +/- 1
    I = tf.multiply(I, 2.0) - 1.0
    
    # Output image is a blend of the original image and the noise
    image = tf.multiply(I, tf.math.sqrt(a)) + tf.multiply(noise, tf.math.sqrt(1-a))

    # Strange to now be returning t again here - but necessary for the mapping process
    return tf.cast(L, tf.dtypes.float32), tf.cast(t, tf.dtypes.int32), \
        tf.cast(image, tf.dtypes.float32), tf.cast(noise, tf.dtypes.float32)

    
@tf.autograph.experimental.do_not_convert
def create_diffusion_dataset(alpha,
                             base_dir='/home/fagg/datasets/radiant_earth/pa', 
                             patch_size=None,
                             fold=0,
                             filt='*[012345678]',
                             cache_dir=None,
                             repeat=False,
                             shuffle=None,
                             repeat_validation=False,
                             shuffle_validation=None,
                             batch_size=8,
                             prefetch=2,
                             num_parallel_calls=4):

    '''
    DO NOTE USE

    Create TF Datasets for training a diffusion model

    :param alpha: numpy array of alpha blending values.  This determines the number of time steps
    :param base_dir: Location of the dataset partitions 
    :param patch_size: the size of each image (on a side)
    :param fold: Fold to load (0 ... 9)
    :param filt: Regular expression filter for the files within the fold directory for the training set. 
                    '*' means use all files
                    '*0' means use all files ending in zero
                    '*[012]' means use all files ending zero, one or two
                    You should not need to change from default unless you want to use a smaller training set
    :param cache_dir: None -> no cache; empty string -> cache to RAM; 
                      directory path -> cache to 3 files (on the supercomputer, this is LSCRATCH)
    :param repeat: repeat data set indefinitely (default = False)
    :param shuffle: shuffle buffer size (default = None -> dont shuffle)
    :param repeat_validation: repeat data set indefinitely (default = False) 
    :param shuffle_shuffle: shuffle buffer size (default = None -> dont shuffle) 
    :param batch_size: Size of the batches to be produced by this data set 
    :param prefetch: Number of batches to prefetch in parallel with training 
    :param num_parallel_calls: Number of threads to use for the data loading process (-1 = use AutoTune)
    
    :return: ds_training, ds_validation: TF Datasets for training formatted for using with model.fit()
        2-Tuple of:
          ins: Dictionary with keys:
                'label_input': label image
                'time_input': time image
                'image_input': corrupted real image

                NOTE: the names of these keys must match the name of the Inputs() in your training model
          outs: noise image
 
    '''
    
    # Base dataset: tuples of individual I/L pairs
    ds = create_single_dataset(base_dir=base_dir,
                                     full_sat=False,
                                     patch_size=patch_size,
                                     partition='train',
                                     fold=fold,
                                     filt=filt,
                                     cache_path=cache_dir,
                                     repeat=repeat,
                                     shuffle=shuffle,
                                     batch_size=None,
                                     prefetch=None,
                                     num_parallel_calls=num_parallel_calls)

    # Convert alpha and nalpha to TF constants
    nalpha = tf.constant(alpha.shape[0], dtype=tf.dtypes.int32)
    alpha_tf = tf.constant(alpha, dtype=tf.dtypes.float32)

    # Random number generator for time indices
    generator = tf.random.Generator.from_seed(42)
    
    # Create DS for Diffusion: this is a 4-tuple
    ds = ds.map(lambda I, L: tf.py_function(func=create_diffusion_example, inp=[I, L, patch_size, alpha_tf,
                                                                                generator.uniform(shape=(1,),
                                                                                                  minval=0,
                                                                                                  maxval=nalpha,
                                                                                                  dtype=tf.dtypes.int32)
                                                                            ], 
                                            Tout=(tf.float32, tf.int32, tf.float32, tf.float32)), 
                num_parallel_calls=num_parallel_calls)
        
    # Batch the individual elements
    if batch_size is not None:
        ds = ds.batch(batch_size)

    # Format for input into a Keras Model
    ds = ds.map(lambda L, T, image, noise: ({'label_input': L, 'time_input': T, 'image_input': image}, noise))
    
    # Buffer multiple batches
    if prefetch is not None:
        ds = ds.prefetch(prefetch)

    ############################ 
    # Validation data set

    # Base dataset: tuples of individual I/L pairs
    ds_valid = create_single_dataset(base_dir=base_dir,
                                     full_sat=False,
                                     patch_size=patch_size,
                                     partition='train',
                                     fold=fold,
                                     filt='*9',
                                     cache_path=cache_dir,
                                     repeat=repeat_validation,
                                     shuffle=shuffle_validation,
                                     batch_size=None,
                                     prefetch=None,
                                     num_parallel_calls=num_parallel_calls)
    
    # Create ds_valid for Diffusion: this is a 4-tuple
    ds_valid = ds_valid.map(lambda I, L: tf.py_function(func=create_diffusion_example, inp=[I, L, patch_size, alpha_tf,
                                                                                            generator.uniform(shape=(1,),
                                                                                                              minval=0,
                                                                                                              maxval=nalpha,
                                                                                                              dtype=tf.dtypes.int32)],
                                                        Tout=(tf.float32, tf.int32, tf.float32, tf.float32)), 
                            num_parallel_calls=num_parallel_calls)
        
    # Batch the individual elements
    if batch_size is not None:
        ds_valid = ds_valid.batch(batch_size)

    # Format for input into a Keras Model
    ds_valid = ds_valid.map(lambda L, T, image, noise: ({'label_input': L, 'time_input': T, 'image_input': image}, noise))
    
    # Buffer multiple batches
    if prefetch is not None:
        ds_valid = ds_valid.prefetch(prefetch)

    #######
    return ds, ds_valid
