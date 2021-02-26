import numpy as np
import tensorflow as tf

#data zipped at https://drive.google.com/file/d/1QCA_o0QU0hlJtpPwe3pEc7_7F8HuvCGU/view?usp=sharing

def make_label_change_array(num_classes):
  """Arg:
  num_classes:int, number of classes
  returns:
  num_classes^2 matrix of categorical change labels.
  """
  label_change_arr = np.zeros((num_classes,num_classes),dtype=np.uint8)
  i = 0
  for x in range(num_classes):
    for y in range(num_classes):
      label_change_arr[x,y] = i *(x!=y)
      i += 1
  return label_change_arr

label_change_arr = make_label_change_array(6)

print(label_change_arr)

def get_class_change_label(before,after,label_change_arr):
  """
  Input:
  two np.array or tf.tensor of shape (batch,Width,Height, num_classes) corresponding to class labeled data in one hot encoding
  the arrays represent the before and after class labels for change detection
  Output:
  a np.array of shape (batch,Width,Height) corresponding to change labels (see figure 2)
  """
  labels_combined = np.einsum("abd,abe->abde",before,after)
  labels_combined = labels_combined*label_change_arr
  labels_combined = np.sum(labels_combined, axis=(-1,-2))
  return labels_combined

def get_item(path):

  #wrap function in tf.numpy_function() or tf.data.dataset.map bugs out

  def _get_item(path):
    """
    args:
    path: Dataset.list_file dataset element

    returns: 
    (h,w,3),(h,w,1),(h,w,3),(h,w,1) image-label 4-tuple in tf.float32 tf.Tensor
    """
    oh_label = False
    fn = tf.strings.split(path,"/")
    base_dir = tf.strings.join(fn[:-2],separator="/")

    image1_fn = (base_dir+"/im1/"+fn[-1]).numpy()
    image2_fn = (base_dir+"/im2/"+fn[-1]).numpy()

    label1_fn = (base_dir+"/label1/"+fn[-1]).numpy()
    label2_fn = (base_dir+"/label2/"+fn[-1]).numpy()

    #label1_fn = "/content/Kaggle_CoverChange/label1/00018.png"
    #label2_fn = "/content/Kaggle_CoverChange/label2/00018.png"


    image1 = tf.keras.preprocessing.image.load_img(image1_fn)
    image1 = tf.keras.preprocessing.image.img_to_array(image1)

    image2 = tf.keras.preprocessing.image.load_img(image2_fn)
    image2 = tf.keras.preprocessing.image.img_to_array(image2)

    label1 = tf.keras.preprocessing.image.load_img(label1_fn, color_mode="grayscale")
    label1 = tf.keras.preprocessing.image.img_to_array(label1)
    label1 = to_categorical(label1,class_dict)
    #label1 = tf.expand_dims(label1,axis=0)

    label1 = tf.one_hot(tf.cast(label1[:,:,0],dtype = tf.uint8),depth = num_classes)




    label2 = tf.keras.preprocessing.image.load_img(label2_fn, color_mode ="grayscale")
    label2 = tf.keras.preprocessing.image.img_to_array(label2)
    label2 = to_categorical(label2,class_dict)
    #label2 = tf.expand_dims(label2,axis=0)

    label2 = tf.one_hot(tf.cast(label2[:,:,0],dtype = tf.uint8),depth=num_classes)

    change_label = get_class_change_label(label1,label2,label_change_arr=label_change_arr)
    
    if oh_label:
      change_label = tf.one_hot(tf.cast(change_label[:,:],dtype = tf.uint8),depth=num_classes**2-1)
    else:
      change_label = tf.expand_dims(change_label, axis=-1)



    #classes = array([ 29.,  38.,  75.,  76., 128., 150., 255.])
    #(h,w,1) tensor of int from 1-6

    return image1,image2,change_label

  output = tf.numpy_function(_get_item,[path],[tf.float32,tf.float32,tf.float32])

  return output

def transform(x1,x2,y):
  """
  Write your data transformations here
  """
  x1=tf.keras.layers.experimental.preprocessing.Rescaling(1./255.0)(x1)
  x2=tf.keras.layers.experimental.preprocessing.Rescaling(1./255.0)(x2)
  #y=tf.one_hot(tf.cast(y[:,:,0],dtype = tf.int32),depth = 7)

  return x1,x2,y
def preprocessing(list_ds,batch_size=8,augmentation=transform):
  """
  applies some preprocessing
  args:
  list_ds-Dataset.list_files dataset object
  batch_size-int 
  augmentation-a function (x,y)->(x,y)
  returns-batched dataset
  """
  #ds = list_ds.cache() 
  ds = list_ds.map(get_item,num_parallel_calls=tf.data.AUTOTUNE) 
  ds = ds.batch(batch_size,drop_remainder=True)
  ds = ds.shuffle(100)

  if augmentation:
    ds = ds.map((lambda x,y,z : augmentation(x,y,z)),num_parallel_calls=tf.data.AUTOTUNE)

  
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds
