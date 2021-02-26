import tensorflow as tf

entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM)#adds a softmax step

def dice_loss (y_true,y_pred,eps = 1e-5,num_classes=6):
  y_true = y_true[:,:,:,0]
  y_pred = tf.nn.softmax(y_pred)
  
  #y_pred = tf.argmax(y_pred,axis=-1)
  y_true = tf.one_hot(tf.cast(y_true,dtype = tf.uint8),depth = num_classes**2-1)
  return tf.math.reduce_mean(1 - (2*y_true*y_pred)/(y_true + y_pred + eps))

def DS_loss_fn(y_true, y_pred, loss_function = entropy, weights=[3,1,1,1,1]):
  loss = 0
  y_true==0
  for i in range(len(weights)):

    #loss = loss+ tf.math.reduce_mean(weights[i]*loss_function(y_true,y_pred[i]) + weights[i]*dice_loss (y_true, y_pred[i]))
    #loss = loss+ tf.math.reduce_mean(weights[i]*dice_loss (y_true, y_pred[i]))
    loss = loss+ weights[i]*loss_function(y_true,y_pred[i])
    y_true = tf.keras.layers.MaxPool2D()(y_true) 
  return loss
