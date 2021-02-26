from DDN import *
import tensorflow as tf
import numpy as np
from layers import *
from pipeline import *

data_dir = "/content/Kaggle_CoverChange"
list_ds = tf.data.Dataset.list_files(str(data_dir+"/im1/*"),shuffle=False)
train_ds = preprocessing(list_ds, batch_size = batch_size)
ddn = DDN(512,6)
save_path = "/content/Saved_Models/"
if not(os.path.isdir(save_path)):
  os.mkdir(save_path)
batch_size = 1
epochs = 2

model_savename="DDN_test"

for epoch in range(epochs):

  train_loss.reset_states()
  train_accuracy.reset_states()

  print("Start of epoch {}".format(epoch))

  for step, (im1,im2,label) in enumerate(train_ds):

    step = step + epoch*steps_per_epoch

    with tf.GradientTape() as tape:

      y_pred = ddn([im1,im2],training=True)

      loss_value = DS_loss_fn(label,y_pred)
      
    acc = train_accuracy(label,y_pred[0]) #makes a metrics.CategoricalAccuracy object
    train_loss(loss_value) #puts the loss value into a metrics.Means object
    
    #with train_summary_writer.as_default():
    #  tf.summary.scalar('loss', train_loss.result(), step=step)
    #  tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
      

    grads = tape.gradient(loss_value,ddn.trainable_weights)
    optimizer.apply_gradients(zip(grads,ddn.trainable_weights))

    if step % 100== 0:
      print(
                "Training loss at step %d: %.4f"
                % (step, float(loss_value.numpy()))
            )
      
      print ("Training accuracy at step {} is {}".format(step, acc.numpy()))
      print("Seen so far: %s samples" % ((step + 1) * batch_size))
      print("Current LR: {}".format(optimizer.learning_rate(step)))
            
  ddn.save('saved_model/{}'.format(model_savename))
