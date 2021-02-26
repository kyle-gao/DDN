import tensorflow as tf
import numpy as np
from layers import *

def DDN (imsize, num_classes = 6):
    #if input image 512*512

    vgg16_512 = tf.keras.models.load_model("/content/Drive/Saved_Model")

    image1 = tf.layers.Input((imsize,imsize,3),name = "image1")
    image2 = layers.Input((imsize,imsize,3),name = "image2")

    feature_layers = [vgg16_512.get_layer("features_c1"),vgg16_512.get_layer("features_c2"),vgg16_512.get_layer('features_c3'),vgg16_512.get_layer('features_c4'),vgg16_512.get_layer('features_c5')]
    feature_extractor = tf.keras.models.Model(inputs = vgg16_512.input, outputs = [layer.output for layer in feature_layers],trainable=True)

    features_1 = feature_extractor(image1)
    features_2 = feature_extractor(image2)

    t1_b5c3 = features_1[4] #(None, 32, 32, 512)
    t2_b5c3 = features_2[4] 

    t1_b4c3 = features_1[3] #(None, 64, 64, 512)
    t2_b4c3 = features_2[3] 
    
    t1_b3c3 = features_1[2] #(None, 128, 128, 256)  
    t2_b3c3 = features_2[2]

    t1_b2c2 = features_1[1]
    t2_b2c2 = features_2[1] #(None, 256, 256, 128) 

    t1_b1c2 = features_1[0]
    t2_b1c2 = features_2[0] #(None, 512, 512, 64)


    """
    pair5 = layers.Input((imsize,imsize,64*2), name='pair5') #rbg images concatenated channel-wise
    pair4 = layers.Input((imsize//2,imsize//2,128*2), name='pair4')
    pair3 = layers.Input((imsize//4,imsize//4,256*2), name='pair3')
    pair2 = layers.Input((imsize//8,imsize//8,512*2), name='pair2')
    pair1 = layers.Input((imsize//16,imsize//16,512*2), name='pair1')


    t1_b5c3 = pair1[:,:,:,:3] #(None, 32, 32, 512)
    t2_b5c3 = pair1[:,:,:,3:] 

    t1_b4c3 = pair2[:,:,:,:3] #(None, 64, 64, 512)
    t2_b4c3 = pair2[:,:,:,3:] 
    
    t1_b3c3 = pair3[:,:,:,:3] #(None, 128, 128, 256)  
    t2_b3c3 = pair3[:,:,:,3:]

    t1_b2c2 = pair4[:,:,:,:3]
    t2_b2c2 = pair4[:,:,:,3:] #(None, 256, 256, 128) 

    t1_b1c2 = pair5[:,:,:,:3]
    t2_b1c2 = pair5[:,:,:,3:] #(None, 512, 512, 64)
    """

    concat_b5c3 = concatenate([t1_b5c3, t2_b5c3], axis=3) #channel 1024
    x = Conv2d_BN(concat_b5c3,256, 3)
    x = Conv2d_BN(x,128,3)
    attention_map_1 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_1])
    x = BatchNormalization(axis=3)(x)

    #branche1
    branch_1 =Conv2D(num_classes**2-1, kernel_size=1, padding='same',name='output_32')(x)

    x = Conv2DTranspose(imsize, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b4c3,t2_b4c3],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,256,3)
    x = Conv2d_BN(x,128,3)
    x = Conv2d_BN(x,64,3)
    attention_map_2 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_2])
    x = BatchNormalization(axis=3)(x)

    #branche2
    branch_2 =Conv2D(num_classes**2-1, kernel_size=1, padding='same',name='output_64')(x)

    x = Conv2DTranspose(256, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b3c3,t2_b3c3],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,256,3)
    x = Conv2d_BN(x,128,3)
    x = Conv2d_BN(x, 64, 3)
    attention_map_3 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_3])
    x = BatchNormalization(axis=3)(x)

    #branche3
    branch_3 =Conv2D(num_classes**2-1, kernel_size=1, padding='same',name='output_128')(x)

    x = Conv2DTranspose(128, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b2c2,t2_b2c2],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,128,3)
    x = Conv2d_BN(x,64,3)
    x = Conv2d_BN(x, 32, 3)
    attention_map_4 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_4])
    x = BatchNormalization(axis=3)(x)

    #branche4
    branch_4 =Conv2D(num_classes**2-1, kernel_size=1, padding='same',name='output_256')(x)

    x = Conv2DTranspose(64, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b1c2,t2_b1c2],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,64,3)
    x = Conv2d_BN(x,32,3)
    x = Conv2d_BN(x, 16, 3)
    attention_map_5 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_5])

    # branche5
    branch_5 =Conv2D(num_classes**2-1, kernel_size=1, padding='same',name='output_imsize')(x)

    DDN = Model(inputs=[image1,image2], outputs=[branch_5,branch_4,branch_3,branch_2,branch_1])

    return DDN
