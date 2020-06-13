# -*- coding: utf-8 -*-
"""HelpersDCGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vu6S9ASTlw49chSs67-z61uysx8Xh4sF
"""

import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Flatten, Input, Dense
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import Mean
from IPython.core.display import display as jupy_display
import numpy as np
import os
import time 


def DCGenerator():
    input_noise = Input(shape=(1, 1, 100))
    #1st conv layer
    x = Conv2DTranspose(1024, kernel_size=4, strides=1, padding="valid")(input_noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    #2nd conv layer
    x = Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    #3rd 
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    #4rth
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    #output image 
    x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)
    
    return Model(inputs=input_noise, outputs=x)


def DCDiscriminator():
    input_image = Input(shape=(64, 64, 1))
    
    x = Conv2D(128, kernel_size=4, strides=2, padding="same")(input_image)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1024, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
    x = Flatten()(x)
    # sigmoid directly in the loss function : binary cross entropy

    return Model(inputs=input_image, outputs=x)


def get_noise(batch_size, nz=100):
    return tf.random.normal([batch_size, 1, 1, nz])

      
def plot_generated_images(images, epoch = 0, dim=(5,5), figsize=(7,7), save = True):
        # by default images must contain the right number of examples 
        plt.figure(figsize=figsize)
        for i in range(images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(images[i, :, :, 0] * 127.5 + 127.5 , cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save : 
            plt.savefig('gan_generated_image %d.png' %epoch)


####################### 1.3. 

class DCGAN(object):
    def __init__(self, generator, discriminator, img_rows=64, img_cols=64, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.generator = generator   # discriminator
        self.discriminator = discriminator   # generator
        #self.AM = None  # adversarial model
        #self.DM = None  # discriminator model
        self.optimizer_d = Adam(learning_rate=1e-4, beta_1=0.5)
        self.optimizer_g = Adam(learning_rate=1e-4, beta_1=0.5)
        
        
        
    @staticmethod
    def get_noise(batch_size, nz=100):
        return tf.random.normal([batch_size, 1, 1, nz])
    @staticmethod
    def plot_generated_images(images, epoch = 0, dim=(5,5), figsize=(7,7), save = True):
        # by default images must contain the right number of examples 
        plt.figure(figsize=figsize)
        for i in range(images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(images[i, :, :, 0] * 127.5 + 127.5 , cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save : 
            plt.savefig('gan_generated_image %d.png' %epoch)

    @staticmethod
    def discriminator_loss( preds_real, preds_fake):
        loss_real = binary_crossentropy(tf.ones_like(preds_real), preds_real, from_logits=True)
        loss_fake = binary_crossentropy(tf.zeros_like(preds_fake), preds_fake, from_logits=True)
        return loss_real + loss_fake

    @staticmethod
    def generator_loss( preds_fake):
        return binary_crossentropy(tf.ones_like(preds_fake), preds_fake, from_logits=True)
    
    @tf.function
    def train_step(self, images) : 
        noise = get_noise(images.shape[0])
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape : 
            generated_images = self.generator(noise, training=True)
            preds_real_im = self.discriminator(images, training=True)
            preds_fake_im = self.discriminator(generated_images, training = True)

            #g_loss = binary_crossentropy(tf.ones_like(preds_fake), preds_fake, from_logits=True)

            #loss_real = binary_crossentropy(tf.ones_like(preds_real), preds_real, from_logits=True)
            #loss_fake = binary_crossentropy(tf.zeros_like(preds_fake), preds_fake, from_logits=True)
            #d_loss = loss_real+loss_fake

            g_loss = self.generator_loss(preds_fake_im)
            d_loss = self.discriminator_loss(preds_real_im, preds_fake_im)
            
        #apply gradient descent
        gradients_generator = g_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_discriminator = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        self.optimizer_d.apply_gradients(zip(gradients_discriminator, self.discriminator.trainable_variables))
        
        return d_loss, g_loss
 
    
    
    def train(self, data_generator, epochs=10, checkpoint_dir = './training_checkpoints'):
        # create checkpoint to save the training progression 
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.optimizer_g,
                                     discriminator_optimizer=self.optimizer_d,
                                     generator=self.generator,
                                     discriminator=self.discriminator)
    
        fixed_noise = get_noise(25)

        #print("Base noise:")
        #fake_images = generator(fixed_noise, training=False).numpy()
        #jupy_display(display(fake_images))
        
        # loop over epochs : 
        for epoch in range(epochs):
            start = time.time()
            print("====== Epoch {:2d} ======".format(epoch))
            #initiaite the mean loss over the epoch for the discriminator and generator
            epoch_loss_d = Mean()
            epoch_loss_g = Mean()

            #epoch_len = tf.data.experimental.cardinality(data_generator)
            for i, real_images in enumerate(data_generator):
                loss_d, loss_g = self.train_step(real_images)
                epoch_loss_d(loss_d)
                epoch_loss_g(loss_g)
        
            print("\nDiscriminator: {}, Generator: {}".format(
                epoch_loss_d.result(), epoch_loss_g.result()))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                fake_images = self.generator(fixed_noise, training=False)
                self.plot_generated_images(fake_images, epoch, save = True)