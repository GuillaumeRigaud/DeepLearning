# -*- coding: utf-8 -*-


import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Flatten, Input, Dense
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers  import InstanceNormalization
from tensorflow.keras.activations import tanh
from tensorflow.keras.metrics import Mean
from IPython.core.display import display as jupy_display
import numpy as np
import os
import time 

import pandas as pd
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow_addons.layers  import InstanceNormalization
from matplotlib import pyplot
 


# define the discriminator model
def CycleDiscriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	return model
 
# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g
 
# define the standalone generator model
def CycleGenerator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(image_shape[2], (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model




def get_noise(batch_size, nz=100):
        return tf.random.normal([batch_size, 1, 1, nz])

      
def plot_generated_images(images, epoch = 0, dim=(5,5), figsize=(7,7), ch=1):
        # by default images must contain the right number of examples 
        plt.figure(figsize=figsize)
        for i in range(images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            if ch == 1 : 
                plt.imshow(images[i, :, :, 0] * 127.5 + 127.5 , cmap='gray')
            else : 
                plt.imshow(images[i, :, :, :] * 0.5+ 0.5 )
            plt.axis('off')
        plt.tight_layout()


####################### Cycle GAN 

class CycleGAN(object):
    def __init__(self, generator_g, generator_f, discriminator_x, discriminator_y, img_rows=128, img_cols=128, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.generator_g = generator_g  # generator
        self.generator_f = generator_f 
        self.discriminator_x = discriminator_x  # discriminator
        self.discriminator_y = discriminator_y   
                                                  
        self.LAMBDA = None 
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.training_losses =  pd.DataFrame( {"disc_y_loss" : [None], "gen_g_loss" : [None],
               "disc_x_loss" : [None], "gen_f_loss" : [None] })


    @staticmethod    
    def discriminator_loss(real, generated):
        real_loss = binary_crossentropy(tf.ones_like(real), real, from_logits=True)

        generated_loss = binary_crossentropy(tf.zeros_like(generated), generated, from_logits=True)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    @staticmethod
    def generator_loss(generated):
        return binary_crossentropy(tf.ones_like(generated), generated, from_logits=True)
    
    def calc_cycle_loss(self,real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1


    def identity_loss(self,real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    
    @staticmethod
    def save_generated_images(images, epoch=0, dim=(5,5), figsize=(7,7), ch=1, title="cyclegan_generated_image"):
        # by default images must contain the right number of examples 
        plt.ioff()
        plt.figure(figsize=figsize)
        for i in range(images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            if ch == 1 : 
                plt.imshow(images[i, :, :, 0] * 127.5 + 127.5 , cmap='gray')
            else : 
                plt.imshow(images[i, :, :, :] * 0.5 + 0.5 )
            plt.axis('off')
        plt.tight_layout()
        plt.savefig( title+ '%d.png' %epoch)
        plt.close()
        
    

    @tf.function
    def train_step(self,real_x, real_y, regularization=False):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
          # Generator G translates X -> Y
          # Generator F translates Y -> X.
        
          fake_y = self.generator_g(real_x, training=True)
          cycled_x = self.generator_f(fake_y, training=True)

          fake_x = self.generator_f(real_y, training=True)
          cycled_y = self.generator_g(fake_x, training=True)

          # same_x and same_y are used for identity loss.
          same_x = self.generator_f(real_x, training=True)
          same_y = self.generator_g(real_y, training=True)

          disc_real_x = self.discriminator_x(real_x, training=True)
          disc_real_y = self.discriminator_y(real_y, training=True)

          disc_fake_x = self.discriminator_x(fake_x, training=True)
          disc_fake_y = self.discriminator_y(fake_y, training=True)

          # calculate the loss
          gen_g_loss = self.generator_loss(disc_fake_y)
          gen_f_loss = self.generator_loss(disc_fake_x)
          
          total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
          
          # Total generator loss = adversarial loss + cycle loss
          total_gen_g_loss = gen_g_loss + total_cycle_loss  
          total_gen_f_loss = gen_f_loss + total_cycle_loss 
          if regularization :
              total_gen_g_loss += self.identity_loss(real_y, same_y)
              total_gen_f_loss += self.identity_loss(real_x, same_x)

          disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
          disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                              self.generator_f.trainable_variables)
        
        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                  self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                  self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                  self.generator_f.trainable_variables))
        
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      self.discriminator_x.trainable_variables))
        
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      self.discriminator_y.trainable_variables))
        
        return disc_y_loss , total_gen_g_loss, disc_x_loss, total_gen_f_loss
    

      
    def train(self, train_x, train_y, fixed_x, regularization=False, LAMBDA = 10, epochs=10, interval_save = 5, checkpoint_dir = './cycletraining_checkpoints', title_g = "cyclegan_generated_image", title_f = "cyclegan_back_generated_image"):
        # create checkpoint to save the training progression 
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_g_optimizer=self.generator_g_optimizer,
                                      generator_f_optimizer=self.generator_f_optimizer,
                                      discriminator_x_optimizer = self.discriminator_x_optimizer,
                                      discriminator_y_optimizer = self.discriminator_y_optimizer,
                                      generator_g=self.generator_g,
                                      generator_f=self.generator_f,
                                      discriminator_x=self.discriminator_x,
                                      discriminator_y=self.discriminator_y)
        self.LAMBDA = LAMBDA

        # loop over epochs : 
        for epoch in range(epochs):
            start = time.time()
            print("====== Epoch {:2d} ======".format(epoch))
            #initiaite the mean loss over the epoch for the discriminator and generator
            epoch_loss_disc_y = Mean()
            epoch_loss_gen_g = Mean()
            epoch_loss_disc_x = Mean()
            epoch_loss_gen_f = Mean()
            n = 0
            #epoch_len = tf.data.experimental.cardinality(data_generator)
            for image_x, image_y in tf.data.Dataset.zip((train_x, train_y)):
                disc_y_loss, total_gen_g_loss, disc_x_loss, total_gen_f_loss = self.train_step(image_x, image_y, regularization)
                epoch_loss_disc_y(disc_y_loss)
                epoch_loss_gen_g(total_gen_g_loss)
                epoch_loss_disc_x(disc_x_loss)
                epoch_loss_gen_f(total_gen_f_loss)
                
                if n % 100 == 0:
                    print ('.', end='')
                n+=1
            
            print ("\nTime for epoch {} is {} sec".format(epoch, time.time()-start))

            #save images and loss images   
            if epoch % interval_save == 0:
                res = np.array([epoch_loss_disc_y.result(), epoch_loss_gen_g.result(), epoch_loss_disc_x.result(), epoch_loss_gen_f.result()] )

                self.training_losses.loc[int(epoch)] = res
                print("Discriminator y: {} , Generator G: {}, Discriminator x: {}, Generator F: {}".format(res[0], res[1], res[2], res[3] ))        
                checkpoint.save(file_prefix = checkpoint_prefix)

                cycled_images = self.generator_g(fixed_x, training=False)
                back_cycled_images = self.generator_f(cycled_images, training = False)
                
                self.save_generated_images(images = cycled_images, epoch = epoch, ch=self.channel, title = title_g)
                self.save_generated_images(images = back_cycled_images, epoch = epoch, ch=self.channel, title = title_f)

            


