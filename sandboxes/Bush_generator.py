# this is inspired by
# https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# works on G.W.Bush images from
# http://vis-www.cs.umass.edu/lfw/

'''
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
from skimage.io import imread
from skimage.transform import rescale
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import matplotlib.pyplot as plt


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class DCGAN(object):
    def __init__(self, img_rows=64, img_cols=64, channel=3):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D

        self.D = Sequential()
        dropout = 0.4
        kernel_size = 3
        strides = 2  # original image size = 64 = pow(strides, 6)
        depth = pow(strides, 4)  # + 6 layers of conv2d will give final layer pow(strides,11)
        alpha = 0.2
        input_shape = (self.img_rows, self.img_cols, self.channel)

        self.D.add(Conv2D(depth*pow(strides,1), kernel_size=kernel_size, strides=strides, input_shape=input_shape,
                          padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*pow(strides,2), kernel_size=kernel_size, strides=strides, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*pow(strides,3), kernel_size=kernel_size, strides=strides, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*pow(strides,4), kernel_size=kernel_size, strides=strides, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*pow(strides,5), kernel_size=kernel_size, strides=strides, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*pow(strides,6), kernel_size=kernel_size, strides=strides, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G

        self.G = Sequential()
        dropout = 0.4
        in_depth = 3*pow(2,7)  # 7 layers of conv2dtranspose
        in_dim = 4  # original image size is 64 = (2^4)*in_dim (4 layers of upsampling)
        kernel_size = 3

        self.G.add(Dense(in_dim*in_dim*in_depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((in_dim, in_dim, in_depth)))
        self.G.add(Dropout(dropout))
        # Out: in_dim x in_dim x in_depth

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(in_depth/pow(2,1)), kernel_size=kernel_size, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # Out: 2*in_dim x 2*in_dim x in_depth/2

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(in_depth/pow(2,2)), kernel_size=kernel_size, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # Out: 4*in_dim x 4*in_dim x in_depth/4

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(in_depth/pow(2,3)), kernel_size=kernel_size, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # Out: 8*in_dim x 8*in_dim x in_depth/8

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(in_depth/pow(2,4)), kernel_size=kernel_size, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # Out: 16*in_dim x 16*in_dim x in_depth/16

        self.G.add(Conv2DTranspose(int(in_depth/pow(2,5)), kernel_size=kernel_size, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # Out: 16*in_dim x 16*in_dim x in_depth/32

        self.G.add(Conv2DTranspose(int(in_depth / pow(2, 6)), kernel_size=kernel_size, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # Out: 16*in_dim x 16*in_dim x in_depth/64

        self.G.add(Conv2DTranspose(int(in_depth/pow(2,7)), kernel_size=kernel_size, padding='same'))
        self.G.add(Activation('sigmoid'))
        # Out: 64 x 64 x 3

        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])
        return self.AM


class BUSH_DCGAN(object):
    def __init__(self, wanted_size=64):
        self.img_rows = wanted_size
        self.img_cols = wanted_size
        self.channel = 3

        self.x_train = load_data()

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=64, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

        x = self.x_train
        # use data augmentation
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True)
        datagen.fit(x)
        # we use a dummy label here because we need to enter something to the automatic augmenter,
        # later we will make a batch specific label
        y_dummy = np.ones([x.shape[0], 1])

        for i in range(train_steps):

            # train discriminator on fake and real images
            batches = 0
            for x_batch, y_dummy_batch in datagen.flow(x, y_dummy, batch_size=batch_size):
                # generate input for fake images
                current_batch_size = x_batch.shape[0]
                noise = np.random.uniform(-1.0, 1.0, size=[current_batch_size, 100])
                x_fake = self.generator.predict(noise)
                x_batch_discriminator = np.concatenate((x_batch, x_fake))
                y_batch_discriminator = np.ones([2 * current_batch_size, 1])
                y_batch_discriminator[current_batch_size:, :] = 0
                d_loss = self.discriminator.train_on_batch(x_batch_discriminator, y_batch_discriminator)
                # we need to break the loop by hand because
                # the generator loops indefinitely
                batches += current_batch_size
                if batches >= x.shape[0]:
                    break

            y_adversarial = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y_adversarial)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],
                                     noise=noise_input, step=(i+1))
                    self.adversarial.save('BushGAN_adversarial.h5')
                    self.discriminator.save('BushGAN_discriminator.h5')
                    self.generator.save('BushGAN_generator.h5')

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'GWB_real.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "GWB_fake_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


def num2paddedstring(i):
    if i<10:
        return '00' + str(i)
    elif i<100:
        return '0' + str(i)
    else:
        return str(i)


def load_one_image(i, wanted_size=64):
    i_string = num2paddedstring(i + 1)
    im = imread("C:\\Users\\Rey\\Projects\\ProP\\sandboxes\\lfw\\George_W_Bush\\George_W_Bush_0%s.jpg" % i_string)
    original_size = im.shape[0]
    im_smaller = rescale(im, (wanted_size/original_size), mode='constant')
    return im_smaller


def load_data(n_images=530):
    # load images of G.W.bush
    im_arr = np.array([load_one_image(i) for i in range(n_images)])
    return im_arr


if __name__ == '__main__':
    mnist_dcgan = BUSH_DCGAN(wanted_size=64)
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=32, save_interval=500)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)







