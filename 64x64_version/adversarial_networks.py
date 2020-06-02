from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, \
    BatchNormalization, UpSampling2D, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam


INPUT_SIZE = 100


def create_generator():
    generator=Sequential()
    generator.add(Dense(units=512*8*8,input_dim=INPUT_SIZE))
    generator.add(Reshape((8,8,512)))

    generator.add(Conv2DTranspose(512, 4, strides=1, activation='relu', padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    
    generator.add(Conv2DTranspose(256, 4, strides=2, activation='relu', padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    
    generator.add(Conv2DTranspose(128, 4, strides=2, activation='relu', padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    
    generator.add(Conv2DTranspose(64, 4, strides=2, activation='relu', padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    
    generator.add(Conv2DTranspose(3, 4, strides=1, activation='tanh', padding='same'))
        
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return generator


def create_discriminator():
    discriminator=Sequential()

    discriminator.add(Conv2D(64, kernel_size = 4, strides = 2, padding = 'same', input_shape=(64,64,3)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, kernel_size = 4, strides = 2, padding = 'same'))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(256, kernel_size = 4, strides = 2, padding = 'same'))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(512, kernel_size = 4, strides = 2, padding = 'same'))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(1, kernel_size = 4, strides = 1, padding = 'same'))

    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return discriminator


def create_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = Input(shape=(INPUT_SIZE,))
    generator_output = generator(gan_input)
    gan_output = discriminator(generator_output)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return gan
