from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from adversarial_networks_scaled import create_generator, create_discriminator, create_gan, INPUT_SIZE
from data import load_images, save_images, generate_noise
from image import denormalize_image
from plot import plot_images, plot_loss
import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
PLOT_FRECUENCY = 5
def load_imgs(image_path):
        X_train = []
        for i in glob.glob(image_path):
            img = Image.open(i)
            img = np.asarray(img)
            X_train.append(img)
        return np.asarray(X_train)

def training(epochs=1, batch_size=32):
    #Loading Data
    #x_train = load_imgs("./data256/*png")

    # Rescale -1 to 1
    #x_train = x_train / 127.5 - 1.
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        'imgs/', batch_size=batch_size, shuffle=True)
    batches = 12385. / batch_size 

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)
    
    # Adversarial Labels
    y_valid = np.ones(batch_size)*0.9
    y_fake = np.zeros(batch_size)
    discriminator_loss, generator_loss = [], []

    for epoch in range(1, epochs+1):
        print('-'*15, 'Epoch', epoch, '-'*15)
        g_loss = 0; d_loss = 0

        for _ in tqdm(range(int(batches-1))):
            # Random Noise and Images Set
            noise = generate_noise(batch_size)
            image_batch, _ = train_generator.next()
            image_batch = image_batch / 127.5 - 1.
            if image_batch.shape[0] == batch_size:
                # Generate Fake Images
                generated_images = generator.predict(noise)

                # Train Discriminator (Fake and Real)
                discriminator.trainable = True
                d_valid_loss = discriminator.train_on_batch(image_batch, y_valid)
                d_fake_loss = discriminator.train_on_batch(generated_images, y_fake)            

                d_loss += (d_fake_loss + d_valid_loss)/2

                # Train Generator
                noise = generate_noise(batch_size)
                discriminator.trainable = False
                g_loss += gan.train_on_batch(noise, y_valid)
        train_generator.on_epoch_end()
            
        discriminator_loss.append(d_loss/batches)
        generator_loss.append(g_loss/batches)
            
        if epoch % PLOT_FRECUENCY == 0:
            plot_images(epoch, generator)
            plot_loss(epoch, generator_loss, discriminator_loss)
            save_path = "./models"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            discriminator.save(save_path + "/discrim_%s.h5"%epoch)
            generator.save(save_path + "/generat_%s.h5"%epoch)
            gan.save(save_path + "/gan_%s.h5"%epoch)
            
    save_images(generator)


if __name__ == '__main__':
    training(epochs=10000)
    