from image import denormalize_image
from data import generate_noise
import matplotlib.pyplot as plt


def plot_images(epoch, generator, size=25, dim=(5,5), figsize=(10,10)):
    noise= generate_noise(size)
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(denormalize_image(generated_images[i]), interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('figures/image_' + str(epoch) + '.png')
    
    
def plot_loss(epoch, g_losses, d_losses):
    plt.figure(figsize=(10,5))
    plt.title("Loss, Epochs 0-" + str(epoch))
    plt.plot(g_losses,label="Generator")
    plt.plot(d_losses,label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('figures/loss_' + str(epoch) + '.png')