import tensorflow as tf
import numpy as np

def load_mnist(start=0, end=1000):
    (train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_x = np.expand_dims(train_x, axis=-1)
    train_x = train_x.astype('float32') / 255.0
    train_x = train_x[start:end]
    return train_x

def add_salt_pepper_noise(images, salt_prob=0.15, pepper_prob=0.15):
    batch_size, height, width, channels = images.shape
    # Crea una maschera con valori random
    noise = np.random.rand(batch_size, height, width, channels)
    # Applica il rumore 'sale'
    noisy_images = np.where(noise < salt_prob, 1.0, images)
    # Applica il rumore 'pepe'
    noisy_images = np.where(noise > (1 - pepper_prob), 0.0, noisy_images)
    return noisy_images.astype('float32')


#Test
def add_gaussian_noise(images, mean=0, std=0.3):
    gaussian_noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + gaussian_noise
    noisy_images = np.clip(noisy_images, 0, 1)  
    return noisy_images.astype('float32')