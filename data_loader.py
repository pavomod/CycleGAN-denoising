import tensorflow as tf
import numpy as np


SALT=0.15
PEPPER=0.15

def load_mnist(train_start=0, train_end=1000, val_start=1001, val_end=1100, test_start=1101, test_end=1200):
    (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x = np.expand_dims(x, axis=-1)
    x = x.astype('float32') / 255.0
    
    train_x = x[train_start:train_end]
    val_x = x[val_start:val_end]
    test_x = x[test_start:test_end]
    
    return train_x, val_x, test_x

def add_salt_pepper_noise(images, salt_prob=SALT, pepper_prob=PEPPER):
    batch_size, height, width, channels = images.shape
    noise = np.random.rand(batch_size, height, width, channels)
    noisy_images = np.where(noise < salt_prob, 1.0, images)
    noisy_images = np.where(noise > (1 - pepper_prob), 0.0, noisy_images)
    return noisy_images.astype('float32')


#Test
def add_gaussian_noise(images, mean=0, std=0.3):
    gaussian_noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + gaussian_noise
    noisy_images = np.clip(noisy_images, 0, 1)  
    return noisy_images.astype('float32')