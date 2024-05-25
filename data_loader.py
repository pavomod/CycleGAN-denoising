import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from params import SALT, PEPPER

def load_mnist(train_start=0, train_end=1000, val_start=1001, val_end=1100, test_start=1101, test_end=1200):
    (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x = np.expand_dims(x, axis=-1)
    x = x.astype('float32') / 255.0
    x = (x - 0.5) * 2
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



def save_models(generator_G, generator_F, discriminator_X, discriminator_Y, epoch):
    save_dir = f'models/epoch_{epoch+1}'
    os.makedirs(save_dir, exist_ok=True)
    generator_G.save(os.path.join(save_dir, 'generator_G.h5'))
    generator_F.save(os.path.join(save_dir, 'generator_F.h5'))
    discriminator_X.save(os.path.join(save_dir, 'discriminator_X.h5'))
    discriminator_Y.save(os.path.join(save_dir, 'discriminator_Y.h5'))
    print(f"Models saved to {save_dir}\n\n")

def load_models(epoch=1):
    try:
        load_dir = f'models/epoch_{epoch}'
        generator_G = load_model(os.path.join(load_dir, 'generator_G.h5'))
        generator_F = load_model(os.path.join(load_dir, 'generator_F.h5'))
        discriminator_X = load_model(os.path.join(load_dir, 'discriminator_X.h5'))
        discriminator_Y = load_model(os.path.join(load_dir, 'discriminator_Y.h5'))
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)
    print(f"Models loaded from {load_dir}\n\n")
    return generator_G, generator_F, discriminator_X, discriminator_Y