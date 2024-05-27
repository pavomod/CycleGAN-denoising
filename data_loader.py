import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from params import SALT, PEPPER, ROBOTICS_TRAIN_PATH, ROBOTICS_VAL_PATH, ROBOTICS_TEST_PATH
import tqdm
import matplotlib.pyplot as plt

def load_mnist(train_size=1000, val_size=1000, test_size=1000, seed=42):
    np.random.seed(seed)  
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    x = np.expand_dims(x, axis=-1)
    x = x.astype('float32') / 255.0
    x = (x - 0.5) * 2

    def get_balanced_set(size):
        indices = []
        for digit in range(10):
            digit_indices = np.where(y == digit)[0]
            np.random.shuffle(digit_indices)
            indices.extend(digit_indices[:size // 10])
        np.random.shuffle(indices)
        return x[indices]

    train_x = get_balanced_set(train_size)
    val_x = get_balanced_set(val_size)
    test_x = get_balanced_set(test_size)
    
    return train_x, val_x, test_x

def load_robotics_data(target_shape=(369, 496, 1), max_train_images=1000):
    train_images = []
    val_images = []
    test_images = []
    
    def load_and_resize_image(file_path):
        img = tf.keras.preprocessing.image.load_img(file_path, color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, target_shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Verifica la dimensione dell'immagine dopo il ridimensionamento
        if img.shape != target_shape:
            print(f"Dimensione errata per {file_path}: {img.shape}, attesa: {target_shape}")
        return img
    
    for i, filename in enumerate(tqdm.tqdm(os.listdir(ROBOTICS_TRAIN_PATH))):
        if i >= max_train_images:
            break
        if filename.endswith(".png"):
            try:
                img = load_and_resize_image(os.path.join(ROBOTICS_TRAIN_PATH, filename))
                train_images.append(img)
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {filename}: {e}")
    
    for filename in tqdm.tqdm(os.listdir(ROBOTICS_VAL_PATH)):
        if filename.endswith(".png"):
            try:
                img = load_and_resize_image(os.path.join(ROBOTICS_VAL_PATH, filename))
                val_images.append(img)
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {filename}: {e}")
    
    for filename in tqdm.tqdm(os.listdir(ROBOTICS_TEST_PATH)):
        if filename.endswith(".png"):
            try:
                img = load_and_resize_image(os.path.join(ROBOTICS_TEST_PATH, filename))
                test_images.append(img)
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {filename}: {e}")
    
    train_images = np.array(train_images)
    val_images = np.array(val_images)
    test_images = np.array(test_images)



    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    train_images = (train_images - 0.5) * 2
    val_images = (val_images - 0.5) * 2
    test_images = (test_images - 0.5) * 2
    
    return train_images, val_images, test_images


def add_salt_pepper_noise(images, salt_prob=SALT, pepper_prob=PEPPER):
    batch_size, height, width, channels = images.shape
    noise = np.random.rand(batch_size, height, width, channels)
    noisy_images = np.where(noise < salt_prob, 1.0, images)
    noisy_images = np.where(noise > (1 - pepper_prob), 0.0, noisy_images)
    return noisy_images.astype('float32')


def remove_pixel(images, remove_prob=0.2):
    batch_size, height, width, channels = images.shape
    noise = np.random.rand(batch_size, height, width, channels)
    white_pixel_mask = (images >= 0.99)  
    removal_mask = (noise < remove_prob) & white_pixel_mask
    noisy_images = np.where(removal_mask, -1.0, images)
    
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