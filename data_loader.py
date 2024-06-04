import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from params import SALT, PEPPER, ROBOTICS_TRAIN_PATH, ROBOTICS_VAL_PATH, ROBOTICS_TEST_PATH, DASH_LENGTH, SPACE_LENGTH
import tqdm

def load_mnist(train_size=1000, val_size=1000, test_size=1000, seed=42):
    '''
    Loads the MNIST dataset and returns balanced subsets for training, validation, and testing. 
    Each subset contains an equal number of samples from each digit class. The sizes of the subsets can be 
    specified, and a random seed is used for reproducibility.

    Parameters:
        train_size (int): Number of samples in the training set.
        val_size (int): Number of samples in the validation set.
        test_size (int): Number of samples in the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Three numpy arrays containing the training, validation, and test sets.
    '''
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

def load_robotics_data(target_shape=(128, 128, 1), max_train_images=5000):
    '''
    Loads robotics image data from specified directories and resizes the images to a target shape. 
    It returns three sets of images: training, validation, and testing. Each image is converted to grayscale, resized, 
    normalized to the range [-1, 1], and stored in numpy arrays.

    Parameters:
        target_shape (tuple): The desired shape of the output images (height, width, channels).
        max_train_images (int): Maximum number of training images to load.

    Returns:
        tuple: Three numpy arrays containing the training, validation, and test sets.
    '''
    train_images = []
    val_images = []
    test_images = []
    
    def load_and_resize_image(file_path):
        img = tf.keras.preprocessing.image.load_img(file_path, color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, target_shape[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
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
    '''
    Adds salt and pepper noise to a batch of images. Salt noise sets some pixels to the maximum value (1.0), 
    and pepper noise sets some pixels to the minimum value (0.0). The probabilities for salt and pepper noise 
    can be specified.

    Parameters:
        images (numpy array): Batch of images to which noise will be added.
        salt_prob (float): Probability of a pixel being set to 1.0 (salt noise).
        pepper_prob (float): Probability of a pixel being set to 0.0 (pepper noise).

    Returns:
        numpy array: Batch of images with added salt and pepper noise.
    '''
    batch_size, height, width, channels = images.shape
    noise = np.random.rand(batch_size, height, width, channels)
    noisy_images = np.where(noise < salt_prob, 1.0, images)
    noisy_images = np.where(noise > (1 - pepper_prob), 0.0, noisy_images)
    return noisy_images.astype('float32')


def remove_pixel(images, dash_length=DASH_LENGTH, space_length=SPACE_LENGTH):
    '''
    Applies a dashed line effect to a batch of images by setting horizontal bands of pixels to black. 
    The length of the dashed lines and the spaces between them can be specified. Images are converted 
    to [0, 255] range for processing and then back to [-1, 1] range.

    Parameters:
        images (numpy array): Batch of images to process.
        dash_length (int): Length of the space between dashed lines.
        space_length (int): Length of the black dashed lines.

    Returns:
        numpy array: Batch of images with dashed line effect applied.
    '''
    processed_images = []

    for image in images:
        # Convertire l'immagine da [-1, 1] a [0, 255]
        img_uint8 = ((image + 1) * 127.5).astype(np.uint8).copy()

        # Dimensioni dell'immagine
        height = img_uint8.shape[0]

        # Disegna linee orizzontali nere
        for y in range(0, height, dash_length + space_length):
            end_y = min(y + space_length, height)
            img_uint8[y:end_y, :] = 0  # Imposta i pixel a nero nella banda orizzontale

        # Convertire l'immagine da [0, 255] a [-1, 1]
        processed_image = (img_uint8 / 127.5) - 1
        processed_images.append(processed_image)


    return np.array(processed_images).astype('float32')

def save_models(generator_G, generator_F, discriminator_X, discriminator_Y, epoch):
    '''
    Saves the models (generators and discriminators) to a specified directory named after the current epoch. 
    Each model is saved in HDF5 format (.h5).

    Parameters:
        generator_G (tf.keras.Model): The generator G model to save.
        generator_F (tf.keras.Model): The generator F model to save.
        discriminator_X (tf.keras.Model): The discriminator X model to save.
        discriminator_Y (tf.keras.Model): The discriminator Y model to save.
        epoch (int): The current epoch number used for naming the save directory.
    '''
    save_dir = f'models/epoch_{epoch+1}'
    os.makedirs(save_dir, exist_ok=True)
    generator_G.save(os.path.join(save_dir, 'generator_G.h5'))
    generator_F.save(os.path.join(save_dir, 'generator_F.h5'))
    discriminator_X.save(os.path.join(save_dir, 'discriminator_X.h5'))
    discriminator_Y.save(os.path.join(save_dir, 'discriminator_Y.h5'))
    print(f"Models saved to {save_dir}\n\n")

def load_models(epoch=1):
    '''
    Loads the models (generators and discriminators) from a specified directory named after the epoch. 
    Each model is loaded from HDF5 format (.h5).

    Parameters:
        epoch (int): The epoch number used for naming the load directory.

    Returns:
        tuple: Loaded generator G, generator F, discriminator X, and discriminator Y models.
    '''
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