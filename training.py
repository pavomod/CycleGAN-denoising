import tensorflow as tf
from models import make_generator_model, make_discriminator_model, CycleGAN
from data_loader import load_mnist, add_salt_pepper_noise
from utils import *
import numpy as np
import tensorflow_addons as tfa
from tqdm import tqdm
from loss import generator_loss, discriminator_loss, cycle_consistency_loss

EPOCHS = 10
BATCH_SIZE = 1
TRAIN_IMAGES = 1000
TEST_IMAGES = 1000

def test_model(model, test_dataset, num_images=5, plot=True):
    test_losses = []
    for image_batch, noisy_batch in test_dataset:
        generated_images = model.generator_G(noisy_batch, training=False)
        generated_images = generated_images.numpy()
        if plot:
            plot_images(image_batch.numpy(), noisy_batch.numpy(), generated_images, num_images=num_images)
        data = (image_batch, noisy_batch)
        test_loss = model.test_step(data)
        test_losses.append(test_loss)
    avg_G_loss = np.mean([loss['G_loss'] for loss in test_losses])
    avg_F_loss = np.mean([loss['F_loss'] for loss in test_losses])
    avg_DX_loss = np.mean([loss['D_X_loss'] for loss in test_losses])
    avg_DY_loss = np.mean([loss['D_Y_loss'] for loss in test_losses])
    print("Average Generator G Loss:", avg_G_loss)
    print("Average Generator F Loss:", avg_F_loss)
    print("Average Discriminator X Loss:", avg_DX_loss)
    print("Average Discriminator Y Loss:", avg_DY_loss)
    
    return {
        "avg_G_loss": avg_G_loss,
        "avg_F_loss": avg_F_loss,
        "avg_DX_loss": avg_DX_loss,
        "avg_DY_loss": avg_DY_loss
    }


def run():
    physical_devices = tf.config.list_physical_devices('GPU')
    print("GPUs:", physical_devices)
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    print('Using device:', device)

    with tf.device(device):
        gen_G = make_generator_model()
        gen_F = make_generator_model()
        disc_X = make_discriminator_model()
        disc_Y = make_discriminator_model()

        g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


        cycle_gan_model = CycleGAN(gen_G, gen_F, disc_X, disc_Y, lambda_cycle=10)
        cycle_gan_model.compile(
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            gen_loss_fn=generator_loss,
            disc_loss_fn=discriminator_loss,
            cycle_loss_fn=cycle_consistency_loss
        )

        train_images = load_mnist(0, TRAIN_IMAGES)
        noisy_images = add_salt_pepper_noise(train_images)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, noisy_images)).batch(BATCH_SIZE)
        test_images = load_mnist(TRAIN_IMAGES, TEST_IMAGES)
        noisy_test_images = add_salt_pepper_noise(test_images)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, noisy_test_images)).batch(BATCH_SIZE)

        train_g_losses = []
        train_f_losses = []
        train_dx_losses = []
        train_dy_losses = []

        test_g_losses = []
        test_f_losses = []
        test_dx_losses = []
        test_dy_losses = []

        for epoch in range(EPOCHS):  
            for batch, (image_batch, noisy_batch) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}")):
                data = (image_batch, noisy_batch)
                train_loss = cycle_gan_model.train_step(data)
                train_g_losses.append(train_loss["G_loss"].numpy())
                train_f_losses.append(train_loss["F_loss"].numpy())
                train_dx_losses.append(train_loss["D_X_loss"].numpy())
                train_dy_losses.append(train_loss["D_Y_loss"].numpy())

                if batch % 100 == 0:  # Save images every 100 batches
                    generated_images = cycle_gan_model.generator_G(noisy_batch, training=False)
                    save_images(image_batch.numpy(), noisy_batch.numpy(), generated_images.numpy(), epoch, batch)

            avg_test_loss = test_model(cycle_gan_model, test_dataset, num_images=5, plot=False)
            test_g_losses.append(avg_test_loss["avg_G_loss"])
            test_f_losses.append(avg_test_loss["avg_F_loss"])
            test_dx_losses.append(avg_test_loss["avg_DX_loss"])
            test_dy_losses.append(avg_test_loss["avg_DY_loss"])

            print(f"Epoch {epoch+1} complete")

    plot_losses(train_g_losses, test_g_losses, 'G')
    plot_losses(train_f_losses, test_f_losses, 'F')
    plot_losses(train_dx_losses, test_dx_losses, 'D_X')
    plot_losses(train_dy_losses, test_dy_losses, 'D_Y')