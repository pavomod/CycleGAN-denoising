
import matplotlib.pyplot as plt
import os


def plot_images(original, noisy, generated, num_images=5):
    '''
    Plots a specified number of original, noisy, and generated images side by side for comparison. 
    Each row corresponds to a category of images: original, noisy, and generated.

    Parameters:
        original (numpy array): Batch of original images.
        noisy (numpy array): Batch of noisy images.
        generated (numpy array): Batch of generated images.
        num_images (int): Number of images to plot.

    Returns:
        None
    '''
    num_images = min(num_images, original.shape[0])
    
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
        
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(generated[i].squeeze(), cmap='gray')
        plt.title("Generated")
        plt.axis('off')
    plt.show()
  
def save_images(original, noisy, generated, epoch, batch, output_dir="output_images"):
    '''
    Saves a specified number of original, noisy, and generated images side by side to a file for comparison. 
    Creates the output directory if it does not exist.

    Parameters:
        original (numpy array): Batch of original images.
        noisy (numpy array): Batch of noisy images.
        generated (numpy array): Batch of generated images.
        epoch (int or str): Current epoch number, used for naming the saved file.
        batch (int or str): Current batch number, used for naming the saved file.
        output_dir (str): Directory where the images will be saved.

    Returns:
        None
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_images = min(5, original.shape[0])
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(generated[i].squeeze(), cmap='gray')
        plt.title("Generated")
        plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch}.png"))
    plt.close()
