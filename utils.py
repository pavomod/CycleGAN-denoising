
import matplotlib.pyplot as plt
import os



def plot_images(original, noisy, generated, num_images=5):
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
