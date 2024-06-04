import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import tensorflow_addons as tfa
from params import NUM_RESNET_BLOCKS, KERNEL_SIZE_RESNET

def resnet_block(input_layer, filters, kernel_size=KERNEL_SIZE_RESNET):
    '''
    Creates a ResNet block with two convolutional layers, instance normalization, and ReLU activation.
    The block adds the input to the output to form a residual connection, facilitating gradient flow 
    during training.

    Parameters:
        input_layer (tensor): Input tensor to the ResNet block.
        filters (int): Number of filters for the convolutional layers.
        kernel_size (int or tuple): Size of the convolution kernels.

    Returns:
        tensor: Output tensor after applying the ResNet block.
    '''
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer=initializer, use_bias=False)(input_layer)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    return layers.add([x, input_layer])

def make_generator_model(shape=(28, 28, 1)):
    '''
    Creates a generator model using convolutional layers, instance normalization, and ReLU activation. 
    The model architecture includes an initial convolution block, downsampling layers, multiple ResNet 
    blocks, and upsampling layers, culminating in a single-channel output with a tanh activation.

    Parameters:
        shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The generator model.
    '''
    inputs = layers.Input(shape=shape)
    
    # Initial Convolution block
    x = layers.Conv2D(64, (7, 7), strides=1, padding='same')(inputs)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    # Downsampling
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet blocks
    for _ in range(NUM_RESNET_BLOCKS):
        x = resnet_block(x, 256)

    # Upsampling
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer
    x = layers.Conv2D(1, (7, 7), padding='same', activation='tanh')(x)

    return Model(inputs=inputs, outputs=x)



def make_discriminator_model(shape=(28, 28, 1)):
    '''
    Creates a discriminator model using convolutional layers, LeakyReLU activation, and dropout. 
    The model architecture includes multiple convolutional layers for feature extraction, followed by 
    a dense layer for binary classification.

    Parameters:
        shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The discriminator model.
    '''
    model = tf.keras.Sequential([
        layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model



class CycleGAN(Model):
    def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y, lambda_cycle=10):
        '''
        Initializes the CycleGAN model with given generators and discriminators. The CycleGAN consists of 
        two generators (G and F) and two discriminators (for domains X and Y). The cycle consistency loss 
        weight is also set.

        Parameters:
            generator_G (tf.keras.Model): Generator model for transforming X to Y.
            generator_F (tf.keras.Model): Generator model for transforming Y to X.
            discriminator_X (tf.keras.Model): Discriminator model for domain X.
            discriminator_Y (tf.keras.Model): Discriminator model for domain Y.
            lambda_cycle (int): Weight for the cycle consistency loss.
        '''
        super(CycleGAN, self).__init__()
        self.generator_G = generator_G  
        self.generator_F = generator_F 
        self.discriminator_X = discriminator_X  
        self.discriminator_Y = discriminator_Y  
        self.lambda_cycle = lambda_cycle 

    def call(self, inputs, training=False):
        '''
        Executes the forward pass of the CycleGAN model. It generates fake images, performs cycle consistency,
        and computes discriminator outputs for real and fake images. In training mode, it returns additional 
        outputs needed for computing losses.

        Parameters:
            inputs (tuple): Tuple containing real images from domain X and domain Y (real_x, real_y).
            training (bool): Boolean flag indicating whether the model is in training mode.

        Returns:
            tuple: In inference mode, returns fake_y, cycled_x, fake_x, cycled_y. 
                In training mode, returns additional discriminator outputs: 
                fake_y, cycled_x, fake_x, cycled_y, disc_real_x, disc_fake_x, disc_real_y, disc_fake_y.
        '''
        real_x, real_y = inputs

        fake_y = self.generator_G(real_x, training=training)
        cycled_x = self.generator_F(fake_y, training=training)
        fake_x = self.generator_F(real_y, training=training)
        cycled_y = self.generator_G(fake_x, training=training)

        if not training:
            return fake_y, cycled_x, fake_x, cycled_y

        disc_real_x = self.discriminator_X(real_x, training=training)
        disc_fake_x = self.discriminator_X(fake_x, training=training)
        disc_real_y = self.discriminator_Y(real_y, training=training)
        disc_fake_y = self.discriminator_Y(fake_y, training=training)

        return fake_y, cycled_x, fake_x, cycled_y, disc_real_x, disc_fake_x, disc_real_y, disc_fake_y
    
    def compile(self, g_optimizer, d_optimizer, gen_loss_fn, disc_loss_fn, cycle_loss_fn,identity_loss_fn):
        '''
        Configures the CycleGAN model for training by setting the optimizers and loss functions. This method 
        is called before training the model.

        Parameters:
            g_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator models.
            d_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator models.
            gen_loss_fn (function): Loss function for the generators.
            disc_loss_fn (function): Loss function for the discriminators.
            cycle_loss_fn (function): Loss function for the cycle consistency.
            identity_loss_fn (function): Loss function for the identity mapping.
        '''
        super(CycleGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss = identity_loss_fn

    def train_step(self, data):
        '''
        Performs a single training step for the CycleGAN model. This includes forward passes through the generators 
        and discriminators, calculating the losses, computing the gradients, and applying the gradients to update 
        the model weights.

        Parameters:
            data (tuple): A tuple containing real images from domain X and domain Y (real_x, real_y).

        Returns:
            dict: A dictionary containing the losses for the generators and discriminators.
        '''
        real_x, real_y = data

        # print real_x.shape, real_y.shape

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through the network
            fake_y = self.generator_G(real_x, training=True)
            cycled_x = self.generator_F(fake_y, training=True)
            fake_x = self.generator_F(real_y, training=True)
            cycled_y = self.generator_G(fake_x, training=True)

            #perdita di identità
            same_y = self.generator_G(real_y, training=True)
            same_x = self.generator_F(real_x, training=True)
            identity_loss_G = self.identity_loss(real_y, same_y)
            identity_loss_F = self.identity_loss(real_x, same_x)
            # Discriminator outputs
            disc_real_x = self.discriminator_X(real_x, training=True)
            disc_fake_x = self.discriminator_X(fake_x, training=True)
            disc_real_y = self.discriminator_Y(real_y, training=True)
            disc_fake_y = self.discriminator_Y(fake_y, training=True)

            # Calculate the generator and discriminator losses
            gen_g_loss = self.gen_loss_fn(disc_fake_y) + identity_loss_G * self.lambda_cycle
            gen_f_loss = self.gen_loss_fn(disc_fake_x) + identity_loss_F * self.lambda_cycle
            total_cycle_loss = self.cycle_loss_fn(real_x, cycled_x) + self.cycle_loss_fn(real_y, cycled_y)
            total_gen_g_loss = gen_g_loss + self.lambda_cycle * total_cycle_loss
            total_gen_f_loss = gen_f_loss + self.lambda_cycle * total_cycle_loss
            disc_x_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

        # Calculate the gradients for each generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_G.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_F.trainable_variables)
        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_X.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_Y.trainable_variables)

        # Apply the gradients separately for each generator and discriminator
        self.g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_G.trainable_variables))
        self.g_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_F.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_x_gradients, self.discriminator_X.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_y_gradients, self.discriminator_Y.trainable_variables))

        return {
            "G_loss": total_gen_g_loss,
            "F_loss": total_gen_f_loss,
            "D_X_loss": disc_x_loss,
            "D_Y_loss": disc_y_loss
        }
    
    def test_step(self, data):
        '''
        Performs a single test step for the CycleGAN model. This includes forward passes through the generators 
        and discriminators, and calculating the losses without updating the model weights.

        Parameters:
            data (tuple): A tuple containing real images from domain X and domain Y (real_x, real_y).

        Returns:
            dict: A dictionary containing the losses for the generators and discriminators.
        '''
        real_x, real_y = data

        # Forward pass through the network
        fake_y = self.generator_G(real_x, training=False)
        cycled_x = self.generator_F(fake_y, training=False)
        fake_x = self.generator_F(real_y, training=False)
        cycled_y = self.generator_G(fake_x, training=False)

        # Discriminator outputs
        disc_real_x = self.discriminator_X(real_x, training=False)
        disc_fake_x = self.discriminator_X(fake_x, training=False)
        disc_real_y = self.discriminator_Y(real_y, training=False)
        disc_fake_y = self.discriminator_Y(fake_y, training=False)

        # Calculate the generator and discriminator losses
        gen_g_loss = self.gen_loss_fn(disc_fake_y)
        gen_f_loss = self.gen_loss_fn(disc_fake_x)
        total_cycle_loss = self.cycle_loss_fn(real_x, cycled_x) + self.cycle_loss_fn(real_y, cycled_y)
        total_gen_g_loss = gen_g_loss + self.lambda_cycle * total_cycle_loss
        total_gen_f_loss = gen_f_loss + self.lambda_cycle * total_cycle_loss
        disc_x_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
        disc_y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

        # Return the loss values to summarize performance
        return {
            "G_loss": total_gen_g_loss,
            "F_loss": total_gen_f_loss,
            "D_X_loss": disc_x_loss,
            "D_Y_loss": disc_y_loss
        }
    
    
