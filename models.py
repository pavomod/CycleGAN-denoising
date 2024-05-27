import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import tensorflow_addons as tfa
from params import NUM_RESNET_BLOCKS, KERNEL_SIZE_RESNET

def resnet_block(input_layer, filters, kernel_size=KERNEL_SIZE_RESNET):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer=initializer, use_bias=False)(input_layer)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    return layers.add([x, input_layer])

def make_generator_model(shape=(28, 28, 1)):
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
        super(CycleGAN, self).__init__()
        self.generator_G = generator_G  # Generator G: X -> Y
        self.generator_F = generator_F  # Generator F: Y -> X
        self.discriminator_X = discriminator_X  # Discriminator for X
        self.discriminator_Y = discriminator_Y  # Discriminator for Y
        self.lambda_cycle = lambda_cycle  # Lambda for cycle consistency loss

    def call(self, inputs, training=False):
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
        super(CycleGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss = identity_loss_fn

    def train_step(self, data):
        real_x, real_y = data

        # print real_x.shape, real_y.shape

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through the network
            fake_y = self.generator_G(real_x, training=True)
            print(f"Dimensione fake_y: {fake_y.shape}")

            cycled_x = self.generator_F(fake_y, training=True)
            print(f"Dimensione cycled_x: {cycled_x.shape}")

            fake_x = self.generator_F(real_y, training=True)
            print(f"Dimensione fake_x: {fake_x.shape}")

            cycled_y = self.generator_G(fake_x, training=True)
            print(f"Dimensione cycled_y: {cycled_y.shape}")

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
    
    
