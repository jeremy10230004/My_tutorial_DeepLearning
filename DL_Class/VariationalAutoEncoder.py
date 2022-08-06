import Tool_DL
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model


class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_model():
    input_layer = Input((28, 28, 1))
    en = Conv2D(8, (4, 4), strides=2, padding='same', activation='relu')(input_layer)
    en = Conv2D(4, (4, 4), strides=2, padding='same', activation='relu')(en)
    f = Flatten()(en)
    d = Dense(64, activation='relu')(f)
    c = Dense(4, activation='sigmoid')(d)

    latent_dim = 4
    z_mean = Dense(latent_dim, name="z_mean")(c)
    z_log_var = Dense(latent_dim, name="z_log_var")(c)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(input_layer, [z_mean, z_log_var, z])

    de_input = Input(shape=(latent_dim))
    d = Dense(7 * 7 * 4, activation='sigmoid')(de_input)
    r = Reshape((7, 7, 4))(d)
    de = Conv2DTranspose(4, (4, 4), strides=2, padding='same', activation='relu')(r)
    de = Conv2DTranspose(8, (4, 4), strides=2, padding='same', activation='relu')(de)
    output_layer = Conv2D(1, (4, 4), padding='same', activation="sigmoid")(de)
    decoder = Model(de_input, output_layer)

    return VAE(encoder, decoder)


def training(model, x, y, optimizer="adam", loss="mse", epochs=3, batch_size=32, **kwargs):
    if kwargs == {}:
        model.compile(optimizer=optimizer, loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])

    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
    Tool_DL.show_training(history, ['loss'], model_name=__name__)


def run():
    (train_x, _), (test_x, _) = Tool_DL.get_mnist()

    model_vae = build_model()
    # RGB --> mse / Gray --> binary_crossentropy
    training(model_vae, train_x, train_x, loss='binary_crossentropy', epochs=20)
    model_vae.encoder.save(f'Result/model_{__name__}_encoder.h5')
    model_vae.decoder.save(f'Result/model_{__name__}_decoder.h5')


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, y = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, input_x):
        de_input = self.encoder.predict(input_x)
        ans = self.decoder.predict(de_input[2])
        return ans

    def get_code(self, input_x):
        return self.encoder.predict(input_x)


if __name__ == '__main__':
    run()
