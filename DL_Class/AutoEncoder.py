import Tool_DL

from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model


def build_model():
    input_layer = Input((28, 28, 1))
    en = Conv2D(8, (4, 4), strides=2, padding='same', activation='relu')(input_layer)
    en = Conv2D(4, (4, 4), strides=2, padding='same', activation='relu')(en)
    f = Flatten()(en)
    d = Dense(64, activation='relu')(f)
    d = Dense(4, activation='sigmoid')(d)
    d = Dense(7*7*4, activation='sigmoid')(d)
    r = Reshape((7, 7, 4))(d)
    de = Conv2DTranspose(4, (4, 4), strides=2, padding='same', activation='relu')(r)
    de = Conv2DTranspose(8, (4, 4), strides=2, padding='same', activation='relu')(de)
    output_layer = Conv2D(1, (4, 4), padding='same', activation="sigmoid")(de)

    return Model(input_layer, output_layer)


def training(model, x, y, optimizer="adam", loss="mse", epochs=3, batch_size=32, **kwargs):
    if kwargs == {}:
        model.compile(optimizer=optimizer, loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])

    model.summary()
    history = model.fit(x, y, epochs=epochs,  batch_size=batch_size, verbose=1)
    Tool_DL.show_training(history, ['loss'], model_name=__name__)


def run():
    (train_x, _), (test_x, _) = Tool_DL.get_mnist()

    model_ae = build_model()
    # RGB --> mse / Gray --> binary_crossentropy
    training(model_ae, train_x, train_x, loss='binary_crossentropy', epochs=20)
    model_ae.save(f'Result/model_{__name__}.h5')


if __name__ == '__main__':
    run()
