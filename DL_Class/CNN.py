import Tool_DL

from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model


def build_model():
    input_layer = Input((28, 28, 1))
    con = Conv2D(16, (4, 4), strides=2, padding='same', activation='relu')(input_layer)
    con = Conv2D(8, (4, 4), strides=2, padding='same', activation='relu')(con)
    f = Flatten()(con)
    d = Dense(64, activation='relu')(f)
    d = Dropout(0.3)(d)
    d = Dense(32, activation='relu')(d)
    d = Dropout(0.3)(d)
    output_layer = Dense(10, activation="softmax")(d)

    return Model(input_layer, output_layer)


def training(model, x, y, optimizer="adam", loss="mse", epochs=3, batch_size=32, **kwargs):
    if kwargs == {}:
        model.compile(optimizer=optimizer, loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])

    model.summary()
    history = model.fit(x, y, epochs=epochs,  batch_size=batch_size, verbose=1)
    Tool_DL.show_training(history, ['loss', 'accuracy'], model_name=__name__)


def run():
    (train_x, train_y), (test_x, test_y) = Tool_DL.get_mnist()

    model_cnn = build_model()
    training(model_cnn, train_x, train_y, loss='categorical_crossentropy', metrics=['accuracy'], epochs=10)
    loss, acc = Tool_DL.testing(model_cnn, test_x, test_y)
    model_cnn.save(f'Result/model_{__name__}.h5')


if __name__ == '__main__':
    run()
