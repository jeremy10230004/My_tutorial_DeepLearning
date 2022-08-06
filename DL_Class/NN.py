import Tool_DL
import pandas as pd
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout
from tensorflow.keras.models import Model


def preprocess(df):
    """
    x1 : Fare
    x2 : Pclass
    x3 : Sex
    """
    # .fillna(train_df["..."].median() fill up NAN with median
    train_x1 = (df["Fare"].fillna(df["Fare"].median())).to_numpy()
    train_x1 = Tool_DL.normalization(train_x1)
    train_x2 = pd.get_dummies(df["Pclass"]).to_numpy(dtype="float32")  # 3 categories

    for idx, sx in enumerate(df['Sex']):
        if sx == "male":
            df.loc[idx, "Sex"] = 1
        elif sx == "female":
            df.loc[idx, "Sex"] = 0

    train_x3 = df["Sex"].to_numpy(dtype="float32")

    train_y1 = df["Survived"]
    return [train_x1, train_x2, train_x3], train_y1


def build_model():
    input_layer_1 = Input(1)
    input_layer_2 = Input(3)
    input_layer_3 = Input(1)
    con = concatenate([input_layer_1, input_layer_2, input_layer_3])
    de = Dense(20, activation="sigmoid")(con)
    de = Dropout(0.3)(de)
    de = Dense(10, activation="sigmoid")(de)
    de = Dropout(0.3)(de)
    output_layer = Dense(1, activation="sigmoid")(de)
    return Model([input_layer_1, input_layer_2, input_layer_3], output_layer)


def training(model, x, y, optimizer="adam", loss="mse", epochs=3, batch_size=32, **kwargs):
    if kwargs == {}:
        model.compile(optimizer=optimizer, loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])

    model.summary()
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
    Tool_DL.show_training(history, ['loss', 'accuracy'], model_name=__name__)


def run():
    train_df = pd.read_csv("DataSet/titanic/train.csv")
    test_df = pd.read_csv("DataSet/titanic/test.csv")

    train_x, train_y = preprocess(train_df)
    test_x, test_y = preprocess(test_df)

    model_nn = build_model()
    training(model_nn, train_x, train_y, loss='binary_crossentropy', metrics=['accuracy'], epochs=150)
    loss, acc = Tool_DL.testing(model_nn, test_x, test_y)
    model_nn.save(f'Result/model_{__name__}.h5')


if __name__ == '__main__':
    run()
