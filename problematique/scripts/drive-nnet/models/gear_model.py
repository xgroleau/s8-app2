from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def create_gear_model():
    model = Sequential()
    model.add(Dense(units=12, activation='sigmoid', input_shape=(9,)))
    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.005), loss='mse')

    print(model.summary())

    return model