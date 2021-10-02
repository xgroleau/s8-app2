from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


def create_driving_model():
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=(31,)))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=2, activation='tanh'))
    model.compile(optimizer=Adam(lr=0.005), loss='mean_squared_error')

    print(model.summary())

    return model