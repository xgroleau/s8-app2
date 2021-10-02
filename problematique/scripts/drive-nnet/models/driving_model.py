from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

def create_driving_model():
    # Define training parameters
    # TODO : Tune the training parameters
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_shape=(4,)))
    model.add(Dense(units=3, activation='relu'))
    model.compile(optimizer=Adam(lr=0.005), loss='mean_squared_error')

    print(model.summary())

    return model