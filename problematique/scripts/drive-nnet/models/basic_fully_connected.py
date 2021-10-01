# -*- coding: utf-8 -*-
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

def create_model():
    # Define training parameters
    # TODO : Tune the training parameters
    model = Sequential()
    model.add(Dense(units=18, activation='relu', input_shape=(6,)))
    model.add(Dense(units=9, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.compile(optimizer=Adam(lr=0.05), loss='mean_squared_error')
    
    print(model.summary())

    return model
    # Perform training
    # TODO : Tune the maximum number of iterations
    model.fit(data, target, batch_size=len(data),
              epochs=500, shuffle=False, verbose=1)

    # Save trained model to disk
    model.save('iris.h5')

    # Test model (loading from disk)
    model = load_model('iris.h5')
    targetPred = model.predict(data)