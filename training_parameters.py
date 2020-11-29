import tensorflow as tf
import pickle

# getting the dataset with pickle
with open('dataset.pkl', 'rb') as f:
    x_train, y_train = pickle.load(f)

# model layers
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

# optimisze, loss and metrics
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# testing if the dataset is empty
if len(x_train) > 0:
    model.fit(x_train, y_train, epochs=20) # model training
    model.save("color_predictor.h5", overwrite=False) # saving the model
else: print("Dataset is empty, you need to chose at least one color!")