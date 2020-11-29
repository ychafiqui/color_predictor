import pickle

x_train, y_train = [], []
with open('dataset2.pkl', 'wb') as f:
    pickle.dump([x_train, y_train], f) # saving the new dataset with pickle