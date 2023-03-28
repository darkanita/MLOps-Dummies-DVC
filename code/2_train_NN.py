import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

import joblib

from dvclive import Live
from dvclive.keras import DVCLiveCallback

import yaml

params = yaml.safe_load(open("params.yaml"))["train"]

batch_size = params["batch_size"]
epochs = params["epochs"]


input_dir = 'data/prepared'
output_dir = 'model'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_dir + '/housepricedata.csv')

dataset = df.values

X = dataset[:,0:10]
y = dataset[:,10]


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_scale, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)


#  sequentially (layer by layer)
#  ‘Dense’ refers to a fully-connected layer
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

with Live(output_dir) as live:
    hist = model.fit(X_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[DVCLiveCallback(live=live,save_dvc_exp=True)])
    #model.load_weights(os.path.join("model", "best_model"))
    # Log additional data after training
    test_loss, test_acc = model.evaluate(X_test,y_test)
    live.summary["test_loss"] = test_loss
    live.summary["test_acc"] = test_acc



plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig(output_dir + '/plot_train_val_loss.png')


# Save the model
joblib.dump(model, output_dir + '/HouserPriceModel.pkl')