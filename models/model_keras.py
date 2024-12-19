from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Aplatir les images de 28x28 en vecteurs de 784 éléments
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

model = Sequential()
# input + 1ere couche cachée
model.add(Dense(units=128, activation="relu", input_dim=784))
model.add(Dropout(0.5))
# input + 2eme couche cachée
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.5))
# input + 3eme couche cachée
model.add(Dense(units=32, activation="relu"))

# classifieur on cherche 10 classes 
model.add(Dense(units=10, activation="softmax"))

model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=128,
                    epochs=50)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.grid()
plt.legend()

plt.figure()
plt.plot(history.history["accuracy"], label="tra_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.grid()
plt.legend()
plt.show()

y_pred = model.predict(x_test)

y_pred.shape

number = 0
plt.plot(y_pred[number])
plt.title(y_test[number])

argmax_y_pred = np.argmax(y_pred, axis=-1)
argmax_y_pred

print("Reg Log")
print("rnn")
report = classification_report(y_test, argmax_y_pred)
print(report)

# Enregistrer le rapport dans un fichier texte avec la date dans le nom
from datetime import datetime
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"../rapports/classification_report_keras_{date_str}.txt"
with open(report_filename, "w") as f:
    f.write(report)