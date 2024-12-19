from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from datetime import datetime

class MonModelClassif:
    def __init__(self, img_dim, nb_units, activation, nb_classe):
        """ initialise l'instance de la classe:
            1 - les attributs
            2 - Création du réseau de neurones
            3 - Compilation du réseau """
        # attributs de ma classe
        self.img_dim = img_dim # taille en x et y (size_x, size_y)
        self.nb_units = nb_units
        self.activation = activation
        self.nb_classe = nb_classe
        # Création du réseau de neurones
        self.MonReseau_creation()
        # Compilation du réseau
        self.MonReseau_compile()

    def MonReseau_creation(self):
        model = Sequential()
        # INPUT LAYER
        model.add(Input(shape=self.img_dim))
        # HIDDEN LAYERS
        model.add(Flatten())
        model.add(Dense(units=self.nb_units, activation=self.activation))
        model.add(Dropout(0.5))
        model.add(Dense(units=64, activation=self.activation))
        model.add(Dropout(0.5))
        model.add(Dense(units=32, activation=self.activation))
        model.add(Dropout(0.5))
        # OUTPUT LAYER
        model.add(Dense(units=self.nb_classe, activation="softmax"))
        self.model = model

    def MonReseau_compile(self):
        self.model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

    def train_model(self, x_train, y_train, x_test, y_test, batch_size=128, epochs=50, verbose=True):
        self.history = self.model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose)
        
    def predire_proba(self, data):
        return self.model.predict(data)
    
    def predire(self, data):
        return np.argmax(self.model.predict(data), axis=-1)
    
    def plot_loss(self):
        plt.figure()
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.grid()
        plt.legend()

    def plot_metrics(self):
        plt.figure()
        plt.plot(self.history.history["accuracy"], label="tra_accuracy")
        plt.plot(self.history.history["val_accuracy"], label="val_accuracy")
        plt.grid()
        plt.legend()

# Chargement des données
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Aplatir les images de 28x28 en vecteurs de 784 éléments
x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28).astype('float32') / 255

# Création d'un objet, qui est une instance de la classe MonModelClassif
model = MonModelClassif((28, 28), 128, "relu", 10)

# Entraînement du modèle
model.train_model(x_train, y_train, x_test, y_test, verbose=True)

# Prédictions
ypred_classe = model.predire(x_test)

# Affichage du rapport de classification
report = classification_report(y_test, ypred_classe)
print(report)

# Enregistrement du rapport dans un fichier texte avec la date dans le nom
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"../rapports/classification_report_classkeras_{date_str}.txt"
with open(report_filename, "w") as f:
    f.write(report)

# Affichage des courbes de perte et de précision
model.plot_loss()
model.plot_metrics()
plt.show()