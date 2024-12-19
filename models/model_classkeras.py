from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
import mlflow
import mlflow.keras
from datetime import datetime
import time
from functools import wraps
import tempfile
import os

# Configurer l'URI de suivi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fashion_MNIST_Models_classKeras")

# Décorateur pour mesurer le temps d'exécution
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = time.perf_counter()
        resultat = func(*args, **kwargs)
        fin = time.perf_counter()
        duree = fin - debut
        print(f"{func.__name__} a pris {duree:.4f} secondes")
        return resultat
    return wrapper

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

    @timer
    def train_model(self, x_train, y_train, x_test, y_test, batch_size=128, epochs=50, verbose=True):
        with mlflow.start_run():
            mlflow.log_param("model_type", "Dense")
            mlflow.log_param("units_layer1", self.nb_units)
            mlflow.log_param("units_layer2", 64)
            mlflow.log_param("units_layer3", 32)
            mlflow.log_param("dropout_rate", 0.5)
            mlflow.log_param("optimizer", "adam")
            
            self.history = self.model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose)
            
            # Enregistrer les métriques
            for epoch in range(epochs):
                mlflow.log_metric("loss", self.history.history["loss"][epoch], step=epoch)
                mlflow.log_metric("val_loss", self.history.history["val_loss"][epoch], step=epoch)
                mlflow.log_metric("accuracy", self.history.history["accuracy"][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", self.history.history["val_accuracy"][epoch], step=epoch)
            
            y_pred = self.model.predict(x_test)
            argmax_y_pred = np.argmax(y_pred, axis=-1)
            report = classification_report(y_test, argmax_y_pred, output_dict=True)
            
            # Enregistrer les métriques finales
            mlflow.log_metric("final_accuracy", report["accuracy"])
            mlflow.log_metric("final_precision", report["weighted avg"]["precision"])
            mlflow.log_metric("final_recall", report["weighted avg"]["recall"])
            mlflow.log_metric("final_f1_score", report["weighted avg"]["f1-score"])
            
            # Enregistrer le modèle avec un exemple d'entrée
            input_example = np.expand_dims(x_test[0], axis=0)
            with tempfile.TemporaryDirectory() as temp_dir:
                input_example_path = os.path.join(temp_dir, "input_classkeras.json")
                np.save(input_example_path, input_example)
                mlflow.keras.log_model(self.model, "model_classkeras", input_example=input_example)
            
            # Enregistrer les artefacts
            self.plot_loss()
            plt.savefig("loss_plot.png")
            mlflow.log_artifact("loss_plot.png")
            
            self.plot_metrics()
            plt.savefig("accuracy_plot.png")
            mlflow.log_artifact("accuracy_plot.png")
            
            # Enregistrer le rapport de classification
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"../rapports/classification_report_keras_{date_str}.txt"
            with open(report_filename, "w") as f:
                f.write(classification_report(y_test, argmax_y_pred))
            mlflow.log_artifact(report_filename)
        
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
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

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