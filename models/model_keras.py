from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import mlflow
import mlflow.keras
from datetime import datetime

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

with mlflow.start_run():
    mlflow.log_param("model_type", "Dense")
    mlflow.log_param("units_layer1", 128)
    mlflow.log_param("units_layer2", 64)
    mlflow.log_param("units_layer3", 32)
    mlflow.log_param("dropout_rate", 0.5)
    mlflow.log_param("optimizer", "adam")
    
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        epochs=50)
    
    # Enregistrer les métriques
    for epoch in range(50):
        mlflow.log_metric("loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("accuracy", history.history["accuracy"][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)
    
    y_pred = model.predict(x_test)
    argmax_y_pred = np.argmax(y_pred, axis=-1)
    report = classification_report(y_test, argmax_y_pred, output_dict=True)
    
    # Enregistrer les métriques finales
    mlflow.log_metric("final_accuracy", report["accuracy"])
    mlflow.log_metric("final_precision", report["weighted avg"]["precision"])
    mlflow.log_metric("final_recall", report["weighted avg"]["recall"])
    mlflow.log_metric("final_f1_score", report["weighted avg"]["f1-score"])
    
    # Enregistrer le modèle
    mlflow.keras.log_model(model, "model")
    
    # Enregistrer les artefacts
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_plot.png")
    mlflow.log_artifact("loss_plot.png")
    
    plt.figure()
    plt.plot(history.history["accuracy"], label="tra_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.grid()
    plt.legend()
    plt.savefig("accuracy_plot.png")
    mlflow.log_artifact("accuracy_plot.png")
    
    # Enregistrer le rapport de classification
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"../rapports/classification_report_keras_{date_str}.txt"
    with open(report_filename, "w") as f:
        f.write(classification_report(y_test, argmax_y_pred))
    mlflow.log_artifact(report_filename)