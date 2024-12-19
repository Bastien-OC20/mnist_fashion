from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np 
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
import time
from functools import wraps

# Configurer l'URI de suivi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fashion_MNIST_Models_sl")

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

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalisation des données
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 28*28)).reshape(-1, 28, 28)
x_test = scaler.transform(x_test.reshape(-1, 28*28)).reshape(-1, 28, 28)

x2_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x2_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

@timer
def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    return model

@timer
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

with mlflow.start_run():
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    
    model_rf = train_random_forest(x2_train, y_train)
    report_rf = evaluate_model(model_rf, x2_test, y_test)
    
    # Enregistrer les métriques finales
    mlflow.log_metric("final_accuracy", report_rf["accuracy"])
    mlflow.log_metric("final_precision", report_rf["weighted avg"]["precision"])
    mlflow.log_metric("final_recall", report_rf["weighted avg"]["recall"])
    mlflow.log_metric("final_f1_score", report_rf["weighted avg"]["f1-score"])
    
    # Enregistrer le modèle
    mlflow.sklearn.log_model(model_rf, "model_rf")
    
    # Enregistrer le rapport de classification
    report_filename = f"../rapports/classification_report_rf_{date_str}.txt"
    with open(report_filename, "w") as f:
        f.write(classification_report(y_test, model_rf.predict(x2_test)))
    mlflow.log_artifact(report_filename)