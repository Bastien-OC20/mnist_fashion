from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np 
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print("train")
# print("x", x_train.shape)
# print("y", y_train.shape)

# print("test")
# print("x", x_test.shape)
# print("y", y_test.shape)

""""
sortie du dataset
train
x (60000, 28, 28)
y (60000,)
test
x (10000, 28, 28)
y (10000,)
"""""

# number = 10 

# plt.imshow(x_train[number,:,:], cmap="gray_r")
# plt.title(y_train[number])
# plt.colorbar()
# plt.axis("off")
# plt.show()

"""""
sortie de l'image
./png/figure_1.png
"""""

# Normalisation des données !!!!!! 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 28*28)).reshape(-1, 28, 28)
x_test = scaler.transform(x_test.reshape(-1, 28*28)).reshape(-1, 28, 28)


# number = 10
# plt.imshow(x_train[number,:,:], cmap="gray_r")
# plt.title(y_train[number])
# plt.colorbar()
# plt.axis("off")
# plt.show()

"""""
sortie de l'image
./png/figure_normalised.png
"""""

# plt.hist(y_train, bins=19)
# plt.show()

"""""
sortie de l'image
./png/hist_y_train.png
"""""

x2_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x2_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Modèle Logistic Regression
# model_lr = LogisticRegression(max_iter=500)  # Augmenter max_iter
# model_lr.fit(x2_train, y_train)
# y_pred_lr = model_lr.predict(x2_test)
# report_lr = classification_report(y_test, y_pred_lr)
# print(report_lr)

# # Sauvegarder le rapport Logistic Regression dans un fichier texte avec la date dans le nom
# report_filename_lr = f"../rapports/classification_report_lr_{date_str}.txt"
# with open(report_filename_lr, "w") as f:
#     f.write(report_lr)

# Modèle Random Forest
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(x2_train, y_train)
y_pred_rf = model_rf.predict(x2_test)
report_rf = classification_report(y_test, y_pred_rf)
print(report_rf)

# Sauvegarder le rapport Random Forest dans un fichier texte avec la date dans le nom
report_filename_rf = f"../rapports/classification_report_rf_{date_str}.txt"
with open(report_filename_rf, "w") as f:
    f.write(report_rf)

# # Modèle SVM
# scaler_svm = StandardScaler()  # Ajouter le scaling des données pour SVM
# x2_train_scaled = scaler_svm.fit_transform(x2_train)
# x2_test_scaled = scaler_svm.transform(x2_test)

# model_svm = SVC()
# model_svm.fit(x2_train_scaled, y_train)
# y_pred_svm = model_svm.predict(x2_test_scaled)
# report_svm = classification_report(y_test, y_pred_svm)
# print(report_svm)

# # Sauvegarder le rapport SVM dans un fichier texte avec la date dans le nom
# report_filename_svm = f"../rapports/classification_report_svm_{date_str}.txt"
# with open(report_filename_svm, "w") as f:
#     f.write(report_svm)