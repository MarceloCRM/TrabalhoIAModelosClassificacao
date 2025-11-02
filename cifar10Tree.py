import keras
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

(xTreino, yTreino), (xTeste, yTeste) = keras.datasets.cifar10.load_data()


xTreino_reshaped = xTreino.reshape((xTreino.shape[0], -1))
xTeste_reshaped = xTeste.reshape((xTeste.shape[0], -1))

xTreino_normalized = xTreino_reshaped / 255.0
xTeste_normalized = xTeste_reshaped / 255.0

yTreino_flat = yTreino.flatten()
yTeste_flat = yTeste.flatten()
clf2 = DecisionTreeClassifier()
clf2.fit(xTreino_normalized, yTreino_flat)
cifar10_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
esperado2 = yTeste_flat
previsto2 = clf2.predict(xTeste_normalized)
print(classification_report(esperado2, previsto2, target_names=cifar10_names))