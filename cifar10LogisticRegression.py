import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

(xTreino, yTreino), (xTeste, yTeste) = keras.datasets.cifar10.load_data()

xTreino = xTreino.reshape((xTreino.shape[0], -1)) / 255.0
xTeste = xTeste.reshape((xTeste.shape[0], -1)) / 255.0

yTreino = yTreino.flatten()
yTeste = yTeste.flatten()

clf = LogisticRegression(max_iter=100, solver='saga', n_jobs=-1)
clf.fit(xTreino, yTreino)

yPrev = clf.predict(xTeste)

cifar10_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(classification_report(yTeste, yPrev, target_names=cifar10_names))
