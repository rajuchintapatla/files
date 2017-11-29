from pandas import read_csv
from sklearn import linear_model
import matplotlib.pyplot as plt

names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv('data.csv', names=names)
array = dataframe.values
print(array.shape)
X = array[:, 0:1]
Y = array[:, 8:9]
print(Y)
regr = linear_model.LinearRegression()
regr.fit(X, Y)
plt.scatter(X, Y, color='black')
print(regr.predict(10))
plt.plot(X, regr.predict(X))
plt.show()