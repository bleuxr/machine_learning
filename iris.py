#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:193]+"\n...")
print("Target names:{}".format(iris_dataset['target_names']))
print("Feature names:{}".format(iris_dataset['feature_names']))
print("Type of data:{}".format(type(iris_dataset['data'])))
print("Shape of data:{}".format(iris_dataset['data'].shape))

print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target:{}".format(type(iris_dataset['target'])))
print("Shape of target:{}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

iris_dataset = load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
#print("X_train:\n{0}\nX_test:\n{1}".format(X_train,X_test))
print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))

print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr=pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='0',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

X_new=np.array([[5,2.9,1,0.2]])
print("X_new.shape:{}".format(X_new.shape))

prediction=knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))