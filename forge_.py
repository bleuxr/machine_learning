#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

X,y=mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:,0],X[:,1],y)

plt.legend(["Class 0","Class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:{}".format(X.shape))

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

X,y=mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")

#%%
import numpy as np
import matplotlib
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
print("Shape of cancer data:{}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n:v for n, v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))

#%%
import numpy as np
import matplotlib
import pandas as pd
import mglearn
from sklearn.datasets import load_boston

boston=load_boston()
print("Data shape:{}".format(boston.data.shape))

X,y=mglearn.datasets.load_extended_boston()
print("X.shape:{}".format(X.shape))

#%%
import numpy as np
import matplotlib
import pandas as pd
import mglearn

#mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

#%%
import numpy as np
import matplotlib
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X,y=mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("Test set predictions:{}".format(clf.predict(X_test)))
print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))

#%%
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#X,y=mglearn.datasets.make_forge()

fig,axes=plt.subplots(1,3,figsize=(10,3))

for n_neighbors, ax in zip([1,3,9],axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)

    ax.set_title("{} neighbors(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    #plt.xlabel("feature 0")
    #plt.ylabel("feature 1")

axes[0].legend(loc=3)

#%%
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

cancer =load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)

training_accuracy=[]
test_accuracy=[]

neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)

    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#%%
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)

#%%
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

X,y=mglearn.datasets.make_wave(n_samples=40)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
print("Test set prediction:\n{}".format(reg.predict(X_test)))
print("Test set R^2:{:.2f}".format(reg.score(X_test,y_test)))


fig, axes=plt.subplots(1,3,figsize=(15,4))

line=np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)

    ax.set_title("{}neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors,reg.score(X_train,y_train),reg.score(X_test,y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions","Training data/target","Test data/target"],loc="best")
