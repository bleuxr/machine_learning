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