#%%
# Setup

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#%%
# Load data

iris = load_iris()
X, Y = iris.data, iris.target

#%%
# Test DTC classifier (Vanilla)

dtc = DecisionTreeClassifier()
dtc.fit(X, Y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    dtc,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)


#%%
# Test DTC classifier (MD 3)

dtc = DecisionTreeClassifier(
    max_depth=3
)
dtc.fit(X, Y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    dtc,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)


#%%
# Test DTC classifier (MSL 4)

dtc = DecisionTreeClassifier(
    min_samples_leaf=4
)
dtc.fit(X, Y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    dtc,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

#%%
# Test DTC classifier (MD 3 / MSL 4)

dtc = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=4
)
dtc.fit(X, Y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    dtc,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

# %%
