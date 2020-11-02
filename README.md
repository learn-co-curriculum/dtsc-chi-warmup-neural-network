# Deep Learnin' on a Week Day

This morning we will build a basic Deep Learning Model, often referred to as a "Multi Layer Perceptron". 

Once we've create this model, we will use an Sklearn wrapper to run cross validation on our deep learning model!

**Import libraries**


```python
# Dataset
from sklearn.datasets import load_breast_cancer

# Data Manipulation
import pandas as pd

# Turn of TensorFlow deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Modeling
from keras.models import Sequential 
from keras.layers import Dense 

# Model Validation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
```

In the cell below we import the Sklearn breast cancer dataset.


```python
data = load_breast_cancer()
X = data['data']
y = data['target']
df = pd.DataFrame(X)
df.columns = data['feature_names']
df['target'] = y
df.head(3)
```

Okok. 

<u><b>In the cell below</b></u>
- Create a train test split of the breast cancer dataset
    - Set the random state to `2020`.


```python
X_train, X_test, y_train, y_test = pass
```

Ok, now that our data is split, let's define a function that will compile a multi layer perceptron. 


```python
def compile_model():
    model = Sequential() 
    # The first parameter in a Keras Dense layer is the output shape after
    # the data has moved through that layer of the neural network. 
    # This parameter is required for every layer.
    # The input_dim parameter is only required for the first layer of a perceptron
    # and is the shape our of the training data. 
    model.add(Dense(14, input_dim=30, activation='relu')) 
    model.add(Dense(7, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
    # compile the keras model 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model
```

Now that we've done this, we can compile the model by calling the `compile_model` function, and then fit the model as we normally would when using deep learning models.


```python
model = compile_model()
model.fit(X_train, y_train, epochs=150, batch_size=23)
```

# Cross Validation

**With deep learning** it may not always be logistically possible to use cross validation. Depending on the size of the dataset, it can sometimes take multiple hours or even multiple days for a neural network to be fit on all of the training data.
 

That being said, if it is feasible to use cross validation, you should! It will allow you to have higher confidence in your model's ability to generalize to new data.

Keras provides a wrapper that allows us to easily use Sklearn's cross validation tools. 

<u><b>In the cell below, we import the Keras-Sklearn wrapper.</b></u>


```python
from keras.wrappers.scikit_learn import KerasClassifier
```

**You may have noticed** that the ``compile_model`` function above does not ``fit`` the model to data.  

*The key to using ``KerasClassifier`` is defining a function that returns an unfit, but compiled model.* 


```python
model = KerasClassifier(build_fn=compile_model, epochs=150, batch_size=23, verbose=0)
```

**Now** we can use the Keras model exactly how we would use a normal sklearn classifier.


```python
results = cross_val_score(model, X, y, cv=10)
```


```python
results
```


```python
print('Average Accuracy: %.2f' % (results.mean()*100))
```
