# Deep Learnin' in the Mornin'

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

    Using TensorFlow backend.


In the cell below we import the Sklearn breast cancer dataset.


```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()
X = data['data']
y = data['target']
df = pd.DataFrame(X)
df.columns = data['feature_names']
df['target'] = y
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.0</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.5</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



Okok. 

<u><b>In the cell below</b></u>
- Create a train test split of the breast cancer dataset
    - Set the random state to `2020`.


```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), 
                                                    df.target, random_state=2020)
```

Ok, now that our data is split, let's define a function that will compile a multi layer perceptron. 


```python
def compile_model():
    model = Sequential() 
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

    Epoch 1/150
    426/426 [==============================] - 0s 352us/step - loss: 33.1645 - accuracy: 0.6479
    Epoch 2/150
    426/426 [==============================] - 0s 45us/step - loss: 6.1891 - accuracy: 0.5516
    Epoch 3/150
    426/426 [==============================] - 0s 47us/step - loss: 2.5098 - accuracy: 0.5751
    Epoch 4/150
    426/426 [==============================] - 0s 41us/step - loss: 2.0494 - accuracy: 0.6549
    Epoch 5/150
    426/426 [==============================] - 0s 52us/step - loss: 1.7925 - accuracy: 0.6854
    Epoch 6/150
    426/426 [==============================] - 0s 45us/step - loss: 1.6425 - accuracy: 0.7207
    Epoch 7/150
    426/426 [==============================] - 0s 46us/step - loss: 1.4539 - accuracy: 0.7582
    Epoch 8/150
    426/426 [==============================] - 0s 41us/step - loss: 1.3679 - accuracy: 0.7770
    Epoch 9/150
    426/426 [==============================] - 0s 43us/step - loss: 1.2976 - accuracy: 0.7606
    Epoch 10/150
    426/426 [==============================] - 0s 42us/step - loss: 1.1766 - accuracy: 0.8146
    Epoch 11/150
    426/426 [==============================] - 0s 48us/step - loss: 0.8902 - accuracy: 0.8239
    Epoch 12/150
    426/426 [==============================] - 0s 42us/step - loss: 0.4820 - accuracy: 0.8850
    Epoch 13/150
    426/426 [==============================] - 0s 44us/step - loss: 0.3291 - accuracy: 0.8920
    Epoch 14/150
    426/426 [==============================] - 0s 43us/step - loss: 0.3570 - accuracy: 0.9155
    Epoch 15/150
    426/426 [==============================] - 0s 44us/step - loss: 0.3009 - accuracy: 0.9178
    Epoch 16/150
    426/426 [==============================] - 0s 46us/step - loss: 0.3010 - accuracy: 0.9155
    Epoch 17/150
    426/426 [==============================] - 0s 44us/step - loss: 0.2863 - accuracy: 0.9225
    Epoch 18/150
    426/426 [==============================] - 0s 40us/step - loss: 0.2585 - accuracy: 0.9249
    Epoch 19/150
    426/426 [==============================] - 0s 43us/step - loss: 0.2376 - accuracy: 0.9343
    Epoch 20/150
    426/426 [==============================] - 0s 47us/step - loss: 0.3121 - accuracy: 0.8991
    Epoch 21/150
    426/426 [==============================] - 0s 54us/step - loss: 0.3856 - accuracy: 0.9108
    Epoch 22/150
    426/426 [==============================] - 0s 42us/step - loss: 0.2361 - accuracy: 0.9296
    Epoch 23/150
    426/426 [==============================] - 0s 44us/step - loss: 0.4002 - accuracy: 0.8920
    Epoch 24/150
    426/426 [==============================] - 0s 46us/step - loss: 0.2798 - accuracy: 0.9155
    Epoch 25/150
    426/426 [==============================] - 0s 43us/step - loss: 0.2382 - accuracy: 0.9225
    Epoch 26/150
    426/426 [==============================] - 0s 42us/step - loss: 0.2596 - accuracy: 0.9202
    Epoch 27/150
    426/426 [==============================] - 0s 44us/step - loss: 0.2779 - accuracy: 0.9178
    Epoch 28/150
    426/426 [==============================] - 0s 45us/step - loss: 0.5106 - accuracy: 0.8709
    Epoch 29/150
    426/426 [==============================] - 0s 46us/step - loss: 0.3652 - accuracy: 0.8944
    Epoch 30/150
    426/426 [==============================] - 0s 43us/step - loss: 0.2534 - accuracy: 0.9202
    Epoch 31/150
    426/426 [==============================] - 0s 61us/step - loss: 0.2771 - accuracy: 0.9202
    Epoch 32/150
    426/426 [==============================] - 0s 55us/step - loss: 0.2774 - accuracy: 0.9225
    Epoch 33/150
    426/426 [==============================] - 0s 48us/step - loss: 0.2076 - accuracy: 0.9319
    Epoch 34/150
    426/426 [==============================] - 0s 46us/step - loss: 0.2046 - accuracy: 0.9343
    Epoch 35/150
    426/426 [==============================] - 0s 44us/step - loss: 0.2125 - accuracy: 0.9319
    Epoch 36/150
    426/426 [==============================] - 0s 47us/step - loss: 0.2045 - accuracy: 0.9366
    Epoch 37/150
    426/426 [==============================] - 0s 51us/step - loss: 0.1951 - accuracy: 0.9343
    Epoch 38/150
    426/426 [==============================] - 0s 44us/step - loss: 0.1887 - accuracy: 0.9413
    Epoch 39/150
    426/426 [==============================] - 0s 56us/step - loss: 0.2420 - accuracy: 0.9131
    Epoch 40/150
    426/426 [==============================] - 0s 43us/step - loss: 0.1861 - accuracy: 0.9366
    Epoch 41/150
    426/426 [==============================] - 0s 45us/step - loss: 0.1958 - accuracy: 0.9272
    Epoch 42/150
    426/426 [==============================] - 0s 44us/step - loss: 0.3158 - accuracy: 0.9038
    Epoch 43/150
    426/426 [==============================] - 0s 44us/step - loss: 0.3247 - accuracy: 0.8920
    Epoch 44/150
    426/426 [==============================] - 0s 43us/step - loss: 0.1946 - accuracy: 0.9390
    Epoch 45/150
    426/426 [==============================] - 0s 46us/step - loss: 0.2151 - accuracy: 0.9296
    Epoch 46/150
    426/426 [==============================] - 0s 44us/step - loss: 0.2008 - accuracy: 0.9272
    Epoch 47/150
    426/426 [==============================] - 0s 44us/step - loss: 0.1950 - accuracy: 0.9272
    Epoch 48/150
    426/426 [==============================] - 0s 44us/step - loss: 0.1712 - accuracy: 0.9437
    Epoch 49/150
    426/426 [==============================] - 0s 50us/step - loss: 0.1639 - accuracy: 0.9413
    Epoch 50/150
    426/426 [==============================] - 0s 49us/step - loss: 0.1742 - accuracy: 0.9484
    Epoch 51/150
    426/426 [==============================] - 0s 52us/step - loss: 0.1568 - accuracy: 0.9484
    Epoch 52/150
    426/426 [==============================] - 0s 53us/step - loss: 0.1517 - accuracy: 0.9366
    Epoch 53/150
    426/426 [==============================] - 0s 45us/step - loss: 0.1768 - accuracy: 0.9413
    Epoch 54/150
    426/426 [==============================] - 0s 53us/step - loss: 0.1688 - accuracy: 0.9319
    Epoch 55/150
    426/426 [==============================] - 0s 52us/step - loss: 0.2657 - accuracy: 0.9178
    Epoch 56/150
    426/426 [==============================] - 0s 48us/step - loss: 0.1790 - accuracy: 0.9272
    Epoch 57/150
    426/426 [==============================] - 0s 52us/step - loss: 0.2516 - accuracy: 0.9061
    Epoch 58/150
    426/426 [==============================] - 0s 42us/step - loss: 0.1722 - accuracy: 0.9437
    Epoch 59/150
    426/426 [==============================] - 0s 43us/step - loss: 0.1933 - accuracy: 0.9390
    Epoch 60/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1492 - accuracy: 0.9484
    Epoch 61/150
    426/426 [==============================] - 0s 42us/step - loss: 0.2167 - accuracy: 0.9272
    Epoch 62/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1693 - accuracy: 0.9296
    Epoch 63/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1781 - accuracy: 0.9319
    Epoch 64/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1555 - accuracy: 0.9437
    Epoch 65/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1520 - accuracy: 0.9507
    Epoch 66/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1484 - accuracy: 0.9531
    Epoch 67/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1527 - accuracy: 0.9507
    Epoch 68/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1654 - accuracy: 0.9390
    Epoch 69/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1570 - accuracy: 0.9437
    Epoch 70/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1528 - accuracy: 0.9366
    Epoch 71/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1675 - accuracy: 0.9366
    Epoch 72/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1450 - accuracy: 0.9531
    Epoch 73/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1817 - accuracy: 0.9296
    Epoch 74/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1798 - accuracy: 0.9343
    Epoch 75/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1869 - accuracy: 0.9343
    Epoch 76/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1799 - accuracy: 0.9460
    Epoch 77/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1471 - accuracy: 0.9460
    Epoch 78/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1971 - accuracy: 0.9249
    Epoch 79/150
    426/426 [==============================] - 0s 39us/step - loss: 0.2440 - accuracy: 0.9155
    Epoch 80/150
    426/426 [==============================] - 0s 39us/step - loss: 0.2138 - accuracy: 0.9225
    Epoch 81/150
    426/426 [==============================] - 0s 41us/step - loss: 0.2080 - accuracy: 0.9249
    Epoch 82/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1745 - accuracy: 0.9319
    Epoch 83/150
    426/426 [==============================] - 0s 38us/step - loss: 0.2025 - accuracy: 0.9296
    Epoch 84/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1427 - accuracy: 0.9577
    Epoch 85/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1764 - accuracy: 0.9437
    Epoch 86/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1402 - accuracy: 0.9554
    Epoch 87/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1457 - accuracy: 0.9484
    Epoch 88/150
    426/426 [==============================] - 0s 43us/step - loss: 0.1441 - accuracy: 0.9460
    Epoch 89/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1791 - accuracy: 0.9343
    Epoch 90/150
    426/426 [==============================] - 0s 42us/step - loss: 0.1390 - accuracy: 0.9460
    Epoch 91/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1454 - accuracy: 0.9507
    Epoch 92/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1549 - accuracy: 0.9437
    Epoch 93/150
    426/426 [==============================] - 0s 42us/step - loss: 0.1689 - accuracy: 0.9343
    Epoch 94/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1605 - accuracy: 0.9437
    Epoch 95/150
    426/426 [==============================] - 0s 41us/step - loss: 0.1742 - accuracy: 0.9390
    Epoch 96/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1465 - accuracy: 0.9531
    Epoch 97/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1529 - accuracy: 0.9437
    Epoch 98/150
    426/426 [==============================] - 0s 41us/step - loss: 0.2221 - accuracy: 0.9202
    Epoch 99/150
    426/426 [==============================] - 0s 37us/step - loss: 0.1495 - accuracy: 0.9437
    Epoch 100/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1705 - accuracy: 0.9319
    Epoch 101/150
    426/426 [==============================] - 0s 41us/step - loss: 0.1410 - accuracy: 0.9390
    Epoch 102/150
    426/426 [==============================] - 0s 46us/step - loss: 0.1368 - accuracy: 0.9531
    Epoch 103/150
    426/426 [==============================] - 0s 37us/step - loss: 0.1351 - accuracy: 0.9554
    Epoch 104/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1498 - accuracy: 0.9437
    Epoch 105/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1559 - accuracy: 0.9390
    Epoch 106/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1988 - accuracy: 0.9178
    Epoch 107/150
    426/426 [==============================] - 0s 40us/step - loss: 0.2371 - accuracy: 0.9390
    Epoch 108/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1734 - accuracy: 0.9460
    Epoch 109/150
    426/426 [==============================] - 0s 41us/step - loss: 0.1649 - accuracy: 0.9319
    Epoch 110/150
    426/426 [==============================] - 0s 47us/step - loss: 0.1541 - accuracy: 0.9366
    Epoch 111/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1776 - accuracy: 0.9366
    Epoch 112/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1461 - accuracy: 0.9390
    Epoch 113/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1602 - accuracy: 0.9507
    Epoch 114/150
    426/426 [==============================] - 0s 41us/step - loss: 0.1346 - accuracy: 0.9531
    Epoch 115/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1594 - accuracy: 0.9296
    Epoch 116/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1737 - accuracy: 0.9390
    Epoch 117/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1425 - accuracy: 0.9390
    Epoch 118/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1596 - accuracy: 0.9390
    Epoch 119/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1657 - accuracy: 0.9390
    Epoch 120/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1653 - accuracy: 0.9413
    Epoch 121/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1329 - accuracy: 0.9531
    Epoch 122/150
    426/426 [==============================] - 0s 37us/step - loss: 0.1285 - accuracy: 0.9484
    Epoch 123/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1607 - accuracy: 0.9484
    Epoch 124/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1887 - accuracy: 0.9296
    Epoch 125/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1511 - accuracy: 0.9272
    Epoch 126/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1390 - accuracy: 0.9460
    Epoch 127/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1533 - accuracy: 0.9484
    Epoch 128/150
    426/426 [==============================] - 0s 41us/step - loss: 0.1481 - accuracy: 0.9413
    Epoch 129/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1436 - accuracy: 0.9507
    Epoch 130/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1830 - accuracy: 0.9413
    Epoch 131/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1461 - accuracy: 0.9507
    Epoch 132/150
    426/426 [==============================] - 0s 50us/step - loss: 0.1454 - accuracy: 0.9484
    Epoch 133/150
    426/426 [==============================] - 0s 46us/step - loss: 0.1321 - accuracy: 0.9577
    Epoch 134/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1915 - accuracy: 0.9085
    Epoch 135/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1700 - accuracy: 0.9390
    Epoch 136/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1659 - accuracy: 0.9413
    Epoch 137/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1588 - accuracy: 0.9296
    Epoch 138/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1471 - accuracy: 0.9413
    Epoch 139/150
    426/426 [==============================] - 0s 38us/step - loss: 0.2004 - accuracy: 0.9319
    Epoch 140/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1481 - accuracy: 0.9413
    Epoch 141/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1956 - accuracy: 0.9272
    Epoch 142/150
    426/426 [==============================] - 0s 41us/step - loss: 0.2014 - accuracy: 0.9343
    Epoch 143/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1562 - accuracy: 0.9413
    Epoch 144/150
    426/426 [==============================] - 0s 39us/step - loss: 0.2533 - accuracy: 0.9038
    Epoch 145/150
    426/426 [==============================] - 0s 41us/step - loss: 0.1867 - accuracy: 0.9319
    Epoch 146/150
    426/426 [==============================] - 0s 39us/step - loss: 0.1846 - accuracy: 0.9343
    Epoch 147/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1892 - accuracy: 0.9249
    Epoch 148/150
    426/426 [==============================] - 0s 38us/step - loss: 0.1394 - accuracy: 0.9390
    Epoch 149/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1300 - accuracy: 0.9484
    Epoch 150/150
    426/426 [==============================] - 0s 40us/step - loss: 0.1462 - accuracy: 0.9507





    <keras.callbacks.callbacks.History at 0x1a36a73890>



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




    array([0.78947371, 0.92982459, 0.94736844, 0.92982459, 0.94736844,
           0.98245615, 0.9649123 , 0.9649123 , 0.91228068, 0.98214287])




```python
print('Average Accuracy: %.2f' % (results.mean()*100))
```

    Average Accuracy: 93.51

