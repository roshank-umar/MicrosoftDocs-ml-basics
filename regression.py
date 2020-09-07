import pandas as pd
bike_data = pd.read_csv('data/daily-bike-share.csv')
bike_data.head()

bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
bike_data.head(32)


numeric_features = ['temp','atemp','hum','windspeed']
bike_data[numeric_features + ['rentals']].describe()

from matplotlib import pyplot as plt
label = bike_data['rentals']
fig = plt.figure(figsize=(9,12))

a=fig.add_subplot(2,1,1)
label.plot.hist(color='lightblue',bins=100)

plt.axvline(label.mean(),color='magenta',linestyle='dashed',linewidth=2)
plt.axvline(label.median(),color='green',linestyle='dashed',linewidth=2)
a.set_title('Histogram')

a=fig.add_subplot(2,1,2)
label.plot(kind='box',vert=False)
a.set_title('Boxplot')




for col in numeric_features : 
    fig=plt.figure(figsize=(9,6))
    ax=fig.gca()
    feature=bike_data[col]
    feature.hist(color='lightblue',bins=100,ax=ax)
    ax.axvline(feature.mean(),color='magenta',linestyle='dashed',linewidth=2)
    ax.axvline(feature.median(),color='green',linestyle='dashed',linewidth=2)
    ax.set_title(col)
    

import numpy as np
categorical_features=['season','mnth','holiday','weekday','workingday','weathersit','day']
for col in categorical_features:
    counts=bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='lightblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    label = bike_data['rentals']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label, color='lightblue')
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
plt.show()


for col in categorical_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    bike_data.boxplot(column = 'rentals', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Bike Rentals")
plt.show()


X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 1000
np.set_printoptions(suppress=True)
print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.size, X_test.size))
# Train the model
from sklearn.linear_model import LinearRegression

# Fit a linear regression model on the training set
model = LinearRegression(normalize=False).fit(X_train, y_train)
print (model)


import numpy as np

predictions = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])


import matplotlib.pyplot as plt



plt.scatter(y_test, predictions, color="lightblue")
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test))
plt.show()