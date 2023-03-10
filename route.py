# -*- coding: utf-8 -*-
"""Route

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bs1zNru7udgYnXQftsWyu4U4ebApKWRg
"""

!pip install geopandas

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

shapefile = gpd.read_file("/content/drive/MyDrive/Datasets/routes/bluetooth_routes_wgs84.shp")

print(shapefile)

#shapefile.plot(column='normalDriv', figsize=(16,8))
shapefile.plot()
shapefile.info()

travel = pd.read_csv('/content/drive/MyDrive/Datasets/travel-time-2014.csv')
travel

# column merged from csv
reshape = shapefile.merge(travel, on='resultId') 
reshape.head()

from google.colab import files
reshape.to_csv('filename.csv') 
files.download('filename.csv')

file1 = pd.read_csv('/content/filename.csv')
file1

#travel.dtypes

#travel.set_index('updated', inplace=True)
#travel.head()

#travel.updated = pd.to_datetime(travel.updated)

#travel.updated= travel['updated'].astype('datetime64[ns]')

#travel.dtypes

#travel('updated').tz_convert(None)
#travel.head()

#travel.updated = pd.to_datetime(travel.updated)
#travel.groupby(pd.Grouper(key="updated", freq="1H")).mean()
#travel.head()

#travel.set_index(['updated'],inplace=True)
#travel['count'].resample('MS' , on = 'updated').mean()
#travel.head()



CI = (reshape['timeInSeconds'] - reshape['normalDriv'])/(reshape['normalDriv'])
reshape['CI'] = CI
reshape.head()

conditions = [
    
(reshape['CI'] < 0),
(reshape['CI'] >= 0) & (reshape['CI'] < 0.15) ,
(reshape['CI'] >= 0.15) & (reshape['CI'] < 0.85),
    (reshape['CI'] >= 0.85),
]

predictions = ['smooth','smooth', 'congestion', 'blockage']



reshape['predictions'] = np.select(conditions, predictions)
reshape.head(5)

reshape['Street_Id'] = pd.factorize(reshape['resultId'])[0]

#reshape = reshape.loc[:50,:]
reshape

from google.colab import files
reshape.to_csv('data.csv') 
files.download('data.csv')

reshape1 = reshape.sort_values(by='updated', ascending=True)
reshape1

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

y = pd.factorize(reshape['predictions'])[0]

# View target
y

# Grab our X & Y Columns.
X_Cols = reshape[['Street_Id', 'normalDriv', 'length_m',	'count']]
Y_Cols = y

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

# Create a Random Forest Classifier
rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

# Fit the data to the model
rand_frst_clf.fit(X_train, y_train)

# Make predictions
y_pred = rand_frst_clf.predict(X_test)
#preds = reshape.predictions[(y_pred)]
#preds [0:50]

reshape.predictions

y_test[:50]

print('Correct Prediction (%): ', accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0)

# Create an ROC Curve plot.
rfc_disp = plot_roc_curve(rand_frst_clf, X_test, y_test, alpha = 0.8)
plt.show()

x_ax = range(len(y_test[:50]))
plt.plot(x_ax, y_test[:50], 'o-', linewidth=1, label="original")
plt.plot(x_ax, y_pred[:50], 'o-', linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Congestion Level')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(axis = 'y')
plt.yticks([0,1,2])
#plt.plot(y_test[:50],'o-')
#plt.plot(y_pred[:50],'o-')
plt.show()



# Calculate feature importance and store in pandas series
feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)
feature_imp

# store the values in a list to plot.
x_values = list(range(len(rand_frst_clf.feature_importances_)))

# Cumulative importances
cumulative_importances = np.cumsum(feature_imp.values)

# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')

# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')

# Format x ticks and labels
plt.xticks(x_values, feature_imp.index, rotation = 'vertical')

# Axis labels and title
plt.xlabel('Variable')
plt.ylabel('Cumulative Importance')
plt.title('Random Forest: Feature Importance Graph')

# Create confusion matrix
print('Confusion matrix')

pd.crosstab(y_test[:50], y_pred[:50], rownames=['Actual congestion'], colnames=['Predicted congestion'])

cf_matrix = confusion_matrix(y_test,y_pred)
import seaborn as sns
print('Confusion matrix:\n',cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.show()

cf_matrix = confusion_matrix(y_test[:50],y_pred[:50])
import seaborn as sns
print('Confusion matrix:\n',cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.show()