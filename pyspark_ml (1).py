#!/usr/bin/env python
# coding: utf-8

# In[28]:


pip install pyspark


# In[29]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-congestion').getOrCreate()

data = spark.read.csv('OneDrive/Desktop/Datasets/data.csv', header = True, inferSchema = True)
data.printSchema()


# In[30]:


df = data.drop(*['_c0', 'geometry'])
df.columns


# In[31]:


df.groupBy('predictions').count().show()
df.groupBy('resultId').count().show()


# In[32]:


import pandas as pd

from pyspark.ml.feature import VectorAssembler

numericCols = ['Street_Id', 'normalDriv', 'length_m', 'count']
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
df = assembler.transform(df)
df.show()


# In[33]:


from pyspark.ml.feature import StringIndexer

label_stringIdx = StringIndexer(inputCol = 'predictions', outputCol = 'labelIndex')
df = label_stringIdx.fit(df).transform(df)
df.show()


# In[34]:


train, test = df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# In[35]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'labelIndex')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('count', 'timeInSeconds', 'labelIndex', 'rawPrediction', 'prediction', 'probability').show(25)



# In[36]:


predictions.select("labelIndex", "prediction").show(50)


# In[37]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))


# In[43]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


lrmetrics = MulticlassMetrics(predictions['labelIndex','prediction'].rdd)
print('Confusion Matrix:\n', lrmetrics.confusionMatrix())


# In[52]:


x_ax = range(len(predictions.prediction[:50]))
plt.plot(x_ax, predictions.select('prediction')[:50], 'o-', linewidth=1, label="original")
plt.plot(x_ax, predicitons.select('labelIndex')[:50], 'o-', linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Congestion Level')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(axis = 'y')
plt.yticks([0,1,2])
#plt.plot(y_test[:50],'o-')
#plt.plot(y_pred[:50],'o-')
plt.show() 


# In[ ]:




