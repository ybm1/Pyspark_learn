
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext, HiveContext
from pyspark import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F

import numpy as np
import pandas as pd
import time


from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.ml.feature as ft

spark = SparkSession.builder.appName("pyspark_test").enableHiveSupport().getOrCreate()


sentenceData = spark.createDataFrame([
 (0.0, "I like Spark",2),
 (1.0, "Pandas is useful",3),
 (2.0, "They are coded by Python",1)
], ["label", "sentence","a"])

# print(type(sentenceData))
#

print(sentenceData.agg(F.max("a")).toPandas().iloc[0,0])
#print(sentenceData.show())

pandas_df = sentenceData.toPandas()
#print(pandas_df)

d = sentenceData.rdd
d = d.map(lambda row: (row[0],row[2]))

d = d.toDF(["a","b"])
print(d.show())








from pyspark.ml.stat import *


spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = spark.createDataFrame(data, ["label", "features"])

r = ChiSquareTest.test(df, "features", "label").head()
# print("pValues: " + str(r.pValues))
# print("degreesOfFreedom: " + str(r.degreesOfFreedom))
# print("statistics: " + str(r.statistics))
