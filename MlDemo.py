
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
import seaborn as sns

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.ml.feature as ft

"""
用pyspark做机器学习
主要参考：
https://www.jianshu.com/p/4d7003182398

https://www.jianshu.com/p/20456b512fa7

https://blog.csdn.net/cymy001/article/details/78483723?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

https://blog.csdn.net/qq_40587575/article/details/91170554

"""


spark = SparkSession.builder.appName("pyspark_test").enableHiveSupport().getOrCreate()

## 读取数据，既可以读取本地的csv等数据，也可以读取HDFS上的Hive表

spark_df = spark.read.csv('./data/diamonds.csv',inferSchema=True,header=True)

spark_df.show()

"""
# 还可以使用pandas方式读取本地csv，转换pandas dataframe为spark dataframe。

import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
pandas_df = pd.read_csv('./data/diamonds.csv')
sc = SparkContext()
sqlContest = SQLContext(sc)
df = sqlContest.createDataFrame(pandas_df)

# hive数据库读取
spark.sql('select * from XX')

"""

##  常用操作


# **基础描述**
spark_df.count()  # 行数
spark_df.columns  # 列名称
spark_df.printSchema()  # 结构及属性显示
spark_df.show(5,False)  # truncate=False表示不压缩显示
spark_df.describe().show()  # 均值/最值等描述

# **dataframe操作**
# 取'age','mobile'两列
spark_df.select("carat","color").show(5)
# 新增一列：x+y+z
spark_df.withColumn("x_y_z",(spark_df["x"]+spark_df["y"]+spark_df["z"])).show(10,False)
# 新建一列x_double，将x转换为double属性
spark_df.withColumn("x_double",spark_df["x"].cast(DoubleType())).show(10,False)
# 筛选
spark_df.filter(spark_df["depth"]>=60).show()
spark_df.filter(spark_df["depth"]>=spark_df["table"]).select('x','y','depth','table').show()
spark_df.filter(spark_df['depth']<58).filter(spark_df['cut'] =="Ideal").show()
# 去重统计
spark_df.select('cut').distinct().show()
# 行去重
spark_df=spark_df.dropDuplicates()
# 删除列
df_new=spark_df.drop('x_y_z')
# groupby操作
spark_df.groupBy('color').count().show(5,False)
spark_df.groupBy('cut').count().orderBy('count',ascending=False).show(5,False)
spark_df.groupBy('color').agg({'price':'sum'}).show(5,False)  # 根据color分区，计算price的sum
# udf 自建sql函数

from pyspark.sql.functions import udf

def label_transform(cut):
    if cut in ["Good","Very Good","Premium"]:
        return "High"
    else:
        return "Low"


myudf=udf(label_transform,StringType())  # create udf using python function # 输出为string格式
# 新建一列price_range
spark_df = spark_df.withColumn("label",myudf(spark_df['cut']))
# 使用lamba创建udf
price_udf = udf(lambda price: "high_price" if price >= 330 else "low_price", StringType())  # using lambda function
# 新建一列age_group
spark_df = spark_df.withColumn("price_group", price_udf(spark_df["price"]))



spark_df.show()


























