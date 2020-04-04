
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
from pyspark.ml.feature import MaxAbsScaler,StringIndexer, VectorIndexer,VectorAssembler
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
spark_df,_ =spark_df.randomSplit([0.3,0.7])



#spark_df.show()

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

"""

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

"""


# udf 自建sql函数

"""
在spark中很少会用for循环去处理一个个特征，一般使用函数/自建UDF，批量处理掉了。
比如计算Review列每个数据的长度。

python模式
review_length = []
for info in text_df['Review']:
···· review_length.apend(length(info))
text_df['length'] = review_length

pyspark模式

from pyspark.sql.functions import length 

text_df=text_df.withColumn('length',length(text_df['Review']))


"""
from pyspark.sql.functions import udf

def label_transform(cut):
    if cut in ["Good","Very Good"]:
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







## 特征处理

## pyspark.ml.feature提供的特征处理功能，满足了大部分机器学习的特征处理需求。


## 这里进行两个基本操作，把数值型的特征进行归一化，而对类别型特征进行label编码


# 归一化函数，将列标准化到[0,1]之间

num_cols =["carat","depth","table","price","x","y","z"]

cate_cols = ["color","clarity","cut","label","price_group"]

# # 用于将多个列合并为一个向量列，直接transform即可，经常用的
# numsAssembler = VectorAssembler(inputCols=num_cols, outputCol="num_features")
# spark_df = numsAssembler.transform(spark_df)

for num_col in num_cols:
    numsAssembler = VectorAssembler(inputCols=[num_col], outputCol=num_col+"_ass")
    spark_df = numsAssembler.transform(spark_df)
    maScaler = MaxAbsScaler(inputCol=num_col+"_ass", outputCol=num_col+"_scale")
    scalemodel = maScaler.fit(spark_df)
    spark_df = scalemodel.transform(spark_df)



# 针对单个类别型特征进行转换，把字符串的列按照出现频率进行排序
for cate_col in cate_cols:
    stringIndexer = StringIndexer(inputCol=cate_col,outputCol=cate_col+"_index")
    stringmodel = stringIndexer.fit(spark_df)
    spark_df = stringmodel.transform(spark_df)



num_cols_out = [x+"_scale" for x in num_cols]

cate_cols_out = [c+"_index" for c in cate_cols if c != 'label']
all_cols = num_cols_out+cate_cols_out
print(all_cols)

featureAssembler = VectorAssembler(inputCols=all_cols, outputCol="features")
spark_df = featureAssembler.transform(spark_df)

# 数据集划分
from pyspark.sql.functions import col

model_df=spark_df.select(col("features"),col('label_index').alias("label"))
train_df,test_df=model_df.randomSplit([0.5,0.5])


# 模型构建
print("开始训练=====>")
rf_classifier=RandomForestClassifier(labelCol='label',numTrees=10).fit(train_df)
print("开始预测=====>")
rf_predictions=rf_classifier.transform(test_df)
rf_predictions.show()
# 结果查看
print("特征重要性===>>>\n",rf_classifier.featureImportances)  # 各个特征的权重

# 评估模型性能
import pyspark.ml.evaluation as ev



def evluate_result(prediction_df):
    time1 = time.time()
    TP = prediction_df.filter('label==1 and prediction==1').count()
    TPTN = prediction_df.filter('(label==1 and prediction==1) or (label==0 and prediction==0)').count()
    TPFP = prediction_df.filter('prediction==1').count()
    TPFN = prediction_df.filter('label==1').count()
    ALL = prediction_df.count()

    #print(prediction_df.show())
    print("ALL:"+str(ALL))
    print("TP:"+str(TP))
    print("TPTN:"+str(TPTN))
    print("TPFP:"+str(TPFP))
    print("TPFN:"+str(TPFN))

    aa = 0.00000001
    val_precision = TP / (TPFP + aa)
    val_recall = TP / (TPFN + aa)
    val_accuracy = TPTN / (prediction_df.count() + aa)
    f1 = 2*val_precision * val_recall / ((val_precision + val_recall) + aa)
    print('val_precision:%f , val_recall:%f, val_total_accuracy:%f, val_f1_score:%f' % (val_precision, val_recall, val_accuracy, f1))
    print("evluate time: "+str((time.time()-time1)/ float(60) / float(60))+" h")

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
    areaUnderPR = evaluator.evaluate(prediction_df)
    print("BinaryClassificationEvaluator:areaUnderPR:%f" % (areaUnderPR))
    evaluator = BinaryClassificationEvaluator()
    areaUnderRoc = evaluator.evaluate(prediction_df)
    print("BinaryClassificationEvaluator:areaUnderRoc:%f" % (areaUnderRoc))
    return (val_precision, val_recall, val_accuracy, f1, areaUnderPR, areaUnderRoc)


evluate_result(rf_predictions)

import matplotlib as mpl
import matplotlib.pyplot as plt

FI = pd.Series(rf_classifier.featureImportances, index=all_cols)  # pySpark
# FI = pd.Series(rf.feature_importances_,index = featureArray) # sklearn
FI = FI.sort_values(ascending=False)
fig = plt.figure(figsize=(12, 5))
plt.bar(FI.index, FI.values, color="blue")
plt.xlabel('features')
plt.ylabel('importance')
plt.title("features importance")
plt.show()




# pipeline = Pipeline(stages=[encoder, featuresCreator, logistic])
#
# # 拟合模型
# birth_train, birth_test = births.randomSplit([0.7,0.3],seed=123)
#
# model = pipeline.fit(birth_train)
# test_model = model.transform(birth_test)
#































