
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext, HiveContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import MinMaxScaler,MaxAbsScaler,StringIndexer, VectorIndexer,VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.ml.feature as ft
import pyspark.ml.evaluation as ev
import pyspark.ml.tuning as tune
from pyspark.ml.stat import ChiSquareTest

import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


def get_test_result(prediction_df):
    TP = prediction_df.filter('label==1 and prediction==1').count()
    TPTN = prediction_df.filter('(label==1 and prediction==1) or (label==0 and prediction==0)').count()
    TPFP = prediction_df.filter('prediction==1').count()
    TPFN = prediction_df.filter('label==1').count()

    val_precision = TP / TPFP
    val_recall = TP / TPFN
    val_accuracy = TPTN / prediction_df.count()
    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
    val_areaUnderPR = evaluator.evaluate(prediction_df)
    evaluator = BinaryClassificationEvaluator()
    val_areaUnderRoc = evaluator.evaluate(prediction_df)

    print('val_precision: {} , \n'
          'val_recall: {} ,\n '
          'val_total_accuracy: {} ,\n'
          ' val_f1_score: {} ,\n '
          'val_areaUnderPR: {} ,\n '
          'val_AUC: {} \n'.format(val_precision, val_recall,
                                  val_accuracy, val_f1,
                                  val_areaUnderPR, val_areaUnderRoc))


def get_best_cv_par(cvModel):
    # 查看最佳模型参数
    param_maps = cvModel.getEstimatorParamMaps()
    eval_metrics = cvModel.avgMetrics

    param_res = []

    for params, metric in zip(param_maps, eval_metrics):
        param_metric = {}
        for key, param_val in zip(params.keys(), params.values()):
            param_metric[key.name] = param_val
        param_res.append((param_metric, metric))
    res = sorted(param_res, key=lambda x: x[1], reverse=True)
    for r in res:
        print(r)
        print("\n")
    return res


def plot_feature_importance(rf_classifier):
    FI = pd.Series(rf_classifier.featureImportances, index=all_cols)  # pySpark
    # FI = pd.Series(rf.feature_importances_,index = featureArray) # sklearn
    FI = FI.sort_values(ascending=False)
    plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.bar(FI.index, FI.values, color="blue")
    plt.xlabel('features')
    plt.xticks(rotation=-30)
    plt.ylabel('importance')
    plt.title("features importance")
    plt.show()


def main():

    spark = SparkSession.builder.appName("pyspark_test").enableHiveSupport().getOrCreate()


    ## 读取数据，既可以读取本地的csv等数据，也可以读取HDFS上的Hive表

    spark_df = spark.read.csv('./data/diamonds.csv',inferSchema=True,header=True)
    spark_df,_ =spark_df.randomSplit([0.05,0.95])


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

    # **基础描述**
    print(spark_df.count())  # 行数
    print(spark_df.columns) # 列名称
    print(spark_df.printSchema()) # 结构及属性显示
    print(spark_df.show(5,False))
    print(spark_df.describe().show()) # 均值/最值等描述

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

    def label_transform(cut):
        if cut in ["Good","Very Good"]:
            return "High"
        else:
            return "Low"


    myudf=udf(label_transform,StringType())  # create udf using python function # 输出为string格式
    # 利用udf进行label转换
    spark_df = spark_df.withColumn("label",myudf(spark_df['cut']))
    spark_df = spark_df.drop('cut')
    # **another udf的例子**
    # 使用lamba创建udf
    price_udf = udf(lambda price: "high_price" if price >= 330 else "low_price", StringType())  # using lambda function
    # 新建一列price_group
    spark_df = spark_df.withColumn("price_group", price_udf(spark_df["price"]))



    ## 特征处理
    ## pyspark.ml.feature提供的特征处理功能，满足了大部分机器学习的特征处理需求。
    ## 这里进行两个基本操作，把数值型的特征进行归一化，而对类别型特征进行label编码


    # 归一化函数，将列标准化到[0,1]之间

    num_cols =["carat","depth","table","price","x","y","z"]

    cate_cols = ["color","clarity","label","price_group"]

    # 批量地对数值的特征进行归一化
    for num_col in num_cols:
        # 用于将多个列合并为一个向量列，直接transform即可，经常用的
        # numsAssembler = VectorAssembler(inputCols=num_cols, outputCol="num_features")
        # spark_df = numsAssembler.transform(spark_df)
        # 要用VectorAssembler转成Dense vector
        numsAssembler = VectorAssembler(inputCols=[num_col], outputCol=num_col+"_ass")
        spark_df = numsAssembler.transform(spark_df)
        mmScaler = MinMaxScaler(inputCol=num_col+"_ass", outputCol=num_col+"_scale")
        scalemodel = mmScaler.fit(spark_df)
        spark_df = scalemodel.transform(spark_df)

    # 这里的代码用到了for循环，似乎有些丑，但是没有想到更好的方法
    # 除了for循环，可以借助于map，代码如下
    """
    # 首先计算出所有数值列的最大值和最小值，待后面使用
    cols_max = []
    cols_min = []
    for col in num_cols:
        cols_max.append(spark_df.agg(F.max(col)).toPandas().iloc[0,0])
        cols_min.append(spark_df.agg(F.min(col)).toPandas().iloc[0,0])
    
    spark_df_rdd = spark_df.rdd.map(lambda row:(float((row[0]-cols_min[0])/(cols_max[0]-cols_min[0])),
                                                row[1],row[2],
                                                float((row[3] - cols_min[1]) / (cols_max[1] - cols_min[1])),
                                                float((row[4] - cols_min[2]) / (cols_max[2] - cols_min[2])),
                                                float((row[5] - cols_min[3]) / (cols_max[3] - cols_min[3])),
                                                float((row[6] - cols_min[4]) / (cols_max[4] - cols_min[4])),
                                                float((row[7] - cols_min[5]) / (cols_max[5] - cols_min[5])),
                                                float((row[8] - cols_min[6]) / (cols_max[6] - cols_min[6])),
                                                row[9],row[10]
                                                 ))
    fields = [("carat_scale", FloatType()),  
              ("color", StringType()), 
              ("clarity", StringType()),
              ("depth_scale", FloatType()),
              ("table_scale",FloatType()), 
              ("price_scale", FloatType()),
              ("x_scale", FloatType()),
              ("y_scale", FloatType()), 
              ("z_scale", FloatType()),
              ("label", StringType()),
              ("price_group", StringType())]
    schema = StructType([StructField(e[0], e[1], True) for e in fields])
    spark_df = spark.createDataFrame(spark_df_rdd,schema)
    
    spark_df.show()
    
    # 这种方法虽然不用for，但是也很麻烦，当列数一多尤其麻烦，所以还是用for吧 -_-||
    # 这里的map其实和udf一样，所以可以提前定义好每一列的归一化的udf，然后新增归一化的列即可，但是也很麻烦。
    
    """

    # 批量地，针对单个类别型特征进行转换，把字符串的列按照出现频率进行排序
    for cate_col in cate_cols:
        stringIndexer = StringIndexer(inputCol=cate_col,outputCol=cate_col+"_index")
        stringmodel = stringIndexer.fit(spark_df)
        spark_df = stringmodel.transform(spark_df)



    num_cols_out = [x+"_scale" for x in num_cols]

    cate_cols_out = [c+"_index" for c in cate_cols if c != 'label']

    all_cols = num_cols_out+cate_cols_out
    # 把全部特征合并
    featureAssembler = VectorAssembler(inputCols=all_cols, outputCol="features")
    spark_df = featureAssembler.transform(spark_df)

    # 卡方检验
    print("======>>>对类型型变量进行卡方检验")


    cate_featureAssembler = VectorAssembler(inputCols=cate_cols_out, outputCol="cate_features")
    spark_df = cate_featureAssembler.transform(spark_df)
    r = ChiSquareTest.test(spark_df, "cate_features", "label_index").head()

    print("pValues: " + str(r.pValues))
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    print("statistics: " + str(r.statistics))
    #print(spark_df.show())


    # 模型构建加网格搜索
    print("开始k-fold网格搜索并且训练=====>")
    rf_classifier=RandomForestClassifier(labelCol='label')

    grid = tune.ParamGridBuilder()\
        .addGrid(rf_classifier.numTrees, [3,10,20])\
        .addGrid(rf_classifier.maxDepth, [3,10,20])\
        .build()
    evaluator = BinaryClassificationEvaluator()
    # 使用K-Fold交叉验证评估各种参数的模型
    cv = tune.CrossValidator(
        estimator=rf_classifier,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=2
    )

    # 数据集划分
    from pyspark.sql.functions import col
    model_df=spark_df.select(col("features"),col('label_index').alias("label"))
    train_df,test_df=model_df.randomSplit([0.7,0.3])

    # cvModel 返回估计的最佳模型
    cvModel = cv.fit(train_df)
    best_cv_par = get_best_cv_par(cvModel)

    # 以最优的参数再进行一次训练，否则直接从cvModel里面拿不到特征重要性
    rf_classifier_best=RandomForestClassifier(labelCol='label',
                                              numTrees = best_cv_par[0][0]["numTrees"],
                                                maxDepth=best_cv_par[0][0]["maxDepth"]).fit(train_df)


    # 以最佳模型进行预测
    print("开始预测=====>")
    rf_predictions = rf_classifier_best.transform(test_df)
    rf_predictions.show(5)
    get_test_result(rf_predictions)

    # 查看特征重要性
    print("特征重要性===>>>\n",rf_classifier_best.featureImportances)  # 各个特征的权重
    plot_feature_importance(rf_classifier_best)

    # 模型保存
    rf_classifier_best.write().overwrite().save("./data/RF_model")

    print("模型保存完毕")



if __name__ == '__main__':
    main()


























