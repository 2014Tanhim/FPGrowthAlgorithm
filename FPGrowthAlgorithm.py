# -*- coding: utf-8 -*-
"""FPGrowthAlgorithm.ipynb

"""

from pyspark.ml.fpm import FPGrowth
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)



from google.colab import files
uploaded = files.upload()

# This opens a handle to your file, in 'r' read mode
file_handle = open('sampletest.txt', 'r')
# Read in all the lines of your file into a list of lines
lines_list = file_handle.readlines()
# Extract dimensions from first line. Cast values to integers from strings.
#cols, rows = (int(val) for val in lines_list[0].split())
# Do a double-nested list comprehension to get the rest of the data into your matrix
my_data = [[int(val) for val in line.split()] for line in lines_list[1:]]

for x in range(9): 
  print ("(",x,",",my_data[x],")")

x = [( 0 , [1, 3, 5, 9, 11, 17, 18, 20, 22, 26, 31, 33, 39, 40, 42, 48, 49, 53, 58, 61, 67, 70, 75, 79, 81, 85, 87, 90, 92, 94, 97, 99, 100] ),
( 1 , [1, 3, 5, 8, 11, 16, 18, 21, 22, 27, 32, 33, 39, 40, 42, 47, 50, 53, 57, 61, 67, 71, 75, 79, 81, 85, 87, 89, 91, 95, 97, 99, 100] ),
( 2 , [1, 3, 5, 8, 12, 16, 18, 20, 22, 26, 31, 33, 39, 40, 42, 46, 49, 54, 58, 61, 66, 70, 74, 79, 81, 85, 87, 89, 92, 95, 97, 99, 101] ),
( 3 , [1, 3, 5, 9, 10, 17, 19, 20, 22, 29, 31, 36, 39, 40, 43, 48, 50, 54, 56, 61, 66, 72, 76, 79, 83, 86, 88, 90, 92, 95, 97, 99, 101] ),
( 4 , [1, 3, 5, 8, 12, 17, 18, 20, 22, 27, 31, 36, 38, 40, 43, 48, 49, 53, 56, 61, 67, 70, 76, 79, 81, 85, 87, 90, 91, 95, 97, 99, 101] ),
( 5 , [1, 3, 5, 8, 12, 16, 18, 20, 22, 26, 31, 33, 39, 40, 42, 46, 49, 54, 57, 63, 67, 69, 73, 79, 84, 85, 87, 89, 93, 94, 97, 99, 100] ),
( 6 , [1, 3, 5, 9, 11, 16, 18, 20, 22, 26, 32, 35, 39, 40, 42, 46, 49, 54, 57, 61, 66, 70, 75, 79, 81, 85, 88, 89, 92, 95, 97, 98, 100] ),
( 7 , [1, 3, 5, 9, 11, 17, 18, 20, 22, 30, 31, 33, 38, 41, 42, 46, 49, 53, 56, 61, 66, 70, 75, 79, 81, 85, 88, 89, 92, 95, 97, 99, 100] ),
( 8 , [1, 3, 5, 9, 11, 16, 18, 20, 22, 26, 31, 35, 39, 40, 42, 47, 49, 55, 60, 61, 66, 70, 75, 79, 81, 85, 87, 89, 92, 94, 97, 99, 100] )]

from pyspark.ml.fpm import FPGrowth

df = spark.createDataFrame(x, ["id", "items"])

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show(100)

# Display generated association rules.
model.associationRules.show(100)

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()
