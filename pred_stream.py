import sys
import json

from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession

from predmodel import PredModel

"""
Consumes a stream of weather data
Enriches the data with prediction
Destination is currently the console but not to hard to
output to another queue or database
"""


def run_stream(host, port):

    spark = (
        SparkSession.builder
        .master("local[8]")
        .config("spark.driver.cores", 8)
        .appName("GB_Streaming")
        .getOrCreate())

    sc = spark.sparkContext
    ssc = StreamingContext(sc, 0.5)

    model_file = 'forest_reg.joblib'
    pm = PredModel(model_file)

    lines = ssc.socketTextStream(host, port)
    preds = lines.map(lambda msg: pm.predict(json.loads(msg)))
    # preds.foreachRDD(lambda rdd: rdd.foreach(save_to_db))  TODO

    preds.pprint()

    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: process_stream.py <hostname> <port>", file=sys.stderr)
        sys.exit(-1)

    run_stream(host=sys.argv[1], port=int(sys.argv[2]))
