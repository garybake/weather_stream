# weather_stream
Streaming weather data with prediction enrichment

Run by starting the feeder and then the stream predictor

python3 feeder.py localhost 8001

python3 pred_stream localhost 8001

The feeder feeds data from a file to a socket (mimicking a simple queue). (Also adds basic blockchain function)
The pred_stream takes batches off the feed and enriches it with a prediction.

Requires sklearn and pyspark
