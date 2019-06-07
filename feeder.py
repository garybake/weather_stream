import socket
import time
import json
from csv import DictReader
import sys

"""
Simulates a queue of data
Once a connection is setup it sends a single line from a file
per second
Adds a basic blockchain capability to validate data
"""

FILENAME = '/home/gary/Devel/companyx/data/DS_sensor_weather.csv'


def get_hash(obj):
    return hash(frozenset(obj.items()))


def run_feed(host, port):
    print('Binding to {}:{}'.format(host, port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print('Bound succesfully')

    conn, addr = s.accept()

    last_hash = 0
    input_file = DictReader(open(FILENAME))
    for row in input_file:
        del row['relative_humidity_pm']  # Drop the target field

        for field in row:
            if row[field] != '':
                row[field] = float(row[field])
            else:
                row[field] = None

        row['feed_timestamp'] = time.time()
        row['last_hash'] = last_hash
        last_hash = get_hash(row)
        try:
            msg = json.dumps(row) + '\n'
            msg = msg.encode('utf-8')
            print(msg)
            conn.sendall(msg)
        except BrokenPipeError:
            print("Client Broke connection")
            break
        time.sleep(0.5)

    conn.close()
    print('Stopped')

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: feeder.py <hostname> <port>", file=sys.stderr)
        sys.exit(-1)

    run_feed(host=sys.argv[1], port=int(sys.argv[2]))
