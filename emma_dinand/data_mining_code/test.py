import datetime
import time

def run_script():

    return f"Script ran at {datetime.datetime.now()}"

def data_source():
    i = 0
    while True:
        yield i  # Simulating streaming data
        i += 1
        time.sleep(0.1)