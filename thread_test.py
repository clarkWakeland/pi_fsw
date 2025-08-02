import threading
import time

class MyWorker:
    def __init__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        print("Thread started")

    def run(self):
        while True:
            print("Thread is working")
            time.sleep(1)

worker = MyWorker()

for i in range(5):
    print("Main is working", i)
    time.sleep(1)
