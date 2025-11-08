import math
import time
import pantilthat

w = 0
while True:

    a = (math.sin(w*2) * 90 + 90) / 2
    b = math.sin(w*2.5) * 90
    a = int(a)
    b = int(b)

    pantilthat.tilt(a)
    pantilthat.pan(b)
    print(a, b)

    w += 0.001
    time.sleep(0.005)