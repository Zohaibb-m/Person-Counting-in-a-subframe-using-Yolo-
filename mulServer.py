import random
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

stream = VideoGear(source=0).start()

options = {"multiserver_mode": True}

# initialize frame ID counter
frame_id = 1

# create 3 servers with different port numbers
servers = []
for i in range(10):
    servers.append(NetGear(port=5555+i, logging=True, **options, pattern=2,protocol="tcp"))


def randomize(data):
    num = random.uniform(0.9, 1.1)
    sleep(num)
    return data


workers = ThreadPoolExecutor(max_workers=10)


def sendFrame(future):
    # extract frame object from tuple
    frame = future.result()[0]
    # send frame with sequential ID to client
    servers[future.result()[2] % 10].send(frame, message=future.result()[1])


while True:
    try:
        frame = stream.read()
        if frame is None:
            break
        future = workers.submit(randomize, [frame, frame_id, frame_id])
        future.add_done_callback(sendFrame)
        frame_id += 1
        sleep(1 / 10)

    except KeyboardInterrupt:
        break

stream.stop()

for i in range(10):
    servers[i].close()
