from time import sleep

from vidgear.gears import NetGear
import cv2
from queue import PriorityQueue

# activate Bidirectional mode
options = {"multiserver_mode": True}

ports = []
# define 3 NetGear Clients with `receive_mode = True` and defined parameters
for i in range(10):
    ports.append(5555 + i)

client = NetGear(
    receive_mode=True,
    port=ports,
    pattern=2,
    protocol="tcp",
    **options
)

# initialize a queue to store incoming frames
frame_queue = PriorityQueue(20)

# initialize a flag to indicate if the queue is full
queue_full = False

# loop over
while True:
    # prepare data to be sent
    target_data = "Hi, I am a Client here."

    # receive data from servers and also send our data
    data = client.recv(return_data=target_data)

    # check for data if None
    if data is None:
        break
    # print(data)
    # extract server_data & frame from data
    unique_address, server_data, frame = data

    # again check for frame if None
    if frame is None:
        break

    # add the frame to the queue using its ID as a reference
    if server_data is not None:
        frame_id = server_data
        frame_queue.put((frame_id, frame))

        # check if the queue is full
    if not queue_full and frame_queue.full():
        queue_full = True
    # display frames in order
    if queue_full:
        while not frame_queue.empty():
            _, frame_to_display = frame_queue.get()
            cv2.imshow("Output Frame", frame_to_display)

    # lets print extracted server data
    if not (server_data is None):
        print(server_data)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    sleep(0.05)
# close

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
