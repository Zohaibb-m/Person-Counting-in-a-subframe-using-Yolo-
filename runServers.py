import subprocess
from time import sleep

clientCMD = "python Client.py"
serverCMD = "python mulServer.py"

# start server process and wait for it to start
server = subprocess.Popen(serverCMD,shell=True)
sleep(2)

# start client process
client = subprocess.Popen(clientCMD,shell=True)

# wait for client process to complete
client.wait()

# close server process
server.terminate()
