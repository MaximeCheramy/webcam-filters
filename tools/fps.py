import time

pTime = 0

def print_fps():
    global pTime
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    print(f"FPS: {fps}")