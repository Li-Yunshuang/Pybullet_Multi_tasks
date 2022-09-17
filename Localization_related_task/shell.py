import os
import time

for num in range(100):
    cmd = 'python localize_new.py --num '+str(num+101)
    print(cmd)
    os.system(cmd)
            #time.sleep(45)

