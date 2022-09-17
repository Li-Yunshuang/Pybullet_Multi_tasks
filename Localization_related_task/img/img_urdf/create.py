import os

for i in range(100):
    cmd = 'mkdir -p '+str(i+101)
    os.system(cmd)
