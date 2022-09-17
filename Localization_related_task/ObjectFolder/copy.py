import os

for i in range(100):
    cmd = 'cp -r ./model.urdf ./'+ str(i+201) +'/'
    os.system(cmd)
