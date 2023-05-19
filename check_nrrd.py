import os, sys

key1 = b'space directions: '
v1 = b'(1,0,0) (0,-0,-1) (0,-1,-0)'
key2 = b'space origin: '
v2 = b'(-128.0,128.0,128.0)'
r = sys.argv[1]
fs = os.listdir(r)
fs = [os.path.join(r, f) for f in fs if f.endswith('.nrrd')]
for f in fs:
    with open(f, 'rb') as file:
        data = file.read()

    keyid = data.find(key1)
    tail_id = data[keyid+len(key1):].find(b')\n')
    data = data[:keyid+len(key1)] + v1 + data[keyid+len(key1)+tail_id+1:]
    keyid = data.find(key2)
    tail_id = data[keyid+len(key2):].find(b')\n')
    data = data[:keyid+len(key2)] + v2 + data[keyid+len(key2)+tail_id+1:]
    with open(f, 'wb') as file:
        file.write(data)