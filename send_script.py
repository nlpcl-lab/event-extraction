import requests

filenames = ['10_1_mic1.wav','10_1_mic2.wav']

files = dict()
for filename in filenames:
    f = open(filename, 'rb')
    files[filename] = f

res = requests.post('http://localhost:8080/upload', files=files)
print('res :', res)

r = requests.get('http://localhost:8080/download')
with open('result.wav', 'wb') as f:
    f.write(r.content)
