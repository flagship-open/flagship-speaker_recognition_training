import requests


sess = requests.Session()
name = 'aaa@yonsei.ac.kr-0.mp4'

ip_address = 'http://0.0.0.0:5000/Identification_Request'
path_dir = '.'
re_register = 'False'
test = 'True'
test_list = 'test_data/test_list.txt'

r = sess.post(ip_address, data={'path_dir': path_dir,'name': name, 're_register': re_register,
                                'test': test, 'test_list': test_list})

if r.status_code == 200:
    print(r,'success')
    print(r.content)
else:
    print('fail :(')
