[Flagship] speaker recognition module
======================

#### 0. Note

* (2020/4/29) 4�� ������ ������ ������Ʈ �Ǿ����ϴ�.
* (2020/5/29) 5�� ������ ������ ������Ʈ �Ǿ����ϴ�.
> 7���� ȭ�ڷ� ������ ��ü ����(�� class�� 10��, �� 70)�� ���� �׽�Ʈ �ڵ� ÷���Ͽ����ϴ�.
* (2020/6/18) 6�� 1�� ���� �ڵ� ������ ������Ʈ �Ǿ����ϴ�.
> Test : True �� ��� ��� ���� �����Ϳ� ���� Identification accuracy�� ������ False�� ������ ��� �� �̹����� ���� identification ����� ���� �� �ֽ��ϴ�.
* (2020/10/23) ���� ������ ���� ������ ������ ������Ʈ �Ǿ����ϴ�.
> ���� ���� �ڵ��� ���� �ý��ۿ� ���� pytorch�� tensorflow�� ��� ��ȯ(network, app.py, app_web.py)�Ͽ��� ���� ��Ʈ��ũ �����Ͽ� Final_result �κ� �߰��Ͽ����ϴ�.
* (2020/11/10) ���� ������ ���� ������ ������ ������Ʈ �Ǿ����ϴ�.
> ���� ȭ�� �ν� �� �ε忡 ������ �ִ� �κ��� �ذ��Ͽ� ���� Test �ڵ带 �ϼ��Ͽ����ϴ�.
* (2020/12/29) UTF-8 ������ �ڵ� ������ �Ϸ��Ͽ����ϴ�.

#### 1. ���� ȯ��

* OS : Ubuntu 16.04
* GPU driver : Nvidia CUDA 9.0 �̻�

#### 2. System/SW Overveiw

* ���� ��ǥ: ���� �Էµ� ȭ�ڿ� ���Ͽ� ����, ���� ȭ�� �νı⸦ �̿��Ͽ� ���ٸ� �α��� ���� �������� ���� �� �־�� �Ѵ�.
* ���� �����:
 
![�ý��� �ܺ� �������̽�](./img/1.png)
![��� �帧��](./img/2.png)

#### 3. How to install

> pip install -r requirements.txt

#### 4. Main requirement

* Python 3.5
* tensorflow-gpu 1.12.0 
* Keras 2.2.5 (2.3 �̻��� �ȵ�)

#### 5. Network Architecture and features

![ȭ�� �ν� ��Ʈ��ũ ��ü ����](./img/3.png)

* **Model:**
* We used depthwise separable convolution as a CNN-based lightweight network.
* The number of parameters is 1/20 times lighter than the existing ResNet, and the amount of computational cost is also much less

* **Metrics:**
* Accuracy is divided into identification and verification, each performance is about 99% and 77%
* Each image extracts 512-dimensional features through the network
* It uses **cosine similarity** and is calculated by dividing the absolute value by the dot product.
* It has a value from -1 to 1 and uses the value to find the nearest speaker (The closer its value near 1, the more similar the target is.)

![���� ��Ʈ��ũ ��ü ����](./img/4.png)

* **Dataset:**
* Youtube Faces �����ͼ�
* 1595���� ȭ�ڿ� ���� 3425���� ���������� ����. �� ������ frame ���� �̹��� �����Ͱ� ������

* **Model:**
* We used dense layer to integrate image and voice model
* After concatenating their feature, final feature is fed into dense layer.

#### 6. Quick start

* 2���� ������� �̿��� �� �ִ�

##### 1. �͹̳η� �̿��ϴ� ���

> python app.py�� �����ϰ� �ּҰ� ������� Ȯ���Ѵ� ex)http://127.0.0.1:5000/
> python Request.py�� �����Ͽ� Input�� �������ָ� ������ ���� response�� ��´�.
> * {"100001": "Multi-modal Speaker Recognition - Speaker ID : OOOO", "100002": "Image Based Speaker Recognition - Speaker ID : OOOO", "100003": "Voice Based Speaker Recognition - Speaker ID: OOOO"}

##### 2. Web �󿡼� �̿��ϴ� ���

> python app_web.py�� ������ �� ����� �ּҷ� ����.
> "Try it out"�� ���� Input ���Ŀ� �°� �Է°��� �־��ְ� �����Ų �� response�� ���� �� �ִ�.

##### 3. Training ���

> ./train ��ο� �� python train_nets.py �� ���డ���ϴ�.
> �����ͼ��� ��� YoutubeFace�� ����Ͽ��� �ش� �����ʹ� ./dataset�� ��ο� �־��ָ� �ȴ�.

#### 7. HTTP-server API description

* path

> /SSD/project/ai_5th/flask_v2

* Parameter

|Parameter|Type|Description|
|---|---|---|
|name|string|string of target's mp4 file name|
|path_dir|string|root path|
|re_register|string|when you've registered before, but the feature doesn't express you properly. you can use re_register=True| 

* Request

> '''
> data: {
> 'name': '�̸���-��ȣ.mp4'
> 'ip_address': 'http://0.0.0.0:5000/Identification_Request'
> 'path_dir': '/SSD/project/ai_5th/flask_v2/'
> 're_register': 'False'
> }
> '''

* Response OK

> '''
> 200 success
> {
> "100001" : "Multi-modal Speaker Recognition - Speaker ID : OOOO"
> }

#### 8. Repository overview

* 'utils/' -������ ����, ���翩��, identification �ڵ�
* 'data/' -Ÿ�� �����ͷ� mp4�������� ����Ǿ� ����
* 'network/' -����/���� �ν� ��Ʈ��ũ
* 'align/' -�� �κ��� �����ϱ� ���� ��ó�� �ڵ�
* 'best_model/' -model weight ����
* 'gallery/voice_gallery' -ȭ�ڰ� ó�� ��ϵ� �� ���� feature�� �����صδ� ����
