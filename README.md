[Flagship] speaker recognition module
======================

#### 0. Note

* (2020/12) ���� �н� �ڵ� ������ ���� ������ ������ ������Ʈ �Ǿ����ϴ�.

#### 1. ���� ȯ��

* OS : Ubuntu 16.04
* GPU driver : Nvidia CUDA 10.1

#### 2. System/SW Overveiw

* ���� ��ǥ: ���� �Էµ� ȭ�ڿ� ���Ͽ� ����, ���� ȭ�� �νı⸦ �̿��Ͽ� ���ٸ� �α��� ���� �������� ���� �� �־�� �Ѵ�.
* ���� �����:
 
![�ý��� �ܺ� �������̽�](./image/1.png)
![��� �帧��](./image/2.png)

#### 3. How to install

> pip install -r requirements.txt

#### 4. Main requirement

* Python 3.5
* tensorflow-gpu 1.12.0 
* Keras 2.2.5 (2.3 �̻��� �ȵ�)

#### 5. Network Architecture and features

![��Ʈ��ũ ��ü �н� ����](./image/4.png)

* **Model:**
* We used dense layer to integrate image and voice model
* After concatenating their feature, final feature is fed into dense layer.

#### 6. Dataset
* Youtube Faces �����ͼ�
* 1595���� ȭ�ڿ� ���� 3425���� ���������� ����. �� ������ frame ���� �̹��� �����Ͱ� ������

#### 7. Quick start

> python train_nets.py
