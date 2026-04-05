---
layout: post
title: "ResNet, GloVe, Mel-spectrogram을 이용한 데이터 수치화 및 시각화 실습"
date: 2026-03-16 23:17:26 +0900
category: Multimodal
---

## 😊 들어가며
멀티모달 인공지능의 핵심은 서로 다른 형태의 데이터를 컴퓨터가 이해할 수 있는 공통된 숫자 공간에 모으는 것입니다. 하지만 그 전에 반드시 선행되어야 할 작업이 있습니다. 바로 각 데이터(Image, Text, Audio) 를 독립적으로 분석하여 고유의 특징을 추출하는 Unimodal Representation 단계입니다.

이번 포스팅에서는 이미지 처리를 위한ResNet50, 텍스트 처리를 위한 GloVe 와 LSTM, 그리고 오디오 처리를 위한 Mel-spectrogram를 활용하여, 세 가지 서로 다른 모달리를 수학적 벡터로 변환하는 방법을 실습해보겠습니다.

---

## 🖼️ Image Representation

### 🟩 Raw Image Load: 원본 데이터 불러오기
딥러닝 모델, 특히 ResNet 이나 ViT 같은 시각 지능 모델을 다룰 때 가장 먼저 만나는 코드가 바로 이미지 로드입니다. 단순히 파일을 여는 것처럼 보이지만, 이 안에는 모델 학습을 위한 데이터 처리 원리가 숨어 있습니다.

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 불러오기 및 RGB 변환
img = Image.open('dog.jpg').convert('RGB')

# 이미지의 차원 확인 (Height, Width, Channel)
print(np.array(img).shape) # 출력 예: (500, 800, 3)
plt.imshow(img)
```

#### ◼️ .convert('RGB') 를 하는 이유
우리가 인터넷에서 다운로드하는 이미지 파일 (jpg, png, gif 등)은 저장 방식에 따라 데이터 채널이 제각각입니다.

* **RGBA 문제:** png 파일의 경우 투명도를 나타내는 Alpha 채널이 포함되어 4 채널 (R, G, B, A) 인 경우가 많습니다.
* **Grayscale 문제:** 흑백 이미지는 1 채널로 저장됩니다.
* **일관성 유지:** ResNet50과 같은 사전 학습 모델은 항상 3 채널 (R, G, B) 입력을 기대합니다. 만약 채널 수가 맞지 않으면 연산 과정에서 런타임 에러가 발생하므로, 모든 입력을 강제로 RGB 표준으로 통일해주는 과정이 필수적입니다.

#### ◼️ np.array(img) 를 통한 수치화 확인
이미지는 우리 눈에는 아름다운 그림이지만, 컴퓨터에게는 그저 숫자가 가득 찬 고차원 행렬일 뿐입니다. 우리가 PIL 객체를 굳이 Numpy 배열로 변환하여 확인하는 이유는 다음과 같습니다.

* **차원 검증:** 이미지의 높이($H$), 너비($W$), 채널($C$) 이 모델이 원하는 규격에 맞는지 확인합니다.
* **픽셀 값의 분포 확인:** 일반적인 이미지 데이터는 0 부터 255 사이의 정수(uint8) 값을 가집니다. 나중에 모델에 넣을 때는 이를 $0 \sim 1$ 사이의 실수로 정규화해야 하는데, 그 전단계로서 데이터의 날것(Raw) 상태를 확인하는 표준적인 방법입니다.

### 🟩 Preprocess Image: 모델을 위한 데이터 규격화
두 번째 단계인 이미지 전처리로 넘어가 보겠습니다. 이 과정은 단순히 이미지의 모양을 바꾸는 것이 아니라, ResNet 과 같은 딥러닝 모델이 가장 잘 이해할 수 있는 형태로 데이터를 정규화하는 단계입니다.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

preprocess = T.Compose([
    T.Resize((224, 224)), # 1. 공간 해상도 통일
    T.ToTensor(),         # 2. 데이터 타입 및 범위 변경 
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 3. 분포 정규화
])

x = preprocess(img)
print(x.shape) # (3, 224, 224)
```

#### ◼️ T.Resize((224, 224))
대부분의 사전 학습된 CNN 모델 (Resnet, VGG 등) 은 224x224 크기의 이미지를 입력받도록 설계되었습니다.
* **이유:** 모델 내부의 Convolution 연산과 Pooling 과정을 거치며 최종적으로 특징 벡터가 추출될 때, 입력 크기가 고정되어야 출력 차원도 일정하게 유지되기 때문입니다.

#### ◼️ T.ToTensor()
이 함수는 Numpy 배열을 PyTorch Tensor 로 바꾸면서 동시에 아주 중요한 작업을 수행합니다.

* **차원 순서 변경:** Numpy 의 (H, W, C) 구조를 파이토치 표준인 (C, H, W) 로 재배치합니다.
* **값의 스케일링:** $0 \sim 255$ 사이의 정수 값을 $0 \sim 1$ 사이의 실수(float32) 로 변환합니다. 이는 모델의 가중치 연산 시 수치적 안정성을 제공합니다.

#### ◼️ T.Normalize
입력한 [0.485, 0.456, 0.406] 과 같은 수치들은 ImageNet 이라는 거대 데이터셋의 통계값입니다.

* **Z-score Normalization:** 각 채널의 픽셀 값에서 평균을 빼고 표준편차로 나눕니다.
* **효과:** 데이터의 중심을 $0$ 근처로 맞추어 Gradient Descent 과정에서 학습이 더 빠르고 안정적으로 진행되도록 돕습니다. 우리가 사용하는 ResNet 이 이 분포에서 가장 잘 작동하도록 학습되었기 때문에, 우리 데이터도 똑같은 분포로 맞춰주는 것입니다.

```python
x = x.unsqueeze(0)
print(x.shape) # [1, 3, 224, 224]
```

#### ◼️ x = x.unsqueeze(0)
unsqueeze 함수는 텐서의 특정 위치에 크기가 1인 차원을 추가하는 함수입니다.

* **필요성:** 딥러닝 모델은 데이터 한 장이 아닌 배치 (Batch) 단위로 연산합니다.
* **결과:** unsqueeze(0)은 0번째 인덱스에 차원을 하나 추가하여, [배치 크기, 채널, 높이, 너비] 라는 4차원 규격을 완성합니다.

### 🟩 Resnet50 모델 불러오기
멀티모달 학습에서 이미지 인코더를 처음부터 학습시키는 것은 매우 비효율적입니다. 따라서 수백만 장의 이미지를 이미 학습한 모델을 빌려옵니다.

```python
# 사전 학습된 가중치(Weights)와 함께 모델 로드
resnet = torchvision.models.resnet50(weights='DEFAULT')
resnet.eval()
```
#### ◼️ weights='DEFAULT'
* **의미:** ImageNet 이라는 거대 데이터셋에서 이미지 분류를 위해 최적화된 수천만 개의 파라미터(가중치)를 그대로 가져옵니다.
* **효과:** 만드려는 모델은 처음부터 데이터를 학습할 필요가 없습니다. 이미 완성된 모델을 바탕으로 실습하려는 데이터의 특징만 효과적으로 추출하면 됩니다.

#### ◼️ resnet.eval()
모델을 불러온 뒤 .eval() 을 호출하는 것은 선택이 아닌 필수입니다. 모델 내부에는 학습 시와 테스트 시 동작이 다른 레이어들이 존재하기 때문입니다.

* **Batch Normalization (BN):** 학습 시에는 현재 배치의 통계값을 사용하지만, 평가 모드에서는 학습 때 저장된 Running Average 값을 사용하여 일관된 출력을 보장합니다.

* **Dropout:** 학습 시에는 뉴런을 무작위로 끄지만, 평가 모드에서는 모든 뉴런을 사용하여 최선의 예측을 수행합니다.

* **미적용시 부작용:** 이 과정을 생략하면 똑같은 이미지를 넣어도 넣을 때마다 결과 벡터가 미세하게 달라지는 대참사가 발생할 수 있습니다.

### 🟩 Calculate Image Embedding: 단계별 데이터 흐름 분석
#### ◼️ 출력 크기 계산 공식
CNN 레이어를 통과할 때 출력되는 이미지의 크기($O$) 는 입력 크기($I$), 커널 사이즈($K$), 패딩($P$), 스트라이드($S$) 에 의해 결정됩니다.

$$O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1$$

이 수식을 머릿속에 담아두고, ResNet50 의 각 층을 통과할 때 어떤 일이 벌어지는지 살펴보겠습니다.

```python
# 전처리 및 배치 차원 추가
x = preprocess(img).unsqueeze(0)
print(x.shape) # (1, 3, 224, 224)

x = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(x))))
print(x.shape) # (1, 64, 56, 56)
```

#### ◼️ Initial Blocks (Conv1 ~ Maxpool)
가정 먼저 실행되는 이 구간은 이미지의 거대한 공간 정보를 빠르게 압축하여 핵심 특징만 남기는 역할을 합니다.

* **conv1:** $7 \times 7$ 커널과 $S=2, P=3$ 을 사용하여 해상도를 절반으로 줄이고 채널을 64 개로 늘립니다.4
* **bn1 & relu:** 수치적 안정성을 더하고 비선형성을 부여합니다.
* **maxpool:** $3 \times 3$ 커널과 $S=2, P=1$ 을 사용하여 해상도를 다시 한번 절반으로 줄입니다.

```python
x = resnet.layer1(x)
print(x.shape) # (1, 256, 56, 56)
x = resnet.layer2(x)
print(x.shape) # (1, 512, 28, 28)
x = resnet.layer3(x)
print(x.shape) # (1, 1024, 14, 14)
x = resnet.layer4(x)
print(x.shape) # (1, 2048, 7, 7)
```

#### ◼️ Sequential Layers (Layer 1 ~ 4)
이곳은 ResNet 의 핵심인 Residual Block 들이 모여 있는 곳입니다. 층을 거칠수록 이미지는 작아지지만, 채널(정보의 깊이)은 깊어집니다.

* **Layer 1:** 해상도는 유지하면서 채널만 확장합니다.
* **Layer 2:** 이미지 크기를 절반으로 줄이고 채널을 늘립니다.
* **Layer 3:** 다시 한번 크기를 줄여 특징을 더 압축합니다.
* **Layer 4:** 마지막 시각적 특징 추출 단계입니다.

```python
x = resnet.avgpool(x)
print(x.shape) (1, 2048, 1, 1)
print(torch.flatten(x, 1).shape) (1, 2048)
```

#### ◼️ Global Average Pooling & Flatten
공간 정보($7 \times 7$) 가 더 이상 필요하지 않은 시점입니다. 이를 하나의 수치로 요약합니다.

* **avgpool (Global Average Pooling):** $7 \times 7$ 격자 안에 있는 모든 값의 평균을 내어 $1 \times 1$ 로 만듭니다.
* **torch.flatten(x, 1):** 불필요한 차원을 제거하여 1차원 벡터로 펼칩니다.

---

## 🔠 Text Representation

### 🟩 gensim 라이브러리 설치
본격적인 실습에 앞서, 대규모 텍스트 코퍼스를 처리하고 단어 임베딩(Word Embedding) 을 자유자재로 다룰 수 있게 해주는 gensim 라이브러리를 설치합니다.

```python
# 자연어 처리 모델 로드 및 벡터 연산을 위한 gensim 설치
!pip install gensim
```

#### ◼️ Gensim을 사용하는 이유
* **효율적인 메모리 관리:** 수십만 개의 단어 벡터(GloVe, Word2Vec 등) 를 메모리에 효율적으로 올리고 빠르게 조회할 수 있도록 설계되었습니다.

* **사전 학습 모델 저장소:** Google News, Wikipedia 등 거대 데이터셋으로 미리 학습된 임베딩 모델을 단 한 줄의 코드로 다운로드할 수 있는 API 를 제공합니다.

* **유사도 연산 최적화:** 단어 간의 거리를 계산하거나 유사한 단어를 찾는 알고리즘이 고도로 최적화되어 있어, 멀티모달 학습의 텍스트 파트를 구현하기에 최적의 도구입니다.

### 🟩 GloVe 모델 로드
이제 gensim 의 API 를 사용하여 사전 학습된 GloVe 모델을 로드합니다. 이번에는 위키피디아와 기가워드 데이터셋으로 학습된 300 차원 버전을 선택했습니다.

```python
import gensim.downloader as api

# 1. 300차원 GloVe 임베딩 모델 로드
glove = api.load('glove-wiki-gigaword-300')

# 2. 모델의 규모 확인
print(len(glove)) # 400,000 개의 단어
print(glove.vector_size) # 300 차원 단어 임베딩
```

#### ◼️ 400,000 개의 단어
* **의미:** 이 모델 안에는 일상적인 단어부터 전문 용어까지 약 40만 개 의 단어에 대한 이해가 담겨 있습니다.

* **효과:** 우리가 가진 Flickr8k 데이터셋에 생소한 단어가 등장하더라도, GloVe 가 이미 학습한 주변 단어와의 관계를 통해 그 의미를 추론할 수 있게 됩니다.

#### ◼️ 300 차원 (300d)
* **수치적 해석:** 하나의 단어는 300 개 의 실수(float) 로 이루어진 벡터로 변환됩니다.

* **비교:** 2차원 평면이 '위치' 만 나타낸다면, 300 차원 의 공간은 단어의 성별, 품사, 감정, 주제 등 수많은 '속성' 들을 각 축에 담고 있습니다.

### 🟩 단어 사이 거리 측정
임베딩 공간에서 두 벡터가 가리키는 방향이 얼마나 유사한지를 측정하기 위해 코사인 유사도 (Cosine Similarity) 를 사용합니다.

```python
# 1. 코싸인 유사도 연산 함수 정의 (0번째 차원을 기준으로 계산)
cos = nn.CosineSimilarity(dim=0)
```
#### ◼️ 코싸인 유사도 사용 이유
* **방향성의 중요성:** 단어 임베딩에서는 벡터의 절대적인 크기(길이)보다 가리키는 방향 이 단어의 의미를 더 잘 나타냅니다.

* **수치적 의미:** 결과값이 1 에 가까울수록 두 단어는 임베딩 공간 내에서 같은 방향을 바라보는 '이웃' 임을 뜻합니다. 반대로 0 에 가깝다면 서로 아무런 상관이 없는 독립적인 단어라는 의미입니다.

```python
# GloVe 에서 단어 벡터 추출 후 텐서 변환
dog = torch.tensor(glove['dog'])
cat = torch.tensor(glove['cat'])
person = torch.tensor(glove['person'])

# 유사도 출
print(cos(dog, cat))    # 결과: 약 0.6817
print(cos(dog, person)) # 결과: 약 0.3366
print(cos(torch.tensor(glove['have']), torch.tensor(glove['has']))) # 결과: 0.7619
```

#### ◼️ 실습 결과 분석
* **dog vs cat:** 둘 다 '네 발 달린 반려동물' 이라는 공통 속성을 공유하므로 벡터가 매우 밀접하게 배치되어 높은 유사도가 나옵니다.

* **dog vs person:** 생명체라는 공통점은 있지만, '사람' 과 '동물' 이라는 큰 차이점이 존재하여 유사도가 낮게 측정됩니다.

* **have vs has:** 의미는 같지만 문법적 형태만 다른 단어들입니다. GloVe 는 이러한 문법적 관계(Syntactic relationship) 도 벡터의 위치로 훌륭하게 표현해 냅니다.

### 🟩 Convert raw text sentence into GloVe word embeddings
자연어 문장을 딥러닝 모델에 입력하기 위해서는 토큰화 (Tokenization) 와 인덱싱 (Indexing) 과정을 거쳐야 합니다.

```python
sentence = 'a dog is sitting on the grass'

# 토큰화: 문장을 단어 단위로 분리
tokens = sentence.split()
print(tokens) # 출력: ['a', 'dog', 'is', 'sitting', 'on', 'the', 'grass']

# 인덱싱: 각 단어를 GloVe 사전의 고유 번호(ID)로 변환
ids = [glove.key_to_index[w] for w in tokens]
print(ids) # 출력: [7, 2926, 14, 2995, 13, 0, 4614]

# 벡터 추출: 인덱스를 사용하여 300차원 임베딩 벡터들을 가져옴
vecs = torch.tensor(glove.vectors[ids])
print(vecs.shape) # 출력: (7, 300)
```

#### ◼️ Indexing: 단어의 주소 찾기
* **glove.key_to_index:** 40만 개의 단어가 저장된 거대한 GloVe 사전에서 해당 단어가 몇 번째 칸에 있는지 알려주는 '색인' 역할을 합니다.
* **효과:** 문자열 데이터를 직접 다루는 대신 정수 (Integer) 로 변환함으로써 메모리 효율을 높이고 빠른 조회가 가능해집니다.

#### ◼️ vecs.shape [7, 300]
* **7:** 문장에 포함된 단어의 개수 (Sequence Length) 입니다.
* **300:** 각 단어가 가진 의미의 깊이 (Embedding Dimension) 입니다.
* **결과:** 이제 이 문장은 $7 \times 300$ 크기의 행렬이 되어, LSTM 이나 Transformer 와 같은 신경망 모델에 들어갈 준비가 되었습니다.

### 🟩 Prepare nn.Embedding & LSTM
단어 인덱스를 벡터로 바꿔주는 Embedding Layer와 문장의 정보를 압축하는 LSTM을 정의합니다.

```python
# 사전 학습된 GloVe 가중치를 텐서로 변환
glove_vectors = torch.FloatTensor(glove.vectors)
print(glove_vectors.shape)

# Embedding Layer 정의 (사전 학습된 가중치 주입)
embed = nn.Embedding.from_pretrained(glove_vectors)

# LSTM 모델 정의 (입력: 300차원, 출력: 256차원)
lstm = nn.LSTM(300, 256, batch_first=True)
```

#### ◼️ nn.Embedding.from_pretrained
* **동작:** 우리가 직접 모델을 학습시키는 대신, 이미 검증된 GloVe 의 40만 개 단어 벡터를 모델의 첫 번째 층으로 고정하여 사용합니다.

* **이점:** 모델이 단어의 의미를 처음부터 배울 필요 없이, 300 차원의 풍부한 언어적 관계를 즉시 활용할 수 있게 됩니다.

#### ◼️ nn.LSTM(300, 256)
* **input_size (300):** GloVe 에서 넘어오는 단어 벡터의 크기입니다.

* **hidden_size (256):** 문장을 다 읽었을 때 남기고 싶은 '요약 정보' 의 크기입니다. 우리는 300 차원의 시퀀스 정보를 256 차원의 압축된 문장 벡터로 변환할 것입니다.

* **batch_first=True:** 입력 데이터의 형태를 [배치 크기, 문장 길이, 단어 차원] 순서로 받겠다는 설정입니다. 파이토치에서 데이터를 가장 직관적으로 다룰 수 있는 옵션입니다.

### 🟩 Calculate sentence embedding
이제 정의한 모델에 인덱싱된 문장을 통과시켜 최종적인 문장 벡터를 산출합니다.

```python
# 문장 인덱스를 텐서로 변환 및 배치 차원 추가
x = torch.tensor(ids).unsqueeze(0)
print(x.shape) # 출력: (1, 7) (1개 배치, 7개 단어)

# 임베딩 레이어를 통해 300차원 벡터로 변환
x = embed(x)
print(x.shape) # 출력: (1, 7, 300) (1개 배치, 7개 단어, 각 300차원)

# LSTM 통과 (순차적 문맥 파악)
out, (hidden, cell) = lstm(x)
print(out.shape)    # [1, 7, 256] -> 각 단어 위치마다의 요약 정보
print(hidden.shape) # [1, 1, 256] -> 문장 전체를 다 읽은 최종 요약 정보
print(cell.shape)   # [1, 1, 256] -> LSTM 내부의 장기 기억 소자 (실제 임베딩으론 잘 안 씀)

# 최종 문장 임베딩 벡터 (배치 차워 제거)
print(hidden.squeeze(0).shape) # 출력: (1, 256)
```

#### ◼️ out vs hidden
* **out (Output):** 문장의 각 단어를 읽을 때마다 튀어나오는 Hidden State 들의 집합입니다. 문장의 중간 정보를 모두 담고 있어 Attention 메커니즘 등에서 주로 활용됩니다.

* **hidden (Hidden State):** 문장의 마지막 단어까지 모두 읽고 난 뒤의 최종 상태 입니다. 문장 전체의 맥락이 하나로 응축된 '문장 요약본' 이라고 볼 수 있습니다.

#### ◼️ hidden.squeeze(0)
* **차원 정리:** 모델 연산을 위해 붙여두었던 배치 차원(1) 을 제거하고, 순수한 256 차원의 문장 벡터만 남기는 과정입니다.

* **결과:** 이제 이 256 차원의 벡터는 'a dog is sitting on the grass' 라는 문장의 벡터 됩니다.

---

## 🔊 Speech Representation

### 🟩 Raw Speech Load
음성 데이터를 다루기 위해 PyTorch 의 오디오 전용 라이브러리인 torchaudio 를 사용합니다.

```python
import torchaudio

# 음성 파일 로드 (데이터와 샘플링 레이트 반환)
wav, sr = torchaudio.load('sample_speech.wav')

print(wav.shape, sr) # 출력 예: (1, 268985), 8000
print(268985/8000) # 영상의 길이인 33.6초

# 파형 시각화 (앞부분 일부)
plt.plot(wav[0, :40000].numpy())
plt.show()
```

#### ◼️ wav.shape [1, 268985]
* **1:** 채널 (Channel) 수입니다. 1 은 모노 (Mono), 2 는 스테레오 (Stereo) 를 의미합니다.
* **268985:** 전체 샘플 (Sample) 의 개수입니다. 소리는 연속적인 신호지만, 컴퓨터는 이를 아주 잘게 쪼갠 점(숫자)들의 집합으로 저장합니다.

#### ◼️ Sample Rate (SR)
* **의미:** 1초 당 몇 개의 숫자로 소리를 표현했는지를 나타내는 단위입니다.
* **해석:** SR 이 8,000 이라면 1초에 8,000 개 의 점을 찍어 소리를 기록했다는 뜻입니다.
* **계산법:** 전체 샘플 수를 SR 로 나누면 음성의 실제 길이(초)를 구할 수 있습니다.
  * $\frac{268,985}{8,000} \approx 33.6$ 초

### 🟩 멜 스펙트로그램 변환 및 시각화
```python
# 멜 스펙트로그램 변환기 정의 (샘플링 레이트 기준)
mel_transform = torchaudio.transforms.MelSpectrogram(sr)

# 파형 데이터를 주파수 데이터로 변환
x = mel_transform(wav)
print(x.shape) # 출력 예: [1, 128, 1345] (채널, 주파수 빈, 시간 축)

# 수치적 안정성을 위해 아주 작은 값을 더한 뒤 로그(Log) 연산
x = torch.log(x + 1e-9)

# 시각
plt.imshow(x[0].numpy(), aspect='auto', origin='lower', cmap='magma')
plt.colorbar()
plt.show()
```

#### ◼️ 시각화 결과 해석
* **가로축 (Time):** 시간의 흐름입니다.

* **세로축 (Frequency):** 아래쪽은 저음, 위쪽은 고음 영역을 나타냅니다.

* **색상의 밝기:** 특정 시간대와 주파수에서 에너지가 얼마나 강한지를 나타냅니다. 밝은 선들이 바로 인간의 목소리가 가진 고유한 특징인 포먼트 (Formant) 구조입니다.

---

## 🥱 마치며
지금까지 이미지, 텍스트, 음성 이라는 서로 다른 성질의 데이터를 각각 2,048 차원, 256 차원, 그리고 2차원 행렬 의 형태로 변환하는 과정을 살펴보았습니다. 중요한 것은 이 모든 데이터가 결국 텐서 (Tensor) 라는 공통의 형식을 갖추게 되었다는 점입니다. 이미지는 공간적 특징을, 텍스트는 문맥적 의미를, 음성은 주파수의 에너지를 각각의 숫자 속에 압축하여 담고 있습니다.

다음 포스팅에서는 이렇게 정제된 각 모달리티의 벡터들을 하나의 공통 공간 (Joint Space)에 정렬하여 다양한 작업들을 수행하는 멀티모달 태스크에 대해서 알아보겠습니다. 감사합니다.
