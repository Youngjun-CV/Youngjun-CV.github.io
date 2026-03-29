---
layout: post
title: "Flickr8k 데이터셋 구축 및 이미지-텍스트 임베딩 실습"
date: 2026-03-25 23:46:25 +0900
category: Multimodal
---

## ☺️ 들어가며
지난 포스팅에서는 Joint Embedding 과 Triplet Loss 를 중심으로, 서로 다른 모달리티가 어떻게 하나의 공통된 벡터 공간에서 만날 수 있는지 그 원리를 살펴보았습니다. "강아지 사진"과 "Dog"라는 단어가 수학적으로 '가깝다'고 정의되기 위해 모델이 어떤 손실 함수를 통해 학습되는지 이해하는 시간이었습니다.

하지만 이러한 멀티모달 학습이 가능하려면, 우선 각 데이터를 모델이 이해할 수 있는 숫자 (Tensor) 로 정교하게 번역하는 작업이 선행되어야 합니다. 따라서 이번 포스팅에서는 본격적인 모델 설계에 앞서, PyTorch 프레임워크를 활용해 Flickr8k 데이터셋을 정제하고 각 모달리티를 수치화하는 데이터 전처리 (Data Preprocessing) 과정을 직접 코드로 구현해 보겠습니다.

---

## 📚 0. 데이터셋 이해하기
이론으로만 배웠던 **이미지-텍스트 검색(Image-Text Retrieval)** 을 구현하기 위해 가장 먼저 해야 할 일은 데이터를 준비하는 것입니다. 이번 실습에서는 멀티모달 연구의 교과서라고 불리는 **Flickr8k** 데이터셋을 활용합니다. 이 데이터 셋은 약 **8,000 장** 의 이미지와, 각 이미지에 대해 사람이 직접 작성한 **5개** 의 캡션으로 구성되어 있습니다.

---

## 🛠️ 1. 환경 설정
이미지와 텍스트를 실제로 다루기 위해 PyTorch를 비롯한 라이브러리들을 활용합니다. 특히 **torchvision** 은 이미지 재현을, **gensim** 은 텍스트 임베딩을 실습하는 데 핵심적인 역할을 수행합니다.

```python
!pip install -q gensim
```
gensim 라이브러리를 설치해줍니다. gensim 은 주로 Word2Vec 이나 GloVe 와 같은 **단어 임베딩 (Word Embedding)** 모델을 다룰 때 필수적인 도구입니다.

```python
import torch                        
import torch.nn as nn              # 딥러닝 모델의 기본 구성 요소 (Layer 등)
import torchvision
import torchvision.transforms as T # 이미지 전처리 (Resize, Normalize 등) 
from PIL import Image              # 이미지 파일 로드 및 기본 처리
import numpy as np                 # 수치 연산 및 텐서 변환 보조
import os, pickle                  # 파일 경로 관리 및 데이터 저장/로드
from tqdm import tqdm              # 학습/처리 과정의 진행률 표시

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
```
실습을 진행하는데 필요한 라이브러리들을 임포트하고, 모델의 연산 속도 향상을 위해 **cuda gpu** 를 사용합니다.

---

## 📁 2. 데이터 다운로드 및 데이터 구조 파악
본격적인 실습을 위해 사용할 데이터셋은 **Flickr8k** 입니다. 이 데이터셋은 8,000장의 자연스러운 일상 이미지와, 각 이미지에 대해 인간이 직접 작성한 5개의 텍스트 캡션으로 구성되어 있습니다. 이미지와와 텍스트라는 서로 다른 두 모달리티 사이의 관계를 학습하기에 가장 적합한 입문용 데이터셋입니다.

```python
# 웹상에서 데이터 가져온 다음에 압축 풀기
!wget -q --show-progress https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!wget -q --show-progress https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
!unzip -q Flickr8k_Dataset.zip
!unzip -q Flickr8k_text.zip -d Flickr8k_text
```
웹상에서 데이터를 가져온 뒤 압축을 해제하여 실습환경을 구축해봅시다. 압축 해제가 완료되면, 각 파일이 어떤 용도로 사용되는지 경로를 설정하고 데이터의 규모를 확인합니다.

```python
IMAGE_DIR = 'Flicker8k_Dataset' # 실제 이미지 파일(.jpg)들이 들어있는 폴더
TOKEN_FILE = os.path.join('Flickr8k_text', 'Flickr8k.token.txt') # 이미지-캡션 매핑 정

# 데이터셋 분할 정보를 담은 파일들 (학습/검증/평가용)
TRAIN_SPLIT = os.path.join('Flickr8k_text', 'Flickr_8k.trainImages.txt') # 이미지 6,000 장
DEV_SPLIT = os.path.join('Flickr8k_text', 'Flickr_8k.devImages.txt')     # 이미지 1,000 장
TEST_SPLIT = os.path.join('Flickr8k_text', 'Flickr_8k.testImages.txt')   # 이미지 1,000 장

#전체 이미지 개수 확인
num_images = len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
print(f'Images found: {num_images}')
```
* **Flickr8k.token.txt:** 이번 실습의 중추적인 역할을 합니다. **"이미지파일명.jpg#번호 캡션내용"** 과 같은 형식으로 저장되어 있습니다. 여기서 번호는 한 이미지당 존재하는 5개의 캡션 중 몇 번 째인지(0 ~ 4)를 의미합니다.

* **Flickr8k.trainImages.txt:** 모델 학습 시 사용할 6,000장의 이미지 파일명 목록이 적혀 있습니다.

* **Flickr8k.devImages.txt / testImages.txt:** 학습에 직접 관여하지 않고, 모델의 성능을 검증하거나 최종적으로 평가할 때 사용할 각 1,000장의 이미지 목록입니다. 

---

## 🧹 3. 캡션 파싱
텍스트 파일에서 저장된 원본 데이터는 이미지 하나당 5개의 캡션이 할당된 복잡한 구조로 되어 있어 모델 학습 시 데이터 불균형이나 복잡도를 초래할 수 있습니다. 따라서 이번 실습의 효율성을 위해 각 이미지의 첫 번째 캡션(#0) 만을 추출하여 직관적인 1:1 매칭 구조의 데이터셋을 생성합니다.

```python
from collections import OrderedDict

# "이미지 파일명: 캡션 문장" 형식으로 저장할 순서가 보장된 딕셔너리
image_captions = OrderedDict()
# 실제 이미지 폴더 내에 파일이 존재하지 않아 건너뛴 파일명을 저장할 집합
skipped = set()
```
데이터셋의 순서가 학습/검증/평가 분할 시 중요하게 작용하므로, 파일에서 읽어온 순서를 그대로 유지해야 합니다. 따라서 파이썬의 기본 딕셔너리 대신 삽입된 순서를 기억하는 **OrderedDict** 을 활용합니다.

```python
# 캡션 정보가 담긴 Flickr8k.token.txt 파일 열기
with open(TOKEN_FILE, 'r') as f:
    for line in f:
        line = line.strip() # 줄바꿈 제거
        if not line:
            continue

        # 탭('\t')을 기준으로 파일명#번호와 캡션을 분리
        img_id, caption = line.split('\t', 1)
        # '#'을 기준으로 파일명과 캡션 인덱스를 분리
        filename, cap_idx = img_id.rsplit('#', 1)

        # 첫 번째 캡션(0번)만 선택하여 사용
        if cap_idx == '0':
            # 경로 내에 실제 이미지 파일이 존재하는지 최종 검증
            if os.path.exists(os.path.join(IMAGE_DIR, filename)):
                # 소문자 변환을 통해 텍스트 데이터의 일관성 확보
                image_captions[filename] = caption.strip().lower()
            else:
                skipped.add(filename)
```
이제 텍스트 파일을 한 줄씩 읽어 내려가며 데이터를 정제합니다. 텍스트 파일 내에서 **이미지ID#번호** 와 **캡션 내용** 사이의 구분자인 탭(Tab) 키를 기준으로 데이터를 나눕니다.

```python
# 이미지 파일명과 캡션을 각각 리스트로 변환 (인덱스 접근 용이성 확보)
filenames = list(image_captions.keys())
captions = list(image_captions.values())

print(f'Total image-caption pairs: {len(filenames)}') # 전체 규모 확인

# 누락된 파일이 있다면 출력
if skipped:
    print(f'Skipped (file not found): {len(skipped)}')

# 데이터 파싱 결과 샘플 확인
print(f'\nExample: {filenames[0]}')
print(f'  {captions[0]}')
```
이 과정이 끝나면 약 8,000개의 정제된 이미지-텍스트 쌍이 준비됩니다. 이후 인덱싱을 통해 특정 데이터를 빠르게 불러오거나 슬라이싱하기 편하도록, 딕셔너리에 저장된 값들을 각각 독립된 리스트로 변환하여 관리합니다.

---

## ✂️ 4. 데이터 분할
모델 학습 시 수천 장의 파일명을 직접 문자열로 관리하는 것은 메모리 낭비가 심하며 연산 속도를 저하시킵니다. 따라서 각 데이터를 정수 형태의 인덱스(Index)로 관리하는 맵핑 시스템을 구축하여 메모리 효율과 연산 속도를 극대화합니다.

```python
def load_split(filepath):
    # 특정 분할에 속하는 이미지 파일명 리스트를 불러오는 함수
    with open(filepath, 'r') as f:
        # 데이터 존재 여부 확인 시 검색 속도가 0(1)인 Set 자료형을 사용합니다.
        return set(line.strip() for line in f if line.strip())

# 각 용도별 파일명 집합 로드
train_files = load_split(TRAIN_SPLIT)
val_files = load_split(DEV_SPLIT)
test_files = load_split(TEST_SPLIT)
```
먼저 공식적으로 제공된 분할 파일들로부터 학습, 검증, 평가에 속할 이미지 파일명 리스트를 각각 불러옵니다.

```python
# 파일명을 Key로, 인덱스를 Value로 갖는 매핑 사전 생성
fname_to_idx = {fname: i for i, fname in enumerate(filenames)}

# 전체 데이터를 순회하며 해당 파일이 어느 세트(Train/Val/Test)에 속하는지 확인 후 인덱스로 저장
train_idx = [fname_to_idx[f] for f in filenames if f in train_files]
val_idx = [fname_to_idx[f] for f in filenames if f in val_files]
test_idx = [fname_to_idx[f] for f in filenames if f in test_files]

# 관리의 편의성을 위해 딕셔너리 형태로 통합
splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}

print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
```
이제 파일명을 입력하면 해당 파일이 전체 리스트에서 몇 번째 인덱스인지를 즉시 알려주는 역방향 매핑 딕셔너리를 생성합니다. 이를 통해 문자열 기반의 탐색을 정수 기반의 접근으로 변환할 수 있습니다.

---

## 🚪 5. 이미지 특징 추출
이미지 데이터는 그 자체로 너무 방대하여 멀티모달 모델을 밑바닥부터 학습시키는 것은 연산 자원 측면에서 매우 비효율적입니다. 따라서 이미 수백만 장의 사진을 학습한 ResNet50 모델을 가져와, 이미지의 핵심 정보가 응축된 **특징 벡터(Feature Vector)** 만을 추출하여 사용합니다.

```python
# ImageNet으로 사전 학습된 ResNet50 모델 로드
resnet = torchvision.models.resnet50(weights='DEFAULT')

# 분류(Classification)가 아닌 특징 추출이 목적이므로, 마지막 레이어(FC Layer)를 통과 장치로 변경
resnet.fc = nn.Identity()

resnet = resnet.to(device)
resnet.eval() # 특징 추출을 위해 평가 모드로 설정
```
우리는 1,000개의 클래스로 분류하는 결과값이 아니라, 마지막 레이어 직전의 2048 차원의 벡터 자체가 필요합니다. 이 벡터에는 이미지의 공간적/세멘틱 정보가 고도로 압축되어 있습니다.

```python
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
사전 학습된 모델은 특정 규격의 데이터로 학습되었습니다. 따라서 ResNet이 학습되었던 표준 규격인 224x224 크기로 이미지를 맞추고, 통계적으로 검증된 수치로 정규화 (Normalization) 합니다.

```python
print('Extracting image features...')
image_features = []

with torch.no_grad(): # 특징 추출 시에는 기울기 계산이 필요 없으므로 메모리 절약
    for fname in tqdm(filenames):
        img = Image.open(os.path.join(IMAGE_DIR, fname)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device) # [1, 3, 224, 224]로 배치 차원 추가

        # 모델 통과 후 1차원 벡터로 변환하여 CPU로 복사
        feat = resnet(img_tensor).squeeze(0).cpu() # feature를 뽑아서 바로 사용할 수 있도록 cpu로 넘김
        image_features.append(feat)

# 모든 특징 벡터를 하나의 거대한 텐서로 결합
image_features = torch.stack(image_features)
print(f'Image features shape: {image_features.shape}')
```
이제 준비된 전처리 파이프라인과 모델을 통과시켜 모든 이미지를 2048 차원의 수치 데이터로 변환합니다.

---

## 🔢 6. 텍스트 수치화와 단어 임베딩
컴퓨터는 단어의 내용 자체를 이해하지 못합니다. 따라서 방대한 말뭉치를 학습한 GloVe 임베딩을 사용하여 단어를 300차원의 고밀도 벡터로 변환하고, 단어 간의 '의미적 거리'를 수학적으로 계산 가능한 형태로 재현합니다.

```python
import gensim.downloader as api

# 사전 학습된 300차원의 GloVe 모델 로드 (Wikipedia 데이터 기반)
print('Loading GloVe...')
glove = api.load('glove-wiki-gigaword-300')
print(f'GloVe loaded: {len(glove)} words, {glove.vector_size}d')
```
사전 학습된 300차원의 GloVe 모델을 로드합니다.

```python
from collections import Counter

# 모든 캡션을 순회하며 단어 빈도수 계산
word_counts = Counter()
for cap in captions:
    word_counts.update(cap.split())
```
데이터셋에 등장하는 모든 단어를 임베딩할 수는 없습니다. 따라서 "Counter"를 사용하여 전체 캡션 데이터에서 어떤 단어가 얼마나 자주 등장하는지 전수 조사하고, 우리만의 단어 사전(Vocabulary)을 구축할 준비를 합니다.

```python
# 특별 토큰 설정: 패딩(0)과 모르는 단어(1)
PAD_IDX, UNK_IDX = 0, 1
word2idx = {'<pad>': PAD_IDX, '<unk>': UNK_IDX}

# 임베딩 행렬 초기화 (0번은 제로 벡터, 1번은 랜덤 벡터)
vectors = [np.zeros(300), np.random.randn(300) * 0.01] # 임베딩 벡터 

found, not_found = 0, 0
for word in word_counts:
    if word in glove:
        # GloVe에 단어가 있다면 인덱스 부여 후 벡터 저장
        word2idx[word] = len(word2idx)
        vectors.append(glove[word])
        found += 1
    else:
        # 단어가 없다면 건너뜀 (나중에 <unk>로 처리됨)
        not_found += 1

glove_vectors = torch.FloatTensor(np.stack(vectors))
print(f'Vocab: {len(word2idx)} (GloVe hit: {found}, miss: {not_found})')
print(f'Embedding matrix: {glove_vectors.shape}')
```
단어 사전을 만드는 가장 큰 이유는 단어에 '주소(Index)' 를 부여하기 위함입니다. GloVe에 존재하는 단어는 해당 벡터를 가져오고, 존재하지 않는 단어는 알 수 없음(Unknown) 처리를 하여 모델이 일관된 규격의 입력을 받도록 설계합니다.

---

## 🧹 7. 텍스트 정제 - 패딩과 텐서 변환
우리가 가진 캡션들은 단어의 개수가 모두 다릅니다. 하지만 딥러닝 모델은 배치(Batch) 단위로 데이터를 처리하기 위해 일정한 크기의 입력을 기대합니다. 따라서 모든 문장을 정해진 최대 길이(MAX_LEN)에 맞춰 자르거나 부족한 부분을 빈 값으로 채우는 패딩(Padding) 과정을 거쳐 고정된 크기의 텐서로 변환합니다.

```python
# 문장을 인덱스 리스트로 변환하고 패딩을 적용하는 함수
def caption_to_ids(caption, word2idx, max_len=32):
    tokens = caption.split()

    # 사전에 없는 단어는 UNK_IDX로 대체하며 인덱스 변환
    ids = [word2idx.get(w, UNK_IDX) for w in tokens]

    # 1. 최대 길이만큼 자르기 (Truncation)
    ids = ids[:max_len]
    length = len(ids) # 패딩 넣기 전 실제 문장 길이

    # 2. 부족한 길이를 PAD_IDX로 채우기 (Padding)
    ids = ids + [PAD_IDX] * (max_len - length)

    return ids, length

MAX_LEN = 32
all_ids, all_lens = [], []

for cap in captions:
    ids, length = caption_to_ids(cap, word2idx, MAX_LEN)
    all_ids.append(ids)
    all_lens.append(length)

# 모델 입력에 적합한 PyTorch 텐서(LongTensor)로 변환
caption_ids = torch.tensor(all_ids, dtype=torch.long)
caption_lengths = torch.tensor(all_lens, dtype=torch.long)

# 결과 확인: [전체 데이터 개수, 최대 길이] 형태가 출력되어야 함. (8091, 32)와 (8091,) 이 출력되면 성공
print(f'Caption IDs: {caption_ids.shape}, Lengths: {caption_lengths.shape}')
```

---

## 💾 8. 전처리 데이터 저장
지금까지 진행한 이미지 특징 추출, 단어 사전 구축, 텍스트 패딩은 매번 실습을 시작할 때마다 반복하기에는 연산 비용과 시간이 너무 많이 소모됩니다. 따라서 가공된 모든 데이터와 메타 정보를 PyTorch의 .pt 형식으로 하나의 딕셔너리에 담아 저장함으로써, 다음 단계인 모델 학습 시 즉시 로드하여 사용할 수 있는 환경을 구축합니다.

```python
SAVE_PATH = 'flickr8k_data.pt'

# 전처리된 모든 정보를 하나의 딕셔너리로 통합하여 저장
torch.save({
    'image_features': image_features,                # 추출된 이미지 특징 벡터 (N, 2048)
    'glove_vectors': glove_vectors,                  # GloVe 기반 임베딩 행렬 (vocab, 300)
    'caption_ids': caption_ids,                      # 패딩 완료된 텍스트 인덱스 (N, 32)
    'caption_lengths': caption_lengths,              # 각 문장의 실제 길이 정보 (N,)
    'filenames': filenames,                          # 데이터 매칭용 파일명 리스트
    'captions': captions,                            # 원본 텍스트 데이터
    'word2idx': word2idx,                            # 단어 -> 인덱스 변환 사전
    'idx2word': {v: k for k, v in word2idx.items()}, # 인덱스 -> 단어 역변환 사전
    'splits': splits,                                # Train/Val/Test 인덱스 분할 정보
    'max_len': MAX_LEN                               # 설정된 최대 문장 길이
}, SAVE_PATH)

# 저장된 파일의 크기를 확인하여 정상 저장 여부 체크
size_mb = os.path.getsize(SAVE_PATH) / 1e6
print(f'Saved: {SAVE_PATH} ({size_mb:.1f} MB)')
```

---

## 🫡 마치며: 데이터가 모델의 언어가 되기까지
지금까지 이미지와 텍스트라는 서로 다른 성격의 데이터를 컴퓨터가 이해할 수 있는 형태인 숫자 (Tensor) 로 변환하는 과정을 진행하였습니다. 

단순히 라이브러리를 사용하는 것을 넘어, ResNet 을 통한 시각 정보의 압축, GloVe 를 활용한 언어적 의미 보존, 그리고 효율적인 연산을 위한 인덱싱과 패딩 까지 딥러닝 프로젝트의 가장 기본이면서도 중요한 과정을 마친 셈입니다. 우리가 정성껏 가공한 이 데이터들은 이제 멀티모달 모델에 사용할 준비가 되었습니다.

다음 포스팅에서는 이렇게 저장된 .pt 파일을 불러와 드디어 PyTorch Dataset 과 DataLoader 를 정의하고, 이미지와 텍스트를 동시에 처리하는 멀티모달 모델의 구조를 직접 설계해 보도록 하겠습니다. 긴 전처리 과정을 함께해 주셔서 감사합니다!
