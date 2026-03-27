---
layout: post
title: "Flickr8k 데이터셋 로드 및 전처리 실습"
date: 2026-03-25 23:46:25 +0900
category: Multimodal
---

😗 멀티모달 학습의 시작, 데이터셋 이해하기
이론으로만 배웠던 **이미지-텍스트 검색(Image-Text Retrieval)** 을 구현하기 위해 가장 먼저 해야 할 일은 데이터를 준비하는 것입니다. 이번 실습에서는 멀티모달 연구의 교과서라고 불리는 **Flickr8k** 데이터셋을 활용합니다.

이 데이터 셋은 약 **8,000 장** 의 이미지와, 각 이미지에 대해 사람이 직접 작성한 **5개** 의 캡션으로 구성되어 있습니다.

---

## 🛠️ 1. 환경 설정
먼저 텍스트 데이터를 처리하기 위한 **gensim** 라이브러리를 설치하고, 실습에 사용할 라이브러리들을 불러옵니다.

```python
!pip install -q gensim
```
**gensim** 은 주로 **Word2Vec** 이나 **GloVe** 와 같은 **단어 임베팅 (Word Embedding)** 모델을 다룰 때 필수적인 도구입니다.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os, pickle
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
```
실습을 진행하는데 필요한 라이브러리들을 임포트하고, 연산 속도 향상을 위해 **cuda gp** 를 사용합니다.

---

## 📁  데이터 다운로드 및 데이터 구조 파악

```python
# 웹상에서 데이터 가져온 다음에 압축 풀기
!wget -q --show-progress https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!wget -q --show-progress https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
!unzip -q Flickr8k_Dataset.zip
!unzip -q Flickr8k_text.zip -d Flickr8k_text
```
웹상에서 데이터를 가져온 다음, 압축을 해제합니다.

```python
IMAGE_DIR = 'Flicker8k_Dataset' # 실제 이미지들이 들어있는 폴더
TOKEN_FILE = os.path.join('Flickr8k_text', 'Flickr8k.token.txt') # 형식: 이미지파일명.jpg#번호 캡션내용
# 학습시 사용
TRAIN_SPLIT = os.path.join('Flickr8k_text', 'Flickr_8k.trainImages.txt') # 이미지 6,000 장
DEV_SPLIT = os.path.join('Flickr8k_text', 'Flickr_8k.devImages.txt') # 이미지 1,000 장
TEST_SPLIT = os.path.join('Flickr8k_text', 'Flickr_8k.testImages.txt') # 이미지 1,000 장

num_images = len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
print(f'Images found: {num_images}')
```
**Flickr8k.token.txt:** 파일이 이번 실습의 핵심입니다. 한 이미지당 **5 개** 의 캡션이 `이미지파일명.jpg#번호 캡션내용` 과 같은 형식으로 저장되어 있습니다. 여기서 번호는 한 이미지의 n번째 캡션이라는 뜻입니다. 

Flickr8k 데이터 셋은 학습(Train)용 6,000장, 검증(Validation)용 1,000장 그리고 평가(Test)용 1,000장으로 이루어져 있습니다.

---

## 🧹 캡션 파싱
한 이미지당 5개의 캡션이 있지만, 이번 실습의 효율을 위해 **첫 번째 캡션(#0)** 만 사용하여 **1:1 매칭 구조** 를 만듭니다.

```python
from collections import OrderedDict

# "이미지 파일명: 캡션 문장" 형식으로 저장할 빈 딕셔너리
image_captions = OrderedDict()
# 실제 이미지 파일이 존재하지 않아 건너뛴 파일명을 저장할 집합
skipped = set()
```
데이터셋의 순서가 학습/검증/평가 분할(**Split**) 시 중요하게 작용하므로, 파일에서 읽어온 순서를 그대로 유지해야 합니다. 따라서 파이썬의 기본 딕셔너리와 달리 삽입된 순서를 기억하는 **OrderedDict** 을 불러옵니다.

```python
# 캡션 정보가 담긴 Flickr8k.token.txt 파일 열기
with open(TOKEN_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 한줄마다 넘어가고 #을 기준으로 split을 한다.
        img_id, caption = line.split('\t', 1)
        filename, cap_idx = img_id.rsplit('#', 1)
        if cap_idx == '0':
            if os.path.exists(os.path.join(IMAGE_DIR, filename)):
                image_captions[filename] = caption.strip().lower()
            else:
                skipped.add(filename)
```
텍스트 파일 내에서 `이미지ID#번호` 와 `캡션 내용` 사이의 구분자인 탭(**Tab**) 키를 기준으로 딱 한 번만 나눕니다. 그리고 첫번째 캡션만 사용하기 위해서 `cap_idx == 0` 을 만족하고 해당하는 이미지가 파일에 존재하면 `image_captions` 딕셔너리에 저장합니다. 이 과정이 끝나면 `image_captions` 에는 약 8,000개의 정제된 **이미지-텍스트 쌍** 이 담기게 됩니다.

```python
# 이미지 파일명과 캡션을 각각 다른 리스트에 저장
filenames = list(image_captions.keys())
captions = list(image_captions.values())

print(f'Total image-caption pairs: {len(filenames)}') # Flickr8k의 전체 규모인 약 8,000 개가 나옴
if skipped:
    print(f'Skipped (file not found): {len(skipped)}')
# 0번째 데이터 확인
print(f'\nExample: {filenames[0]}')
print(f'  {captions[0]}')
```
이후 인덱싱을 통해 특정 데이터를 빠르게 불러오거나, Train/Valid/Test 세트로 슬라이싱 편한 형태로 만들기 위해서 `image_captions` 딕셔너리에 저장된 이미지 파일명들과 대응하는 캡션들을 각각 독립된 리스트로 변환합니다. 

---

## ✂️ 데이터 분할
모델 학습 시 문자열(파일명)보다는 숫자(인덱스)로 데이터를 관리하는 것이 메모리와 연산 측면에서 효율적이기 때문에 기존의 데이터를 인덱스 기준으로 수정해보겠습니다.
```python
def load_split(filepath):
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

train_files = load_split(TRAIN_SPLIT)
val_files = load_split(DEV_SPLIT)
test_files = load_split(TEST_SPLIT)
```
특정 분할(Split)에 속하는 이미지 파일명 리스트를 읽어오는 함수입니다. List가 아닌 Set을 사용하는 이유는 데이터 존재 여부를 확인 할 때 검색 속도가 훨씬 빠르기 때문입니다.

```python
fname_to_idx = {fname: i for i, fname in enumerate(filenames)}

train_idx = [fname_to_idx[f] for f in filenames if f in train_files]
val_idx = [fname_to_idx[f] for f in filenames if f in val_files]
test_idx = [fname_to_idx[f] for f in filenames if f in test_files]

splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
```
파일명을 입력하면 해당 파일이 전체 리스트에서 몇번 째 인덱스인지 바로 알려주는 딕셔너리를 만듭니다. 그리고 나서 전체 데이터(`filenames`)를 순회하면서, 해당 파일이 어떤 세트(`train/val/test`)에 속하는지 확인하여 인덱스 번호를 리스트로 만듭니다. 

이 과정이 끝나면 `train_idx` 에는 학습에 사용할 이미지들의 번호가 담기게 됩니다.

---

## 🚪 이미지 특징 추출
멀티모달 모델을 처음부터 밑바닥까지 학습시키는 것은 매우 비효율적입니다. 대신, 이미 수백만 장의 사진을 학습한 ResNet50을 가져와 사용합니다.

```python
resnet = torchvision.models.resnet50(weights='DEFAULT') # Resnet50을 학습된 가중치를 사용하여 부름
resnet.fc = nn.Identity() # 마지막 레이어인 FC layer 제거
resnet = resnet.to(device)
resnet.eval() # 모델을 평가 모드로 설정
```
`resnet.fc = nn.Identity()`로 설정하여 **ResNet** 의 마지막 레이어인 **1,000** 개 클래스 분류기(**FC layer**)를 제거합니다. 우리는 '분류'가 목적이 아니라, 이미지의 특징이 압축된 **2048** 차원의 벡터 자체가 필요하기 때문입니다.

```python
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
**ResNet** 이 학습되었던 규격인 **224x224** 로 크기를 맞추고 데이터를 정규화 합니다.

```python
print('Extracting image features...')
image_features = []
with torch.no_grad():
    for fname in tqdm(filenames):
        img = Image.open(os.path.join(IMAGE_DIR, fname)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        feat = resnet(img_tensor).squeeze(0).cpu() # feature를 뽑아서 바로 사용할 수 있도록 cpu로 넘김
        image_features.append(feat)

image_features = torch.stack(image_features)
print(f'Image features shape: {image_features.shape}')
```
`(1, 3, 224, 224)` $\rightarrow$ **[배치 크기, 채널, 높이, 너비]** 의 4차원 텐서로 맞춰주기 위해 `unsqueeze(0)`을 사용하여 새로운 차원을 추가해주고, `(1, 2048)` 인 2048 차원의 특징 벡터를 순수한 값만 남기기 위해 `squeeze(0)`을 사용하여 불필요한 차원을 제거합니다.

이 진행과정을 거치면 각 이미지는 **2048** 차원의 벡터로 변환됩니다.

---

## 🔢 텍스트 수치화와 단어 임베딩
**GloVe** 를 사용하여 단어들을 벡터화 합니다.

```python
from collections import Counter

word_counts = Counter()
for cap in captions:
    word_counts.update(cap.split())
```
**Counter** 를 사용하여 전체 캡션 데이터에서 어떤 단어가 얼마나 자주 등장하는지 전수 조사합니다.

```python
PAD_IDX, UNK_IDX = 0, 1
word2idx = {'<pad>': PAD_IDX, '<unk>': UNK_IDX} # "단어: 인덱스" 형태의 딕셔너리
vectors = [np.zeros(300), np.random.randn(300) * 0.01] # 임베딩 벡터 

found, not_found = 0, 0
for word in word_counts:
    if word in glove:
        word2idx[word] = len(word2idx)
        vectors.append(glove[word])
        found += 1
    else:
        not_found += 1

glove_vectors = torch.FloatTensor(np.stack(vectors))
print(f'Vocab: {len(word2idx)} (GloVe hit: {found}, miss: {not_found})')
print(f'Embedding matrix: {glove_vectors.shape}')
```
**word2idx** 가 단어에 고유한 '인덱스 번호' 를 부여하면, 모델은 그 번호를 주소 삼아 **vectors** (임베딩 행렬) 에서 단어의 진짜 의미인 '벡터 값' 을 찾아내어 계산을 시작합니다.

---

## 캡선 -> TOKEN ID
def caption_to_ids(caption, word2idx, max_len=32):
    tokens = caption.split()
    ids = [word2idx.get(w, UNK_IDX) for w in tokens]
    ids = ids[:max_len]
    length = len(ids)
    ids = ids + [PAD_IDX] * (max_len - length)
    return ids, length

MAX_LEN = 32
all_ids, all_lens = [], []
for cap in captions:
    ids, length = caption_to_ids(cap, word2idx, MAX_LEN)
    all_ids.append(ids)
    all_lens.append(length)

caption_ids = torch.tensor(all_ids, dtype=torch.long)
caption_lengths = torch.tensor(all_lens, dtype=torch.long)
print(f'Caption IDs: {caption_ids.shape}, Lengths: {caption_lengths.shape}')

---

## ??? 저장
SAVE_PATH = 'flickr8k_data.pt'

torch.save({
    'image_features': image_features,     # (N, 2048)
    'glove_vectors': glove_vectors,        # (vocab, 300)
    'caption_ids': caption_ids,            # (N, 32)
    'caption_lengths': caption_lengths,    # (N,) 이밎
    'filenames': filenames,
    'captions': captions,
    'word2idx': word2idx,
    'idx2word': {v: k for k, v in word2idx.items()},
    'splits': splits,
    'max_len': MAX_LEN
}, SAVE_PATH)

size_mb = os.path.getsize(SAVE_PATH) / 1e6
print(f'Saved: {SAVE_PATH} ({size_mb:.1f} MB)')


















