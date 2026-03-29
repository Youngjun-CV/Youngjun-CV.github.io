---
layout: post
title: "triplet loss 실습"
date: 2026-03-25 23:58:25 +0900
category: Multimodal
---

## 😊 들어가며
지난 포스팅에서는 이미지와 텍스트라는 서로 다른 도메인의 데이터를 각각 어떻게 수치화하는지, 그리고 이를 하나의 공통 공간으로 모으는 Joint Embedding 의 원리에 대해 살펴보았습니다. 특히 데이터 간의 거리를 조절하여 의미적 유사성을 학습시키는 Triplet Loss 는 멀티모달 모델의 핵심입니다.

이제 이론으로만 접했던 개념들을 실제 코드로 구현해 볼 시간입니다. 이번 실습에서는 전처리 과정에서 만들어둔 flickr8k_data.pt 파일을 활용하여, 사용자가 입력한 문장에 가장 어울리는 이미지를 찾아내는 이미지-텍스트 검색 시스템을 구축해 보겠습니다.

---

## 🛠️ 0. 환경 설정 및 라이브러리 임포트
실무적인 딥러닝 실험을 위해 필요한 도구들을 불러옵니다. 대규모 행렬 연산이 빈번한 멀티모달 학습 특성상 GPU (CUDA) 가용 여부를 체크하는 것은 필수적인 첫 번째 단계입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle, os, copy
from PIL import Image
import matplotlib.pyplot as plt

# 딥러닝 연산을 가속화할 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
```

---

## 📂 1. 데이터 로딩 
이전 Unimodal Representation 실습에서 정성껏 가공하여 저장해두었던 .pt 파일을 불러옵니다. 이 파일은 이미지의 시각적 특징부터 텍스트의 언어적 의미까지, 모델 학습에 필요한 모든 정보가 응집된 통합 데이터베이스입니다.

```python
# Flickr8k 이미지 데이터셋 다운로드 (검증용 이미지 로드 시 필요)
!wget -q --show-progress https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!unzip -q Flickr8k_Dataset.zip

IMAGE_DIR = 'Flickr8k_Dataset'
```
학습 자체는 추출된 특징 벡터(img_feats)를 사용하지만, 나중에 모델이 검색한 결과를 눈으로 확인하기 위해서는 실제 이미지 파일이 필요합니다. 따라서 원본 Flickr8k 데이터셋을 다운로드하고 경로를 설정해 줍니다.

```python
# 전처리된 통합 데이터 로드 (CPU 메모리로 먼저 로드)
data = torch.load('flickr8k_data.pt', map_location='cpu')

# 딕셔너리에서 각 요소 추출
img_feats = data['image_features']        # [N, 2048] 크기의 이미지 특징 벡터
glove_vectors = data['glove_vectors']     # [Vocab, 300] 크기의 사전 학습된 GloVe
caption_ids = data['caption_ids']         # [N, 32] 크기의 패딩 완료된 텍스트
caption_lengths = data['caption_lengths'] # [N] 각 캡션의 실제 길이
filenames = data['filenames']             # 이미지 파일명 리스트
captions = data['captions']               # 원본 텍스트 리스트
# splits(인덱스 분할), word2idx 등 메타 정보 포함
```
torch.load를 통해 이전에 저장한 통합 데이터를 불러옵니다. 이때 map_location='cpu' 옵션을 사용하면 데이터가 저장된 환경에 관계없이 안전하게 현재 시스템의 메모리로 로드할 수 있습니다. 추출된 각 요소들은 **공유 임베딩 공간** 을 구성하는 핵심 자산이 됩니다.

```python
# 로드된 데이터 규모 및 형태 확인
print(f'Image features : {img_feats.shape}')
print(f'GloVe vectors  : {glove_vectors.shape}')
print(f'Caption IDs    : {caption_ids.shape}')
print(f'Splits - train: {len(data["splits"]["train"])}, val: {len(data["splits"]["val"])}, test: {len(data["splits"]["test"])}')
```
이미지 특징 벡터 **(8091, 2048)** 는 ResNet50 이 요약한 고차원 시각 정보로 이후 ImageEncoder 의 첫 입력 차원이 되며, GloVe 임베딩 행렬 **(4280, 300)** 은 TextEncoder 가 참조할 4,280개 단어의 300차원 '의미 사전' 역할을 합니다. 또한 캡션 인덱스 데이터 **(8091, 32)** 는 모든 문장을 LSTM 연산에 최적화된 32개 단어 길이로 정형화한 상태를 의미하며, 마지막으로 데이터 분할 **(6000, 1000, 1000)** 은 학습, 검증, 평가 세트를 엄격히 분리하여 모델의 객관적인 일반화 성능을 측정하기 위한 장치입니다.

---

## 🏗️ 2. Dataset & DataLoader
모델이 학습에 집중할 수 있도록 원본 데이터를 유기적으로 묶어 전달하는 과정이 필요합니다. 따라서 PyTorch의 Dataset 클래스를 상속받아 이미지 특징과 텍스트 인덱스를 쌍으로 반환하는 맞춤형 데이터셋을 정의하고, 이를 DataLoader를 통해 배치(Batch) 단위로 공급하는 과정을 거칩니다.

```python
class Flickr8kDataset(Dataset):
    # 데이터셋 초기화 함수, 전체 데이터 중 지정된 인덱스에 해당하는 데이터만 슬라이싱하여 저장
    #indices: 학습/검증/평가 분할에 해당하는 인덱스 리스트 (e.g., train_idx)
    def __init__(self, img_feats, caption_ids, caption_lengths, indices):
        self.img_feats = img_feats[indices]
        self.caption_ids = caption_ids[indices]
        self.caption_lengths = caption_lengths[indices]

    # 데이터셋의 전체 개수 반환
    def __len__(self):
        return len(self.img_feats)

    # 특정 인덱스(idx)의 데이터를 (이미지 특징, 캡션 ID, 실제 길이) 튜플로 반환
    def __getitem__(self, idx):
        return self.img_feats[idx], self.caption_ids[idx], self.caption_lengths[idx]
```
**Flickr8kDataset** 클래스 는 흩어져 있는 이미지와 텍스트 정보를 하나의 '멀티모달 쌍 (Pair) '으로 묶어주는 역할을 합니다. 특히 전체 데이터 중 필요한 부분만 골라 담는 필터 기능을 수행하며, 모델이 데이터를 요청할 때마다 정해진 규격에 맞춰 데이터를 차례대로 전하는 인터페이스 역할을 담당합니다.

```python
# 학습용 및 평가용 데이터셋 생성
# data['splits']에 저장된 인덱스 정보를 활용하여 데이터를 분할합니다.
train_ds = Flickr8kDataset(img_feats, caption_ids, caption_lengths, data['splits']['train'])
test_ds = Flickr8kDataset(img_feats, caption_ids, caption_lengths, data['splits']['test'])

# 미니배치(Mini-batch) 설정을 통한 데이터 로더 구성
BATCH_SIZE = 256

# 학습 로더: 데이터 셔플(Shuffle)을 통해 모델의 일반화 성능을 높입니다.
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 테스트 로더: 평가 시에는 전체 데이터를 한 번에 확인하기 위해 배치 사이즈를 전체 크기로 설정합니다.
test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
```
이 부분은 정의한 데이터셋을 바탕으로 실제 학습 흐름을 만드는 단계입니다. BATCH_SIZE 를 통해 데이터를 적절한 묶음으로 나누어 전달하며, 학습 시에는 순서를 섞어 모델의 편향을 방지합니다. 평가 시에는 모든 데이터를 한꺼번에 로드하여 전체적인 유사도 관계를 효율적으로 측정할 수 있도록 구성했습니다.

---

## 🧠 3. 이미지와 텍스트 인코딩
이미지 특징 벡터와 텍스트 시퀀스는 각기 다른 차원과 분포를 가지고 있습니다. 따라서 각 모달리티를 전용 인코더를 통해 처리한 뒤, 프로젝션(Projection) 레이어를 거쳐 동일한 차원의 공통 임베딩 공간으로 투영합니다.

```python
class ImageEncoder(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=200, projection='fc'):
        super().__init__()
        # 'fc'일 경우 단일 선형 레이어를, 'mlp'일 경우 2층 구조와 정규화(BatchNorm)를 사용합니다.
        if projection == 'fc':
            self.proj = nn.Linear(input_dim, embed_dim)
        else: # mlp 구조
            self.proj = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, embed_dim)
            )

    def forward(self, img_feat):
        # L2 정규화를 통해 벡터를 단위 구(Unit Sphere) 위로 매핑하여 유사도 계산의 안정성을 확보합니다.
        return F.normalize(self.proj(img_feat), dim=-1)
```
ImageEncoder 클래스 는 ResNet50 에서 추출된 2,048차원의 시각 특징을 우리가 목표로 하는 200차원 공통 공간으로 변환합니다. 단순히 차원만 줄이는 것이 아니라, 비선형적인 구조 (MLP) 를 선택할 수 있게 하여 더 복잡한 시각 정보를 학습할 수 있도록 설계되었습니다. 마지막 단계에서 L2 정규화 를 수행하여 모든 이미지 벡터를 동일한 기준 위에서 비교 가능하게 만듭니다.

```python
class TextEncoder(nn.Module):
    def __init__(self, glove_vectors, hidden_dim=512, embed_dim=200, projection='fc'):
        super().__init__()
        # 사전 학습된 GloVe 가중치를 고정(Freeze)하여 임베딩 레이어 생성
        self.embed = nn.Embedding.from_pretrained(glove_vectors)
        # 문장의 순차적 맥락을 파악하기 위해 LSTM 사용
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True)
        
        if projection == 'fc':
            self.proj = nn.Linear(hidden_dim, embed_dim)
        else: # mlp 구조
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, embed_dim)
            )

    def forward(self, caption_ids, lengths):
        x = self.embed(caption_ids) # 단어 인덱스를 300차원 벡터로 치환
        # 패딩을 무시하고 실제 문장 길이만큼만 연산하여 효율성 향상
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed) # 최종 은닉 상태(Hidden State) 추출
        
        # 마지막 시점의 은닉 상태를 공통 공간으로 투영 후 L2 정규화 수행
        return F.normalize(self.proj(h.squeeze(0)), dim=-1)
```
TextEncoder 클래스 는 문장 데이터를 받아 의미적인 맥락을 파악하고 이미지와 같은 차원으로 정렬합니다. 사전 학습된 GloVe 임베딩으로 단어의 기초 의미를 가져오고, LSTM 을 통해 문장 전체의 흐름을 요약합니다. 특히 가변적인 문장 길이를 고려한 pack_padded_sequence 기법을 사용하여 의미 없는 패딩 영역을 제외한 실제 정보만을 정교하게 추출하여 공통 공간으로 투영합니다.

---

## 🏗️ 4. 통합 모델 정의: RetrievalModel
개별적으로 설계한 인코더들은 각각의 데이터를 처리하는 전문 부품입니다. 따라서 RetrievalModel 클래스를 통해 이미지와 텍스트 인코더를 하나로 통합하고, 입력된 데이터를 동일한 차원의 임베딩 벡터 쌍으로 동시에 반환하는 전체 파이프라인을 완성합니다

```python
class RetrievalModel(nn.Module):
    def __init__(self, glove_vectors, embed_dim=200, projection='fc'):
        """
        통합 모델 초기화 함수
        glove_vectors: 텍스트 인코더에 주입할 사전 학습된 임베딩 행렬
        embed_dim: 공통 공간(Shared Space)의 차원 크기
        projection: 사용할 프로젝션 방식 ('fc' 또는 'mlp')
        """
        super().__init__()
        # 이미지 특징을 공통 공간으로 투영할 이미지 인코더 선언
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, projection=projection)
        
        # 텍스트 시퀀스를 공통 공간으로 투영할 텍스트 인코더 선언
        self.text_encoder = TextEncoder(glove_vectors, embed_dim=embed_dim, projection=projection)

    def forward(self, img_feat, caption_ids, lengths):
        """
        순전파(Forward) 함수: 이미지와 텍스트를 동시에 입력받아 공통 공간의 벡터를 반환
        """
        # 1. 이미지 특징을 200차원 공통 임베딩으로 변환
        img_emb = self.image_encoder(img_feat)
        
        # 2. 텍스트 인덱스를 200차원 공통 임베딩으로 변환
        txt_emb = self.text_encoder(caption_ids, lengths)
        
        # 3. 동일한 차원 위에서 비교 가능한 두 임베딩 벡터를 반환
        return img_emb, txt_emb
```
RetrievalModel 클래스 는 이미지와 텍스트라는 두 개의 독립적인 타워를 거느린 '본부' 역할을 수행합니다. 각 인코더가 생성한 벡터들이 반드시 embed_dim 이라는 동일한 차원 위에서 만날 수 있도록 조율하며, 최종적으로 이미지와 텍스트가 서로의 유사도를 계산할 수 있는 공통 공간 (Shared Space) 의 좌표를 동시에 출력합니다. 이를 통해 각기 다른 형태의 데이터가 하나의 시스템 안에서 유기적으로 비교될 수 있는 환경이 완성됩니다.

---

## ⚖️ 5. Triplet Loss (Hard Negative) & Evaluation
모델이 이미지와 텍스트의 정답 쌍을 가깝게 배치하도록 학습시키기 위해서는 명확한 기준이 필요합니다. 따라서 배치 내에서 가장 헷갈리는 오답(Hard Negative)을 찾아내어 정답과의 차이를 벌리는 triplet_loss_hard_negative 함수와, 이를 Recall@K 지표로 검증하는 평가 함수를 구현합니다.

```python
def triplet_loss_hard_negative(img_emb, txt_emb, margin=0.2):
    # 1. 유사도 행렬 계산: 모든 이미지와 모든 텍스트 사이의 내적(Cosine Similarity) 수행
    # [Batch, Batch] 크기의 행렬이 생성됨
    sim_matrix = img_emb @ txt_emb.t()
    
    # 2. 정답 유사도(Positive Similarity): 행렬의 대각 성분이 실제 정답 쌍의 유사도임
    positive_sim = sim_matrix.diag()
    
    # 3. 마스크 생성: 자기 자신(정답)을 제외하고 오답들 중에서만 비교하기 위함
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    
    # 4. Image-to-Text Hard Negative: 각 이미지에 대해 가장 정답처럼 보이는(유사도가 높은) 오답 추출
    sim_i2t = sim_matrix.masked_fill(mask, -float('inf')) # 정답 위치는 무한대 음수로 가림
    loss_i2t = F.relu(sim_i2t.max(dim=1).values - positive_sim + margin)
    
    # 5. Text-to-Image Hard Negative: 각 텍스트에 대해 가장 유사한 이미지 오답 추출
    sim_t2i = sim_matrix.t().masked_fill(mask, -float('inf'))
    loss_t2i = F.relu(sim_t2i.max(dim=1).values - positive_sim + margin)
    
    # 두 방향의 손실 값을 평균 내어 최종 손실값 반환
    return (loss_i2t.mean() + loss_t2i.mean()) / 2
```
triplet_loss_hard_negative 함수 는 배치 내의 모든 조합을 한꺼번에 비교하는 효율적인 손실 함수입니다. 단순히 정답과의 거리만 줄이는 것이 아니라, 배치 내에서 모델이 가장 정답이라고 착각하기 쉬운 '가장 어려운 오답 (Hard Negative) '을 찾아내어 집중적으로 학습합니다. 이를 통해 모델은 모호한 데이터 사이에서도 정답을 골라내는 강력한 변별력을 갖게 됩니다.

```python
@torch.no_grad() # 평가 시에는 기울기 계산을 하지 않음
def evaluate_model(model, dataloader, device):
    model.eval()
    # 전체 테스트 데이터를 인코딩하여 임베딩 추출
    for img_feat, cap_ids, cap_len in dataloader:
        img_emb, txt_emb = model(
            img_feat.to(device), cap_ids.to(device), cap_len.to(device)
        )
    
    # CPU로 옮겨 유사도 행렬 계산
    img_emb, txt_emb = img_emb.cpu(), txt_emb.cpu()
    sim = img_emb @ txt_emb.t()
    
    results = {}
    # I->T (이미지로 텍스트 찾기)와 T->I (텍스트로 이미지 찾기) 양방향 평가
    for direction, s in [('I→T', sim), ('T→I', sim.t())]:
        ranks = []
        for i in range(s.size(0)):
            # 유사도 순으로 내림차순 정렬했을 때 실제 정답(i)이 몇 번째에 있는지 확인
            rank = (s[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        
        ranks = np.array(ranks)
        # 상위 K개 안에 정답이 포함된 비율 계산
        results[direction] = {f'R@{k}': (ranks < k).mean() * 100 for k in [1, 5, 10]}
    
    # 결과 출력
    for d in ['I→T', 'T→I']:
        print(f'{d}: ' + ', '.join(f'{k}={v:.1f}' for k, v in results[d].items()))
    return results
```
evaluate_model 함수 는 학습된 모델의 객관적인 성적표인 Recall@K 를 산출합니다. 이미지로 텍스트를 찾는 방향과 텍스트로 이미지를 찾는 양방향 검색을 모두 수행하며, 검색 결과 상위 1개, 5개, 10개 안에 실제 정답이 포함되었는지를 퍼센트 (%) 로 계산합니다. 이 수치가 높을수록 모델이 공통 공간 (Shared Space) 에 데이터들을 의미적으로 잘 정렬했다는 증거가 됩니다.

---

## 🎨 6. Embedding 추출 및 시각화 함수
모델의 성능을 Recall@K 라는 숫자로만 확인하는 것보다, 실제 쿼리에 대해 어떤 이미지를 검색해 내는지 시각적으로 확인하는 것이 모델의 특성을 파악하는 데 훨씬 효과적입니다. 따라서 전체 데이터셋의 임베딩을 미리 추출해 두고, 사용자의 텍스트 입력에 대해 유사도가 높은 상위 $K$ 개의 이미지를 출력하는 시각화 시스템을 구축합니다.

```python
@torch.no_grad()
def get_all_embeddings(model):
    """전체 데이터에 대한 embedding 추출 (시각화 및 대용량 검색용)"""
    model.eval()
    # 전체 데이터를 순차적으로 처리하기 위한 데이터로더 생성
    all_loader = DataLoader(
        Flickr8kDataset(img_feats, caption_ids, caption_lengths, list(range(len(img_feats)))),
        batch_size=256, shuffle=False
    )
    
    all_img, all_txt = [], []
    for img_feat, cap_ids, cap_len in all_loader:
        # 모델을 통해 공통 공간(Shared Space)의 벡터 추출
        img_emb, txt_emb = model(
            img_feat.to(device), cap_ids.to(device), cap_len.to(device)
        )
        # 메모리 효율을 위해 결과를 CPU로 옮겨 리스트에 저장
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
        
    # 리스트에 담긴 배치 단위 텐서들을 하나로 합쳐서 반환
    return torch.cat(all_img), torch.cat(all_txt)
```
get_all_embeddings 함수 는 검색 대상이 될 모든 이미지와 텍스트를 공통 공간 (Shared Space) 의 벡터로 변환하여 보관하는 역할을 합니다. 마치 도서관의 모든 책에 고유한 번호표를 붙여 서가에 배치하는 과정과 같으며, 나중에 사용자가 질문을 던졌을 때 매번 모든 데이터를 다시 연산할 필요 없이 미리 저장된 벡터들 사이의 유사도만 계산하면 되도록 효율성을 극대화합니다.

```python
def show_text_to_image(query_caption, model, all_img_embs, top_k=5):
    """사용자 쿼리 문장에 대해 가장 유사한 이미지를 시각화"""
    model.eval()
    word2idx = meta['word2idx']
    
    # 1. 텍스트 전처리: 소문자 변환 후 토큰화 및 인덱스 변환
    tokens = query_caption.lower().split()
    ids = [word2idx.get(w, 1) for w in tokens] # 사전에 없으면 <UNK>(1)로 처리
    ids = ids[:32] + [0] * max(0, 32 - len(ids)) # 길이를 32로 맞춤 (Padding)
    length = min(len(tokens), 32) # 실제 유효 길이 계산

    # 2. 텐서 변환 및 장치(GPU) 이동
    cap_tensor = torch.tensor([ids], dtype=torch.long).to(device)
    len_tensor = torch.tensor([length], dtype=torch.long).to(device)

    # 3. 텍스트 인코딩: 쿼리 문장을 공통 공간 벡터로 변환
    with torch.no_grad():
        query_emb = model.text_encoder(cap_tensor, len_tensor).cpu()

    # 4. 유사도 계산: 쿼리 벡터와 모든 이미지 벡터들 사이의 코사인 유사도 연산
    sims = (query_emb @ all_img_embs.t()).squeeze(0)
    # 유사도 점수가 높은 순서대로 상위 K개의 인덱스 추출
    top_idx = sims.argsort(descending=True)[:top_k]

    # 5. 결과 시각화
    print(f'Query: "{query_caption}"')
    fig, axes = plt.subplots(1, top_k, figsize=(3 * top_k, 3))
    for ax, idx in zip(axes, top_idx):
        # 파일명을 참조하여 실제 이미지 로드
        img = Image.open(os.path.join(IMAGE_DIR, filenames[idx]))
        ax.imshow(img)
        # 상단에 계산된 유사도 점수(sim) 표시
        ax.set_title(f'sim: {sims[idx]:.3f}', fontsize=10)
        ax.axis('off') # 축 정보 숨김
    plt.tight_layout()
    plt.show()
```
show_text_to_image 함수 는 실제 서비스처럼 사용자가 입력한 자유로운 문장을 바탕으로 이미지를 검색해 줍니다. 텍스트 인코더를 통해 실시간으로 쿼리 문장을 벡터화하고, 앞서 구축한 이미지 데이터베이스와 코사인 유사도 (Cosine Similarity) 를 비교합니다. 점수가 높은 순서대로 실제 이미지 파일을 로드하여 출력함으로써, 모델이 문장 속의 핵심 단어와 상황을 얼마나 정확하게 시각적 정보와 연결하고 있는지 직관적으로 보여줍니다.

---

## 🏃 7. 모델 학습: Training Loop
이제 준비된 데이터 로더와 손실 함수를 사용하여 모델의 가중치를 업데이트합니다. 따라서 train_model 함수를 통해 정해진 에포크(Epoch) 동안 반복 학습을 수행하며, 매 걸음마다 기울기(Gradient)를 계산하여 이미지와 텍스트가 공통 공간의 최적의 위치에 정렬되도록 최적화합니다.

```python
def train_model(model, num_epochs=30, lr=5e-4, margin=0.2):
    # 1. 최적화 도구(Optimizer) 설정: 가중치를 효율적으로 업데이트하는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in range(num_epochs):
        model.train() # 모델을 학습 모드로 전환 (Dropout, BatchNorm 활성화)
        epoch_loss = 0
        
        for img_feat, cap_ids, cap_len in train_loader:
            # 2. 순전파(Forward): 이미지와 텍스트를 공통 공간의 벡터로 변환
            img_emb, txt_emb = model(
                img_feat.to(device), cap_ids.to(device), cap_len.to(device)
            )
            
            # 3. 손실 계산: Hard Negative 기반의 Triplet Loss 구하기
            loss = triplet_loss_hard_negative(img_emb, txt_emb, margin)
            
            # 4. 역전파(Backward) 및 최적화:
            optimizer.zero_grad() # 이전 루프의 기울기 초기화
            loss.backward()      # 현재 손실에 대한 기울기 계산
            optimizer.step()      # 계산된 기울기 방향으로 모델 가중치 업데이트
            
            epoch_loss += loss.item()
        
        # 에포크당 평균 손실 기록
        train_losses.append(epoch_loss / len(train_loader))

        # 특정 주기마다 학습 진행 상황 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:2d}/{num_epochs}  train_loss: {train_losses[-1]:.4f}')

    return train_losses
```
train_model 함수 는 설계된 신경망이 데이터로부터 스스로 배울 수 있도록 반복적인 연산을 관리합니다. 이미지와 텍스트를 입력받아 각각의 임베딩을 생성한 뒤, 앞서 정의한 Triplet Loss 를 통해 발생한 오차를 역전파 (Backpropagation) 시켜 모델 내부의 수많은 가중치를 수정합니다. 특히 Adam 최적화 알고리즘을 사용하여 복잡한 손실 함수 평면에서도 안정적으로 최솟값을 찾아가며, 각 에포크 (Epoch) 마다 손실 값을 기록하여 학습이 정상적으로 수렴하고 있는지 모니터링할 수 있게 해줍니다.

---

## 🧪 8. 실험 1: 초기화 상태 (학습 전)
본격적인 학습에 앞서 FC projection 구조를 가진 모델을 생성하고, 아무런 학습이 이루어지지 않은 랜덤 가중치 상태에서의 성능을 측정합니다. 이를 통해 모델이 학습을 통해 점진적으로 성능이 향상되는 과정을 확인하기 위한 대조군 (Control Group) 을 설정하고, 모델 내부의 파라미터 분포를 파악합니다.

```python
EMBED_DIM = 200

# FC 기반의 통합 모델 생성 및 GPU 이동
model_fc = RetrievalModel(glove_vectors, embed_dim=EMBED_DIM, projection='fc').to(device)

# 파라미터를 LSTM / Image Projection / Text Projection 으로 묶어서 확인하는 함수
def count_params(model):
    groups = {'LSTM': 0, 'Image Projection': 0, 'Text Projection': 0, 'Embedding (frozen)': 0}
    for name, param in model.named_parameters():
        n = param.numel()
        if 'lstm' in name:
            groups['LSTM'] += n
        elif 'image_encoder.proj' in name:
            groups['Image Projection'] += n
        elif 'text_encoder.proj' in name:
            groups['Text Projection'] += n
        elif 'embed' in name:
            groups['Embedding (frozen)'] += n
            
    trainable = 0
    for group, n in groups.items():
        frozen = 'frozen' in group
        status = 'FROZEN' if frozen else 'TRAIN'
        print(f'  {status:6s}  {group:25s}  {n:>10,}')
        if not frozen:
            trainable += n
    print(f'         {"Total trainable":25s}  {trainable:>10,}')

# 모델 구조 및 파라미터 수 출력
count_params(model_fc)

print('\n=== 학습 전 (랜덤 weight) ===')
# 학습 전 성능 평가 (Recall@K)
evaluate_model(model_fc, test_loader, device);

# 학습 전 임베딩 추출 및 시각화 테스트
all_img_embs, all_txt_embs = get_all_embeddings(model_fc)
show_text_to_image('a dog running on the beach', model_fc, all_img_embs)
show_text_to_image('a child playing in the snow', model_fc, all_img_embs)
```
count_params 함수 는 모델 내부의 연산 장치들이 각각 어느 정도의 규모를 가지고 있는지 수치화해 줍니다. 특히 Embedding (frozen) 영역은 사전 학습된 GloVe 를 그대로 사용하므로 학습 대상에서 제외되며, 실제로는 LSTM 과 각 Projection 레이어들의 가중치만이 학습을 통해 업데이트됨을 알 수 있습니다.

학습 전의 결과는 모델이 아직 이미지와 텍스트 사이의 연관성을 전혀 배우지 못한 상태이므로, Recall@K 수치가 매우 낮게 나오며 시각화 결과 또한 입력한 쿼리와 무관한 엉뚱한 이미지들을 보여줍니다. 이는 모델의 가중치가 무작위 (Random) 로 설정되어 공통 공간 (Shared Space) 에서 데이터들이 아무런 의미 없이 흩어져 있기 때문입니다.

---
## 🧪 9. 실험 2: FC Projection (Linear)
가장 단순한 형태의 선형 투영 (Linear Projection) 층을 사용하여 이미지와 텍스트를 공통 공간으로 보냅니다. 따라서 projection='fc' 설정을 통해 단일 선형 레이어만을 학습시키며, 선형적인 변환만으로도 모델이 이미지와 텍스트 사이의 의미적 유사성을 어느 정도 파악할 수 있는지 확인합니다.

```python
# FC 기반의 모델 생성 및 학습 진행
model_fc = RetrievalModel(glove_vectors, embed_dim=EMBED_DIM, projection='fc').to(device)

print('--- FC Projection 학습 ---')
# 앞서 정의한 학습 루프를 통해 30 에포크(Epoch) 동안 최적화 수행
fc_train_losses = train_model(model_fc, num_epochs=30)

print('\n=== FC Projection 학습 후 ===')
# 학습 완료 후 성능 평가 (Recall@K 확인)
evaluate_model(model_fc, test_loader, device);

# 실제 쿼리를 입력하여 검색 결과가 어떻게 개선되었는지 확인
all_img_embs, all_txt_embs = get_all_embeddings(model_fc)
show_text_to_image('a dog running on the beach', model_fc, all_img_embs)
show_text_to_image('a child playing in the snow', model_fc, all_img_embs)
```
FC Projection 모델 은 이미지의 2,048차원 특징과 텍스트의 512차원 은닉 상태를 각각 200차원의 공통 공간 (Shared Space) 으로 매핑합니다. 학습이 진행됨에 따라 Triplet Loss 가 줄어들며, 랜덤 가중치 상태일 때와 비교하여 Recall@K 수치가 비약적으로 상승하는 것을 볼 수 있습니다. 시각화 결과 또한 이제는 쿼리에 등장하는 핵심 키워드(dog, beach, child 등)를 인식하여 그에 걸맞은 이미지를 상위권으로 검색해 내기 시작합니다.

단순한 선형 변환만으로도 두 데이터 사이의 기본적인 정렬이 가능하다는 점이 이 실험의 핵심입니다.

---

## 🧪 10. 실험 3: MLP Projection (Non-linear)
단순한 선형 레이어 대신, 비선형 활성화 함수 (ReLU) 와 배치 정규화 (BatchNorm) 가 포함된 MLP (Multi-Layer Perceptron) 구조를 사용합니다. 따라서 projection='mlp' 설정을 통해 더 깊고 복잡한 투영 층을 구성하며, 복잡한 비선형 관계를 학습함으로써 공통 공간 (Shared Space) 에서 이미지와 텍스트 벡터를 더욱 정교하게 분리하고 정렬할 수 있는지 확인합니다.

```python
# MLP 기반의 고도화된 모델 생성 및 파라미터 확인
model_mlp = RetrievalModel(glove_vectors, embed_dim=EMBED_DIM, projection='mlp').to(device)
count_params(model_mlp)

print('\n--- MLP Projection 학습 ---')
# 동일하게 30 에포크(Epoch) 동안 학습 진행
mlp_train_losses = train_model(model_mlp, num_epochs=30)

print('\n=== MLP Projection 학습 후 ===')
# 학습 완료 후 최종 성능 평가 (Recall@K 확인)
evaluate_model(model_mlp, test_loader, device);

# 시각화를 통해 검색 품질의 미세한 변화 관찰
all_img_embs, all_txt_embs = get_all_embeddings(model_mlp)
show_text_to_image('a dog running on the beach', model_mlp, all_img_embs)
show_text_to_image('a child playing in the snow', model_mlp, all_img_embs)
```
MLP Projection 모델 은 선형 모델(FC)보다 훨씬 강력한 표현력을 가집니다. BatchNorm 은 학습 과정을 안정화하고 속도를 높여주며, ReLU 는 모델이 단순한 차원 축소를 넘어 데이터 사이의 복잡하고 미묘한 관계를 파악하도록 돕습니다. 실험 결과, 대개 선형 모델보다 더 높은 Recall@K 수치를 기록하며, 특히 모호하거나 복잡한 문장 쿼리에서도 정답 이미지를 상위권에 배치하는 등 한층 고도화된 검색 실력을 보여줍니다.

---

## 📊 11. 비교 정리: FC vs MLP
실험을 통해 얻은 두 모델의 학습 데이터를 시각화하고 최종 성능을 비교합니다. 따라서 Training Curve 를 통해 학습 안정성을 파악하고, 테스트 세트의 Recall@K 수치를 나란히 대조하여 비선형성 (ReLU) 과 정규화 (BatchNorm) 가 실제 검색 품질에 미치는 영향을 최종적으로 정리합니다.

```python
# Training curve 비교
plt.figure(figsize=(7, 4))
plt.plot(fc_train_losses, label='FC')
plt.plot(mlp_train_losses, label='MLP')
plt.xlabel('Epoch'); plt.ylabel('Train Loss')
plt.title('Training Loss: FC vs MLP')
plt.legend(); plt.grid(alpha=0.3)
plt.show()

print('=== 최종 비교 (Test set) ===\n')
print('FC Projection:')
# 선형 모델의 최종 Recall@K 평가
evaluate_model(model_fc, test_loader, device);

print('\nMLP Projection:')
# 비선형 모델의 최종 Recall@K 평가
evaluate_model(model_mlp, test_loader, device);
```
Training Loss 곡선 을 보면, 일반적으로 MLP 모델 이 FC 모델 보다 초기 손실값이 더 빠르게 떨어지거나 더 낮은 지점에서 수렴하는 경향을 보입니다. 이는 BatchNorm 이 학습 속도를 가속하고, MLP 의 깊은 구조가 데이터 간의 복잡한 정렬을 더 효과적으로 수행했음을 의미합니다.

최종 성적표인 Recall@K 를 비교해 보면, 상위 1개(R@1)부터 10개(R@10)까지 모든 지표에서 MLP Projection 이 우세한 경우가 많습니다. 이는 이미지의 시각적 특징과 텍스트의 맥락 정보가 단순한 선형 결합보다는, 비선형적인 층을 거칠 때 공통 공간 (Shared Space) 에서 더욱 정밀하게 밀집될 수 있음을 증명합니다. 이번 실습을 통해 구조적 고도화가 멀티모달 시스템의 검색 성능을 결정짓는 핵심 요소임을 직접 확인할 수 있었습니다.

---

## 🥲 마치며: 멀티모달 검색 시스템의 가능성
이번 실습을 통해 이미지와 텍스트라는 서로 다른 형태의 데이터를 하나의 공통 임베딩 공간 (Shared Space) 으로 모으는 Retrieval 모델의 전 과정을 직접 구현해 보았습니다.

단순히 코드를 돌려보는 것에 그치지 않고, 임베딩 레이어 의 동작 원리부터 LSTM 을 통한 맥락 요약, 그리고 FC 와 MLP 구조의 성능 차이를 직접 비교하며 분석해보았습니다. 특히 Triplet Loss 와 Hard Negative Mining 을 통해 모델이 스스로 정답과 오답을 정교하게 구분해 나가는 과정은 현대적인 검색 엔진과 추천 시스템의 정수를 보여주었습니다.

숫자로 나타나는 Recall@K 지표의 향상뿐만 아니라, 실제 텍스트 쿼리에 대해 모델이 유의미한 이미지를 찾아내는 시각화 결과를 확인하며 딥러닝 모델이 시각과 언어를 어떻게 연결하는지 깊이 있게 이해할 수 있었습니다. 감사합니다.
