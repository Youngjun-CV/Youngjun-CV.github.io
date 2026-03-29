---
layout: post
title: "triplet loss 실습"
date: 2026-03-25 23:58:25 +0900
category: Multimodal
---

## 😊 들어가며
지난 포스팅에서는 데이터 임베딩을 했으니까, 이번에는 진짜 실습을 해보겠습니다. 이번에는 이전시간에 만든 flickr8k_data.pt 파일이 필요합니다.

---

## 🛠️ 0. 환경 설정 및 라이브러리 임포트
실무적인 실험 구성을 위해 필요한 도구들을 불러옵니다. 연산 속도 향상을 위한 GPU 설정도 잊지 않습니다. 지난 실습과 동일합니다.

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
이전 Unimodal Representation 실습에서 가공하여 저장해두었던 .pt 파일을 불러옵니다. 이 파일에는 이미지의 특징 벡터부터 단어 사전, 텍스트 인덱스까지 모델 학습에 필요한 모든 데이터가 들어있습니다.

```python
# Flickr8k 이미지 데이터셋 다운로드 (검증용 이미지 로드 시 필요)
!wget -q --show-progress https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!unzip -q Flickr8k_Dataset.zip

IMAGE_DIR = 'Flickr8k_Dataset'
```
이미지 데이터셋을 불러옵니다. 검증용 입니다.

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
전에 만든 pt 파일을 불러옵니다. 

```python
# 로드된 데이터 규모 및 형태 확인
print(f'Image features : {img_feats.shape}')
print(f'GloVe vectors  : {glove_vectors.shape}')
print(f'Caption IDs    : {caption_ids.shape}')
print(f'Splits - train: {len(data["splits"]["train"])}, val: {len(data["splits"]["val"])}, test: {len(data["splits"]["test"])}')
```
데이터를 확인합니다. 순서대로 (8091, 2048), (4280, 300), (8091, 32), (6000, 1000, 1000) 입니다.

---

## 🏗️ 2. Dataset & DataLoader
모델이 학습에 집중할 수 있도록 원본 데이터를 유기적으로 묶어 전달하는 과정이 필요합니다. 따라서 PyTorch의 Dataset 클래스를 상속받아 이미지 특징과 텍스트 인덱스를 쌍으로 반환하는 맞춤형 데이터셋을 정의하고, 이를 DataLoader를 통해 배치(Batch) 단위로 공급합니다.

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
데이터셋 클래스 입니다.

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
학습용 및 평가용 데이터셋 생성합니당.

---

## 🧠 이미지와 텍스트 인코딩
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
이미지 인코더 입니다.

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
        # 패딩을 무시하고 실제 문장 길이만큼만 연산하여 효율성을 높입니다.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed) # 최종 은닉 상태(Hidden State) 추출
        
        # 마지막 시점의 은닉 상태를 공통 공간으로 투영 후 L2 정규화 수행
        return F.normalize(self.proj(h.squeeze(0)), dim=-1)
```
텍스트 인코더 입니다.

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
통합모델입니다.

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
트리플렛 로스함수입니다.

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
recall k 평가함수임

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
전체 데이터 수치화

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
실제 텍스트로 이미지 검색

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
