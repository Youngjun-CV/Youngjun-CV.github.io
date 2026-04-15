---
layout: post
title: "🔥 Attention Is All You Need: Transformer 구조 완전 이해"
date: 2026-04-13 19:40:57 +0900
category: NLP
---

## Introduction
지난 포스팅에서는 **Attention**과 **Self-Attention**에 대해 살펴보았다. Attention은 입력 전체를 동일하게 보는 것이 아니라, 현재 시점에서 중요한 부분에 더 집중하도록 만드는 메커니즘이었다. 그리고 Self-Attention은 문장 내의 각 단어가 자기 자신을 포함한 다른 모든 단어들과의 관계를 계산하여 문맥을 이해하는 방식이었다. 

특히 Self-Attention의 등장으로, 기존에 순차적으로 정보를 처리하던 RNN 구조를 사용하지 않고도 문장 전체의 관계를 한 번에 파악할 수 있게 되었다. 그렇다면 RNN 없이 Attention만으로 모델을 구성할 수 있지 않을까? 

이러한 발상에서 등장한 모델이 바로 **Transformer**다. 이번 글에서는 Attention을 기반으로 설계된 Transformer의 구조와 핵심 아이디어를 정리해보겠다.

---

## 🧱 1. Transformer 전체 구조
Transformer는 Encoder와 Decoder로 구성된 구조를 가지며, 기존의 RNN과 달리 Attention 메커니즘만을 사용하여 시퀀스를 처리한다.

### 1.1 Encoder - Decoder 구조
Transformer는 여러 개의 Encoder와 Decoder 레이어를 쌓아 올린 형태로 구성된다.

각 레이어는 Attention과 Feed Forward Network로 이루어져 있으며, 입력을 점점 더 풍부한 표현으로 변환한다. Encoder는 입력 문장의 전체 문맥 정보를 압축하여 표현을 생성하고, Decoder는 이 정보를 바탕으로 출력 시퀀스를 순차적으로 생성한다.

또한 각 레이어 내부에는 Multi-Head Attention, Add & Norm, Feed Forward Network가 포함되어 있어, 정보 간의 관계를 학습하고 안정적으로 표현을 변환할 수 있도록 설계되어 있다.

### 1.2 데이터 흐름
Transformer에서 데이터는 다음과 같은 흐름으로 처리된다.

1. **입력 임베딩 (Input Embedding) :** 입력 문장은 Embedding과 Positional Encoding을 거쳐 Encoder로 들어간다.
2. **인코딩 (Encoding) :** Encoder는 Self-Attention을 통해 문장 전체의 관계를 반영한 표현을 만든다.
3. **디코딩 (Decoding) :** : Decoder는 이전까지 생성된 단어와 Encoder의 출력을 함께 사용한다.
4. **출력 생성 (Output Generation) :** 최종적으로 다음 단어를 예측하여 출력한다.

### 1.3 핵심 아이디
Transformer는 Attention 메커니즘만으로 시퀀스를 처리하는 구조로, 기존 RNN 기반 모델의 한계를 해결하기 위해 다음과 같은 설계 아이디어를 가진다.

#### ◼️ Attention 중심 구조 (Attention-Only Architecture)
* **특징 :** RNN이나 CNN 없이 Attention만으로 시퀀스를 모델링한다.  
* **장점 :** 순차적 의존성을 제거하여 구조를 단순화하고, 병렬 처리가 가능하다.  

#### ◼️



#### ◼️



#### ◼️




---













