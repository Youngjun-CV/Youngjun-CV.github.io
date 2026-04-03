---
layout: post
title: "bidirectional rnn"
date: 2026-03-30 23:19:09 +0900
category: NLP
---

## 🏹 Bidirectional RNNs
일반적인 RNN은 과거의 정보만을 바탕으로 현재를 판단합니다. 하지만 언어는 뒤에 나오는 단어가 앞 단어의 의미를 결정하는 경우가 많습니다.

### (1) 양방향 RNN이 필요한 이유
* **예시:** "the movie was terribly exciting!"

* **문제:** 왼쪽 문맥 ("the movie was") 만 보면 'terribly' 는 부정적인 의미로 해석될 가능성이 높습니다.

* **해결:** 오른쪽 문맥 ("exciting") 을 함께 고려하면 'terribly' 가 ' 매우 ' 라는 긍정적 강조 의미임을 정확히 파악할 수 있습니다.

### (2) 구조와 동작 원리
양방향 RNN은 두 개의 독립적인 RNN 층을 가집니다.

* **Forward RNN:** 문장을 앞에서 뒤로 읽으며 과거 정보를 수집합니다.

* **Backward RNN:** 문장을 뒤에서 앞으로 읽으며 미래 정보를 수집합니다.

* **결합:** 각 시점 $t$ 에서 두 RNN 의 은닉 상태를 하나로 이어 붙여 (Concatenate) 최종적인 문맥적 표현 (Contextual representation) 을 만듭니다.

* **주의점:** 전체 입력 시퀀스를 미리 알고 있어야 하므로 , 실시간으로 다음 단어를 예측해야 하는 언어 모델링 (LM) 에는 사용할 수 없습니다. 하지만 문장 전체를 해석하는 인코딩 (Encoding) 작업에서는 필수적인 기법입니다.

---

## 🧱 Multi-layer RNNs
RNN 은 이미 시간 축으로 길게 펼쳐진 '깊은' 구조이지만, 수직 방향으로도 층을 쌓아 더 고차원적인 특징을 추출할 수 있습니다. 이를 Stacked RNN 이라고도 부릅니다.

### (1) 계층적 특징 추출
* **Lower Layers:** 단어의 형태나 문법 같은 낮은 수준의 특징 (Lower-level features) 을 학습합니다.

* **Higher Layers:** 문장의 전체적인 의미나 논리 구조 같은 높은 수준의 특징 (Higher-level features) 을 학습합니다.

### (2) 실무적인 팁
* **층의 깊이:** CNN 처럼 수십 층을 쌓지는 않으며, 보통 2~4층 정도가 성능과 효율 면에서 가장 적절합니다.

* **스킵 연결 (Skip-connections):** 층이 깊어질수록 여기서도 기울기 소실 문제가 발생할 수 있으므로, ResNet 처럼 이전 층의 정보를 직접 전달하는 경로를 추가하기도 합니다.


