---
layout: post
title: "🚀 프루닝의 마침표: 최적의 Pruning Ratio와 시스템 가속 전략"
date: 2026-04-06 17:06:56 +0900
category: ModelCompression
---

## Introduction
지난 포스팅에서는 프루닝 설계의 두 핵심 요소인 **Granularity (프루닝 단위)** 와 **Criterion (프루닝 기준)** 을 정리했다. Granularity가 연산 효율에 영향을 주는 구조적인 선택이었다면, Criterion은 어떤 파라미터가 중요한지를 판단하는 기준에 해당한다.

하지만 프루닝을 완성하기 위해서는 한 가지 요소를 더 고려해야 한다. 바로 **Pruning Ratio (프루닝 비율)** 다. 모든 레이어에 동일한 비율을 적용하는 것은 비효율적이기 때문에, 레이어별로 얼마나 제거할 것인지를 결정하는 것이 중요하다.

또한 단순히 가중치를 0으로 만드는 것만으로는 실제 속도 향상을 보장할 수 없다. 하드웨어가 sparsity를 효율적으로 활용하지 못한다면, 이론적인 연산 감소가 실제 성능 향상으로 이어지지 않기 때문이다.

따라서 이번 글에서는 레이어별 pruning ratio를 어떻게 설정할 것인지와, 이를 실제 시스템에서 연산 가속으로 연결하는 방법을 함께 살펴보겠다.

---

## 📊 1. Pruning Ratio (프루닝 비율)
프루닝의 **단위(Granularity)** 와 **기준(Criterion)** 을 정했다면, 이제 각 레이어마다 얼마나 제거할 것인지, 즉 프루닝 비율을 결정해야 한다.

### 1.1 개념
![image](/images/ModelCompression/ratio_1.jpg)
프루닝 비율은 모델 내 각 레이어에 어느 정도의 sparsity를 적용할 것인지를 의미한다.

모든 레이어에 동일한 비율을 적용하는 **Uniform Shrinking** 방식보다, 레이어별 특성에 따라 다르게 설정하는 **Non-uniform Pruning** 방식이 일반적으로 성능 보존에 유리하다.

즉, 전체 sparsity 목표가 동일하더라도 레이어별 비율을 어떻게 분배하느냐에 따라 성능이 크게 달라진다.

### 1.2 Finding Pruning Ratio: Sensitivity Analysis
![image](/images/ModelCompression/ratio_2.jpg)
레이어별 적절한 비율을 결정하기 위해 **Sensitivity Analysis (민감도 분석)** 를 수행한다. 각 레이어는 수행하는 역할이 다르기 때문에, 프루닝에 대한 민감도 또한 서로 다르게 나타난다.

* **민감도가 높음:** 작은 비율의 프루닝에도 전체 정확도가 크게 감소한다.
* **민감도가 낮음 (Redundant):** 높은 비율로 제거해도 성능 변화가 거의 없다.

#### ◼️ 분석 방법 (예: VGG-11)
1. **레이어 선택**

    모델의 각 레이어($L_{0}, L_{1}, \dots$)를 순차적으로 선택한다.

2. **프루닝 적용**

    선택한 레이어의 비율을 $0$부터 $0.9$까지 증가시키며 모델의 정확도를 측정한다. 이때 다른 레이어는 프루닝하지 않는다.

3. **민감도 분석**

    비율 변화에 따른 정확도 감소 곡선을 통해 레이어별 민감도를 비교한다.

#### ◼️ 결과 해석
예를 들어, $L_{0}$ 은 약 $70\%$ 이후 정확도가 급격히 감소하는 반면, $L_{1}$ 은 $80\%$ 이상에서도 성능이 유지될 수 있다. 이는 $L_{1}$ 이 상대적으로 중복된 정보를 많이 포함하고 있음을 의미한다.

#### ◼️ 결정 방식과 한계
* **특징:** 정확도 임계값($Threshold~T$)을 기준으로 레이어별 최대 프루닝 비율을 결정한다.
* **장점:** 각 레이어의 특성을 반영하여 비교적 안정적인 비율 설정이 가능하다.
* **단점:** 레이어를 독립적으로 분석하기 때문에 여러 레이어가 동시에 프루닝될 때의 상호작용(Joint Sensitivity)을 반영하지 못한다.

### 1.3 AutoML for Model Compression (AMC)
<img src="/images/ModelCompression/ratio_3.jpg" style="width:70%;">
수동적인 민감도 분석의 한계를 극복하고, 최적의 pruning ratio 조합을 자동으로 찾기 위해 제안된 방법이 **AMC** 다. 강화학습 에이전트가 모델을 하나의 환경으로 보고, 각 레이어에 대한 최적의 압축 전략을 학습한다.

#### ◼️ 작동 메커니즘 (DDPG Agent 기반)
AMC는 모델의 레이어를 순차적으로 탐색하며 각 레이어에 대한 pruning 비율을 결정한다.

* **State ($s_{t}$):** 현재 레이어의 구조 정보를 나타낸다.  
  → 레이어 인덱스($t$), 채널 수($c$), 커널 크기($n$), feature map 크기($h, w$) 등이 포함된다.

* **Action ($a_{t}$):** 해당 레이어에 적용할 pruning ratio를 의미한다.  
  → DDPG를 사용하여 이산 값이 아닌 연속적인 비율을 출력한다.

* **Reward ($r$):** 모든 레이어 결정 이후 모델 성능을 기반으로 계산된다.  
  → 기본적으로 $Reward = -Error$ 를 사용하여 정확도 손실이 작을수록 높은 보상을 부여한다.  
  → 제약 조건을 반영할 경우 $Reward = -Error \times \log(FLOPs)$ 와 같이 정의하여 정확도와 연산량을 동시에 고려한다.

#### ◼️ 특징 및 효과
![image](/images/ModelCompression/ratio_4.jpg)
* **특징:** 강화학습을 통해 레이어 간 상호작용을 고려한 pruning ratio를 자동으로 탐색한다.
* **장점:** 사람이 설계한 방식보다 높은 압축률에서도 정확도를 안정적으로 유지 가능하며, 학습된 결과를 통해 어떤 레이어가 중요한지 해석 가능하다.
* **단점:** 학습 과정이 필요하여 시간과 계산 비용이 크고 구현이 복잡하다.

---

## 💻 2. System Support for Sparsity
프루닝을 통해 **sparsity**를 만들었다 하더라도, 하드웨어가 이를 효율적으로 처리하지 못하면 실제 연산 속도 향상으로 이어지지 않는다. 일반적인 프로세서는 0을 포함한 모든 연산을 동일하게 수행하기 때문이다.

따라서 성능 향상을 위해서는 0을 건너뛰거나, 압축된 데이터만 선택적으로 연산할 수 있는 시스템 수준의 지원이 필요하다.

### 2.1 NVIDIA M:N Sparsity (2:4 Structured Sparsity)
Pattern-based pruning이 실제 하드웨어에서 구현된 대표적인 사례다. NVIDIA는 정확도 손실을 최소화하면서도 연산 효율을 높이기 위해 **2:4 sparsity** 구조를 사용한다.

#### ◼️ 1단계: 행렬 압축 (Compression)
<img src="/images/ModelCompression/ratio_5.jpg" style="width:70%;">
원본 가중치 행렬에서 non-zero 값만 선택하여 압축한다. 이 과정에서 다음 두 가지 정보가 생성된다.

* **Non-zero Values:** 0을 제외한 실제 가중치 값만 저장하여 메모리 사용량을 감소시킨다.
* **2-bit Metadata (Index):** 각 값이 원래 4개 블록 중 어느 위치에 있었는지를 나타내는 정보로, 위치당 2비트를 사용한다.

#### ◼️ 2단계: Tensor Core 매핑 및 연산
![image](/images/ModelCompression/ratio_6.jpg)

압축된 가중치와 입력 데이터가 결합될 때, Tensor Core는 metadata를 활용하여 선택적인 연산을 수행한다.

1. 입력 행렬에서 metadata가 가리키는 위치의 값만 선택한다. **(Gather)**
2. 선택된 입력값과 압축된 non-zero 값을 1:1로 매칭하여 곱셈을 수행한다.
3. 불필요한 0 연산이 제거되어, 이론적으로 최대 2배의 속도 향상을 얻을 수 있다.

### 2.2 Sparse Convolution on Sparse Inputs
가중치뿐만 아니라 입력 데이터까지 sparse한 경우, 연산량을 크게 줄일 수 있는 방식이다.

#### ◼️ 핵심 개념: Submanifold Sparse Convolution
![image](/images/ModelCompression/ratio_7.jpg)
일반적인 convolution은 연산이 진행될수록 값이 주변으로 확산되면서 sparsity가 빠르게 사라진다. 이를 방지하기 위해 **Submanifold Sparse Convolution** 은 입력에서 0이 아닌 위치에 대해서만 출력을 계산한다.

이 방식은 불필요한 연산 확산을 막고, sparse 구조를 유지하는 데 목적이 있다.

#### ◼️ 연산 과정: Weight-Stationary Computation
![image](/images/ModelCompression/ratio_8.jpg)
**Weight-Stationary Computation** 은 가중치를 연산기에 고정하여 메모리 접근을 최소화하는 방식이다. 입력 데이터만 순차적으로 공급하고, 0이 아닌 값에 대해서만 연산을 수행한다.

1. **Weight Grouping & Maps**

    연산 전에 유효한 입력값들이 어떤 가중치와 연산되는지 미리 매핑한다.

2. **Input Buffer**

   입력 feature map에서 0이 아닌 값만 선택하여 버퍼에 저장한다.

3. **Weight-Stationary Computation**

    가중치를 연산기에 고정하고, 해당 가중치와 연산할 입력값들을 순차적으로 처리한다.

4. **Partial Sum Accumulation**
    연산 결과를 partial sum 버퍼에 누적하고, 최종적으로 output feature를 생성한다.

---

## Conclusion

이로써 프루닝의 핵심적인 요소들을 모두 정리해보았다. 모델을 어떤 단위로 제거할지 결정하는 **Granularity**, 어떤 기준으로 중요도를 판단할지 정의하는 **Criterion**, 그리고 레이어별로 얼마나 제거할지를 결정하는 **Pruning Ratio** 까지 살펴봤다. 나아가 이렇게 만들어진 **sparsity**를 실제 연산 가속으로 연결하기 위한 **System Support**까지 함께 정리했다.

프루닝은 단순히 파라미터 수를 줄이는 기법이 아니라, 알고리즘과 하드웨어가 함께 설계되어야 하는 최적화 문제라는 것을 확인할 수 있었다.

이제 모델의 크기를 줄이는 방법을 살펴봤으니, 다음 글에서는 가중치의 **표현 정밀도**를 줄여 효율을 높이는 **Quantization (양자화)** 에 대해 알아보겠다.
