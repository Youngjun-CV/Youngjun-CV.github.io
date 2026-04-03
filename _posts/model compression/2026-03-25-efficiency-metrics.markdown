---
layout: post
title: "이피션시 매트릭스"
date: 2026-03-25 20:34:29 +0900
category: ModelCompression
---

## ☺️ 들어가며
오늘은 인공지능 모델을 성능을 단순히 정확도(Accuracy) 측면이 아닌, 실제 기기에서 얼마나 효율적으로 돌아가는지를 결정짓는 **Efficiency Metrics** (효율성 지표)에 대해 자세히 알아보겠습니다. 

최근 NVDIA GPU와 같은 고성능 디바이스뿐만 아니라, 제한된 자원을 가진 임베디드 디바이스나 노트북에서도 원활하게 동작하는 모델을 만드는 것이 중요해지면서 이러한 효율성 지표에 대한 이해가 필수가 되었습니다.

---

## 🔩 효율성 지표의 개요 (Efficiency Metrics)
모델의 효율성은 크게 **연산 복잡도** 와 **메모리 복잡도** 두 가지 측면에서 평가됩니다.

### (1) 핵심평가 지표
* **Memory 관련 지표:** 파라미터 수 (#parameters), 모델 크기 (model size), 활성화 함수의 총량 및 피크치 (total/peak #activations)

* **Computation 관련 지표:** MAC (Multiply-Accumulate), FLOP/FLOPS, OP/OPS

### (2) 복잡도 최적화 전략
우리의 최종 목표는 더 작고 (Smaller), 더 빠르고 (Faster), 더 친환경적인 (Greener) 모델을 구현하는 것입니다.

* **연산 복잡도:** 이를 줄이기 위해서는 기본적으로 코어의 퍼포먼스가 높아야 합니다.

* **메모리 복잡도:** OFff-chip 메모리와의 Latency를 줄이는 것이 핵심이며, 이를 위해 HBM (High Bandwidth Memory) 같은 기술이 사용됩니다.

---

## 🌿 주요 성능 지표 상세

### (1) Latency (지연시간) vs Throughput (처리량)
* **Latency:** 특정 작업을 완료하는 데 걸리는 시간(Delay)입니다. **CPU** 는 범용적이고 코어 수는 적지만 개별 연산 속도가 빨라 Latency 최적화에 유리합니다.

* **Throughput:** 단위 시간당 처리되는 데이터의 양입니다. **GPU** 는 수만 개의 코어를 통해 대량의 데이터를 병렬 처리하는 Throughput 최적화에 특화되어 있습니다.

* **Throughput 향상을 위한 프로세서 설계 방식**
  1. **Single Processor:** 단일 프로세서 처리
  2. **Parallel Processor:** 여러 프로세서를 평행하게 사용
  3. **Pipelined Processor:** 단계별로 연산을 나누어 파이프라인화
 
### (2) Latency Estimation
실제 디바이스에 올리기 전, 다음과 같은 수식으로 성능을 예측해볼 수 있습니다. 

$$Latency \approx max(T_{computation}, T_{memory})$$

* **$T_{computation}$:** 네트워크 모델 연산 수 / 프로세서의 초당 연산 가능 개수 
* **$T_{memory}$:** 활성화 값 이동 시간 + 가중치(Weight) 이동 시간
* **Weight 이동 시간:** 모델 사이즈 / 메모리 대역폭 (Bandwidth)
* **Activation 이동 시간:** (Input Activation + Output Activation) / 메모리 대역폭 

### (3) 에너지 소비 (Energy Consumption)
데이터 처리 자체보다 데이터 이동 (Data movement)에 훨씬 많은 에너지가 소모됩니다.

* 32 bit DRAM 접근: 640 pJ 
* 32 bit float ADD 연산: 0.9 pJ 

따라서 에너지를 절약하려면 데이터를 Off-chip (DRAM)이 아닌 On-chip (SRAM) 내부에 저장하고 코어와 주고받는 것이 가장 효율적입니다. 이를 위해 Pruning 이나 Quantization 같은 모델 압축 기술이 사용됩니다.

---

## 📏 모델 사양 계산법 

### (1) 파라미터 수 (# Parameters)
모델의 저장 공간을 결정하는 요소입니다.
* Linear Layer: $C_{i} \times C_{o}$
* Convolution Layer: $C_{i} \times C_{o} \times k_{w} \times k_{h}$
* Grouped Conv: $C_{i} \times C_{o} \times k_{w} \times k_{h} / g$
* Depthwise Conv: $C_{o} \times k_{w} \times k_{h}$

AlexNet 의 경우 총 파라미터 수는 약 61M 이며, FP32 기준으로 모델 사이즈는 약 244 MB ($61M \times 4$ Bytes)가 됩니다.

### (2) 활성화 함수 (Activations)
추론 시 발생하는 중간 데이터로, 메모리 병목의 주원인입니다.

Total Activation: 모든 레이어 활성화 값의 합 (AlexNet: 약 932,264) 

Peak Activation: 특정 시점의 최대치 (입력 + 출력 활성화 값의 합, AlexNet: 약 440,928)

### (3) 연산량 (MAC, FLOP, OP)
* MAC: $a \leftarrow a + b \cdot c$ 연산 횟수입니다.
* FLOP: 1 MAC = 2 FLOP (곱셈 1 + 덧셈 1)으로 계산합니다.
* AlexNet 연산량: 약 724M MACs 로, 이를 FLOP 으로 환산하면 약 1.4 GFLOPs 가 됩니다.
