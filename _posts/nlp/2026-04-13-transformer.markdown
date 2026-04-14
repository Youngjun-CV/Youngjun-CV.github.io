---
layout: post
title: "🔥 Attention Is All You Need: Transformer 구조 완전 이해"
date: 2026-04-13 19:40:57 +0900
category: NLP
---

## Introduction
지난 포스팅에서는 **Attention**과 **Self-Attention**에 대해 살펴보았다. Attention은 입력 전체를 동일하게 보는 것이 아니라, 현재 시점에서 중요한 부분에 더 집중하도록 만드는 메커니즘이었다. 그리고 Self-Attention은 문장 내의 각 단어가 자기 자신을 포함한 다른 모든 단어들과의 관계를 계산하여 문맥을 이해하는 방식이었다.

특히 Self-Attention의 등장으로, 기존에 순차적으로 정보를 처리하던 RNN 구조를 사용하지 않고도 문장 전체의 관계를 한 번에 파악할 수 있게 되었다.

그렇다면 한 가지 질문이 자연스럽게 떠오른다.  
“굳이 RNN을 사용할 필요 없이, Attention만으로 모델을 구성할 수 있지 않을까?”

이 질문에 대한 답으로 등장한 모델이 바로 **Transformer**다. 이번 글에서는 Attention을 기반으로 만들어진 Transformer의 구조와 핵심 아이디어를 정리해보겠다.











