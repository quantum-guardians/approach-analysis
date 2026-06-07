# mr2s-module Embedding Reuse Issue

## 문제 요약

`DnCMr2sSolver`의 `embedding_aware` 경로에서 subgraph를 풀 때, 이전 단계에서 얻은 embedding estimate를 재사용한다. 특정 trial에서 재사용된 embedding이 현재 BQM 또는 현재 D-Wave target topology에 대해 유효하지 않았고, `FixedEmbeddingComposite` 생성 시점에 다음 예외가 발생했다.

```text
DisconnectedChainError: chain for e_0_162 is not connected
```

## 정확한 의미

이 에러는 "graph가 절대 embedding 불가능하다"는 뜻이 아니다.

정확히는 `mr2s-module`이 D-Wave에 제공한 특정 embedding mapping이 invalid하다는 뜻이다. Logical variable `e_0_162`에 배정된 physical qubit chain이 QPU graph 위에서 연결된 chain을 이루지 못했다.

따라서 이 에러가 증명하는 것은 다음이다.

- 제공된 reused embedding은 현재 BQM/target topology에 대해 유효하지 않다.

이 에러가 증명하지 않는 것은 다음이다.

- 이 graph 또는 BQM이 어떤 embedding으로도 embedding 불가능하다.

## chain strength 문제가 아닌 이유

`chain_strength`는 valid embedding이 만들어진 뒤 sampling 중 chain break를 줄이기 위한 penalty다.

이번 에러는 sampling 전에 발생한다. D-Wave가 embedding 구조를 검증하는 단계에서 chain이 물리적으로 연결되어 있지 않다고 판단한 것이다. 따라서 `chain_strength`를 키워도 연결되지 않은 chain은 valid chain이 되지 않는다.

## 의심되는 원인

가능한 원인은 다음 중 하나다.

- embedding estimate가 실제 solve 대상 BQM과 다른 BQM에 대해 생성됨
- partition/probe 단계의 subgraph와 solve 단계의 subgraph가 정확히 매칭되지 않음
- BQM variable set 또는 interaction set이 embedding 생성 이후 달라짐
- target sampler 또는 QPU topology가 embedding 생성 시점과 solve 시점에 달라짐
- `EmbeddingAwareFaceCyclePartitionStrategy` 또는 `DnCMr2sSolver`가 embedding estimate를 저장하거나 재사용할 때 validation을 충분히 하지 않음
- invalid embedding을 받았을 때 fresh embedding fallback 없이 바로 `FixedEmbeddingComposite`에 넘김

## 기대 동작

`mr2s-module`은 embedding reuse 전에 embedding을 검증해야 한다.

안전한 동작은 다음 순서가 적절하다.

1. reused embedding이 현재 BQM 변수/edge와 target topology에 대해 valid한지 검사
2. invalid하면 reused embedding을 버림
3. fresh embedding을 다시 시도
4. fresh embedding도 실패하면 명확한 domain error를 raise
5. 호출자가 해당 trial을 failed result로 기록할 수 있도록 error reason 제공

## 이슈용 짧은 설명

`embedding_aware` DnC solver may reuse an invalid embedding estimate. In some trials, the reused embedding contains a disconnected chain, causing `FixedEmbeddingComposite` to raise `DisconnectedChainError` before sampling. This does not prove the graph is unembeddable; it means the provided reused embedding is invalid for the current BQM/target topology.
