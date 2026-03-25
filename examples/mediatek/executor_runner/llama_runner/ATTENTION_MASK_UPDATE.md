# Attention Mask Update: Theory and Implementation

## Theory

### Causal Attention Mask

Transformer 기반 LLM의 디코더는 **causal (autoregressive) attention**을 사용합니다. 각 토큰은 자기 자신과 이전 토큰들에만 attend할 수 있으며, 미래 토큰에는 attend할 수 없습니다. 이를 제어하는 것이 attention mask입니다.

Attention score 계산:

```
Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + mask) @ V
```

mask 값이 `0`이면 해당 위치에 attend 가능하고, mask 값이 매우 큰 음수(예: `-inf`, `-100`, `-32768`)이면 softmax 후 해당 위치의 가중치가 0에 수렴하여 attend 불가능합니다.

### KV Cache와 Mask의 관계

추론 시 KV cache를 사용하면, mask는 두 영역으로 구성됩니다:

```
mask = [cache_mask | attention_mask]
        ← cache →   ← batch →
       (kCacheLength) (tokenBatchSize)
```

- **cache_mask** (길이: `kCacheLength`): KV cache에 저장된 과거 토큰들에 대한 mask
- **attention_mask** (길이: `tokenBatchSize`): 현재 입력 토큰 배치 내의 causal mask

총 mask 길이: `kCacheLength + tokenBatchSize`

### Mask 값 (타입별)

| Type   | True (attend 가능) | False (attend 불가) |
|--------|-------------------|-------------------|
| INT16  | 0                 | -32768            |
| FP16   | 0                 | -100              |
| FP32   | 0                 | -100              |

### Prefill vs Decode 단계의 Mask Shape

- **Prefill** (prompt 처리): shape `[tokenBatchSize, kCacheLength + tokenBatchSize]`
  - 예: batch=128, cache=1024 → `[128, 1152]`
- **Decode** (토큰 생성): shape `[1, kCacheLength + 1]`
  - 예: cache=1024 → `[1, 1025]`

## Implementation

### buildMask: 전체 Mask를 처음부터 구축

초기 상태이거나 mask가 dirty로 표시된 경우 호출됩니다.

```
buildMask(tokenBatchSize, numSeenToken)
```

각 행(row)은 입력 토큰 하나에 대응하며, 다음 규칙으로 구성됩니다:

1. `numVisibleCacheTokens = min(kCacheLength, numSeenToken)` — cache에서 볼 수 있는 토큰 수
2. 각 행 `i`에 대해:
   - `attnTrueCount = i + 1` (causal: 자기 자신까지만)
   - `rowTrueCount = min(windowSize, numVisibleCacheTokens + attnTrueCount)`
   - True 구간: `[firstTrueIdx, lastTrueIdx]`
   - 나머지: False

시각적 예시 (prefill, batch=4, cache=8, numSeenToken=0):

```
cache region    attn region
[F F F F F F F F | T F F F]   ← row 0: 첫 번째 토큰
[F F F F F F F F | T T F F]   ← row 1
[F F F F F F F F | T T T F]   ← row 2
[F F F F F F F F | T T T T]   ← row 3
```

decode 단계 (batch=1, numSeenToken=4):

```
cache region          attn
[F F F F T T T T | T]   ← 과거 4개 토큰 + 현재 토큰
```

### updateMask: 증분 업데이트 (최적화)

decode 단계에서는 매번 전체 mask를 재구축할 필요 없이, 새로 보이게 된 cache 토큰 위치만 True로 설정합니다.

```
updateMask(tokenBatchSize, numSeenToken, length)
```

- `mIsMaskUpdatable`이 false면 `buildMask`로 fallback
- True로 설정할 구간: `startTrueOffset`부터 `trueCount`개

이전 mask:
```
[F F F F T T T T | T]
```

updateMask 후 (numSeenToken=5, length=1):
```
[F F F T T T T T | T]
         ^
      새로 True
```

이렇게 하면 O(length)만큼만 수정하므로 O(maskLength)인 buildMask보다 효율적입니다.

### Sliding Window Attention (SWA)

`mSlidingWindowSize > 0`일 때 활성화됩니다. 전체 과거 토큰이 아닌 최근 `windowSize`개 토큰에만 attend합니다.

```
rowTrueCount = min(windowSize, numVisibleCacheTokens + attnTrueCount)
```

Gemma3 등 일부 모델은 global attention과 SWA를 레이어별로 번갈아 사용하므로, 별도의 SWA mask 버퍼(`IOKind::SWAMask`)를 두고 독립적으로 buildMask를 호출합니다.

### Padding 처리

프롬프트 길이가 batch size의 배수가 아닐 때 padding이 필요합니다.

- **Left padding**: 앞쪽에 패딩 토큰 추가. 패딩된 행 전체와 패딩 attention 영역을 False로 설정
- **Right padding**: 뒤쪽에 패딩 토큰 추가. 패딩된 행 전체를 False로 설정

padding이 적용되면 `mIsMaskUpdatable = false`가 되어, 다음 호출 시 buildMask로 전체 재구축합니다.

### 실행 흐름

```
LlamaModelChunk::Run()
  └─ UpdatePosEmbAndMask(tokenBatchSize)
       ├─ [SWA가 활성화된 경우]
       │    ├─ setMaskBuffer(swaMaskBuffer)
       │    ├─ enableSlidingWindow(windowSize)
       │    └─ buildMask(tokenBatchSize, tokenIndex)
       │
       ├─ setIsMaskUpdatable(savedStatus)
       ├─ setMaskBuffer(globalMaskBuffer)
       ├─ disableSlidingWindow()
       └─ updateMask(tokenBatchSize, tokenIndex, numInputToken)
            └─ [updatable이면 증분 업데이트, 아니면 buildMask fallback]
```

### HotSwapModel과 Mask

prompt→decode 모드 전환 시 `HotSwapModel`이 호출되면:

1. `markMaskDirty()` — batch size가 바뀌므로 mask 재구축 필요
2. `updateMaskSize()` — 새 모델의 mask 버퍼 크기로 업데이트

다음 `updateMask` 호출 시 자동으로 `buildMask`가 실행됩니다.
