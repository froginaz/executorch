# RunnerManagedCacheIOManager: Attention Mask 기법 및 구현

## 개요

`RunnerManagedCacheIOManager`는 runner가 KV cache를 외부에서 직접 관리하는 IO manager입니다. 모델은 과거 KV cache를 입력으로 받고, 새로운 KV cache 항목을 출력으로 생성합니다. Runner가 출력을 cache에 concat하여 다음 step의 입력으로 제공합니다.

## Attention Mask 기법

### Binary Causal Mask (0/1 방식)

이 구현은 **binary mask** 방식을 사용합니다. MediaTek의 `MaskBuilder`가 사용하는 large-negative-value 방식(`-100`, `-32768`)과 달리, attend 가능한 위치를 `1.0f`, 불가능한 위치를 `0.0f`로 표현합니다.

```
mask 값: 1.0 → attend 가능
mask 값: 0.0 → attend 불가능
```

이 방식은 모델 내부에서 mask를 곱하거나 별도의 masking 로직을 적용하는 구조에 적합합니다.

### Mask Shape

| 단계 | Shape | 설명 |
|------|-------|------|
| Prefill | `[attn_mask_rows, attn_mask_cols]` (예: `[128, 1024]`) | 여러 토큰을 한 번에 처리 |
| Decode | `[1, attn_mask_cols]` (예: `[1, 1024]`) | 한 토큰씩 생성 |

Prefill과 decode 모두 **동일한 버퍼**(`attn_mask_`)를 사용하되, tensor shape만 다르게 설정합니다.

### Mask 생성 알고리즘: update_attn_mask

```cpp
void update_attn_mask(size_t pos, size_t seq_len) {
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);  // 전체 초기화
    for (size_t r = 0; r < seq_len && r < config_.attn_mask_rows; r++) {
        size_t visible = pos + r + 1;
        if (visible > config_.attn_mask_cols) {
            visible = config_.attn_mask_cols;
        }
        for (size_t c = 0; c < visible; c++) {
            attn_mask_[r * config_.attn_mask_cols + c] = 1.0f;
        }
    }
}
```

각 query row `r`에 대해 `[0, pos + r]` 범위의 위치에 attend 가능하도록 설정합니다. 이는 lower-triangular causal pattern을 만듭니다.

### Prefill 단계 Mask 예시

`pos=0`, `seq_len=4`, `attn_mask_cols=8`:

```
row 0: [1 0 0 0 | 0 0 0 0]   ← 첫 번째 토큰: 자기 자신만
row 1: [1 1 0 0 | 0 0 0 0]   ← 두 번째: 0~1번 토큰
row 2: [1 1 1 0 | 0 0 0 0]   ← 세 번째: 0~2번 토큰
row 3: [1 1 1 1 | 0 0 0 0]   ← 네 번째: 0~3번 토큰
```

### Micro-batch Prefill Mask 예시

두 번째 micro-batch (`pos=4`, `seq_len=4`, `attn_mask_cols=8`):

```
row 0: [1 1 1 1 1 0 0 0]   ← 0~4번 토큰에 attend
row 1: [1 1 1 1 1 1 0 0]   ← 0~5번
row 2: [1 1 1 1 1 1 1 0]   ← 0~6번
row 3: [1 1 1 1 1 1 1 1]   ← 0~7번
```

### Decode 단계 Mask 예시

`pos=8`, `seq_len=1`, `attn_mask_cols=16`:

```
row 0: [1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]
        ← 0~8번에 attend 가능 →
```

## MediaTek MaskBuilder와의 비교

| 항목 | RunnerManagedCacheIOManager | MediaTek MaskBuilder |
|------|---------------------------|---------------------|
| Mask 값 | `0.0` / `1.0` (binary) | `0` / `-100` 또는 `0` / `-32768` (additive) |
| 데이터 타입 | `float` 고정 | INT16, FP16, FP32 지원 |
| Mask 구조 | `[seq_len, attn_mask_cols]` | `[batch, cache_len + batch]` (cache mask + attn mask 결합) |
| 업데이트 방식 | 매번 전체 재구축 (`std::fill` 후 재계산) | 증분 업데이트 (`updateMask`) + dirty 시 재구축 fallback |
| Sliding Window | 미지원 | 지원 (`enableSlidingWindow`) |
| Padding | 미지원 | Left/Right padding 지원 |
| Cache 영역 분리 | mask에 cache 영역 없음 (pos 기반 offset) | cache mask + attention mask를 concat하여 표현 |

### 설계 차이의 이유

- **RunnerManagedCacheIOManager**: Runner가 cache를 직접 관리하므로, mask는 단순히 "몇 번째 위치까지 attend 가능한지"만 표현하면 됩니다. Cache 내부 구조를 mask가 알 필요가 없습니다.
- **MediaTek MaskBuilder**: 모델이 cache를 input/output으로 직접 다루며, mask가 cache 영역과 attention 영역을 명시적으로 분리하여 표현합니다. NPU 가속을 위해 다양한 데이터 타입과 증분 업데이트를 지원합니다.

## KV Cache 관리와의 연동

### Prefill → Decode 전환

```
1. prepare_prefill() 호출 (여러 번 가능 = micro-batch)
   └─ update_attn_mask(current_pos_, prefill_seq_len)
   └─ KV cache input: [1, H, max_seq_len - prefill_seq_len, D]

2. update_prefill() 호출
   └─ 모델 출력의 KV를 prefill input 버퍼에 누적 (copy_to_cache)
   └─ current_pos_ += prefill_seq_len

3. prepare_decode() 첫 호출
   └─ copy_prefill_to_decode_input()  ← prefill 버퍼 → decode 버퍼 복사
   └─ update_attn_mask(current_pos_, 1)
   └─ KV cache input: [1, H, max_seq_len - 1, D]

4. update_decode() 호출
   └─ 모델 출력의 KV를 decode input 버퍼에 기록 (copy_to_cache)
   └─ current_pos_ += 1
```

### 핵심: pos 기반 mask

mask의 visible 범위가 `current_pos_`에 의해 결정되므로, KV cache에 기록된 위치와 mask가 자동으로 동기화됩니다. cache에 데이터가 없는 위치는 mask가 `0.0`이므로 attend되지 않습니다.
