# Transformer 구조와 Sampling: 예제 기반 이해

## 예제 설정

```
프롬프트: "나는 오늘"
vocab_size: 7
vocab: {0:"나는", 1:"오늘", 2:"학교에", 3:"갔다", 4:"밥을", 5:"먹었다", 6:"집에"}
d_model: 8 (설명을 위한 축소 크기)
```

## 전체 구조

```
Input tokens: [0, 1]  ("나는 오늘")
       │
       ▼
┌─────────────┐
│  Embedding   │  토큰 ID → 벡터
└──────┬──────┘
       ▼
┌─────────────┐
│  Attention   │  토큰 간 관계 학습    ─┐
├─────────────┤                        │ × N layers
│     FFN      │  각 토큰 표현력 확장   ─┘
└──────┬──────┘
       ▼
┌─────────────┐
│   lm_head    │  벡터 → vocab 확률
└──────┬──────┘
       ▼
┌─────────────┐
│  Sampling    │  확률 → 다음 토큰 선택
└─────────────┘
```

---

## Embedding

### 역할

토큰 ID(정수)를 모델이 처리할 수 있는 고차원 벡터로 변환합니다.

### 구조

```
Embedding Table: (vocab_size × d_model) = (7 × 8)

  ID 0 "나는":   [0.12, -0.45,  0.78,  0.33, -0.91,  0.56, -0.23,  0.67]
  ID 1 "오늘":   [0.89, -0.12,  0.34, -0.67,  0.45, -0.78,  0.91, -0.34]
  ID 2 "학교에": [0.23,  0.56, -0.89,  0.12,  0.67, -0.45,  0.34, -0.78]
  ID 3 "갔다":   [...]
  ID 4 "밥을":   [...]
  ID 5 "먹었다": [...]
  ID 6 "집에":   [...]
```

### 과정

```
Input: [0, 1] ("나는", "오늘")

ID 0 → 테이블에서 0번 행 조회 → [0.12, -0.45, 0.78, 0.33, -0.91, 0.56, -0.23, 0.67]
ID 1 → 테이블에서 1번 행 조회 → [0.89, -0.12, 0.34, -0.67, 0.45, -0.78, 0.91, -0.34]

Output: (2, 8) 행렬
  [0.12, -0.45,  0.78,  0.33, -0.91,  0.56, -0.23,  0.67]   ← "나는"
  [0.89, -0.12,  0.34, -0.67,  0.45, -0.78,  0.91, -0.34]   ← "오늘"
```

### Weight

- 학습 가능한 파라미터: `(vocab_size × d_model)` 행렬
- LLaMA-7B 기준: `(32000 × 4096)` = 약 **131M 파라미터**
- 이 테이블 자체가 학습을 통해 의미 있는 벡터 공간을 형성

---

## Multi-Head Self-Attention

### 역할

각 토큰이 시퀀스 내 다른 토큰들과의 관계를 파악합니다. "나는"이 "오늘"과 어떤 관계인지 학습합니다.

### Weight 행렬

```
W_Q: (d_model × d_model) = (8 × 8)   Query 생성
W_K: (d_model × d_model) = (8 × 8)   Key 생성
W_V: (d_model × d_model) = (8 × 8)   Value 생성
W_O: (d_model × d_model) = (8 × 8)   출력 변환
```

### 과정 (단일 head, 간략화)

```
X = Embedding 출력 (2, 8)

Step 1: Q, K, V 생성
  Q = X @ W_Q   (2, 8)   ← 각 토큰이 "무엇을 찾는지"
  K = X @ W_K   (2, 8)   ← 각 토큰이 "무엇을 제공하는지"
  V = X @ W_V   (2, 8)   ← 각 토큰이 "전달할 실제 정보"

Step 2: Attention Score 계산
  Score = Q @ K^T / sqrt(d_k)   (2, 2)

       "나는"  "오늘"
  "나는" [1.0,    0.0 ]   ← causal mask로 미래 차단
  "오늘" [0.6,    1.0 ]   ← "오늘"은 "나는"과 자신 모두 참조

Step 3: Softmax → Attention Weight
       "나는"  "오늘"
  "나는" [1.0,    0.0 ]
  "오늘" [0.35,   0.65]   ← "오늘"은 자신에 65%, "나는"에 35% 주목

Step 4: Value와 가중합
  Output = Attention Weight @ V   (2, 8)
  → "오늘" 위치의 출력은 "나는"의 V에 35%, "오늘"의 V에 65%를 반영

Step 5: 출력 변환
  Output = Output @ W_O   (2, 8)
```

### Multi-Head 구조

```
d_model=8, n_heads=2 → head_dim=4

head 0: Q[:, 0:4], K[:, 0:4], V[:, 0:4] → 문법적 관계 포착
head 1: Q[:, 4:8], K[:, 4:8], V[:, 4:8] → 의미적 관계 포착

각 head의 출력을 concat → W_O로 최종 변환
```

### Add & Norm (Residual + LayerNorm)

```
attention_output = LayerNorm(X + Attention(X))
                             ↑
                         원본 보존 (residual connection)
```

---

## Position-wise Feed-Forward Network (FFN)

### 역할

각 토큰의 표현을 개별적으로 더 풍부하게 변환합니다. Attention이 "어떤 정보를 모을지" 결정했다면, FFN은 "모은 정보를 어떻게 해석할지" 처리합니다.

### 구조

```
FFN(x) = GELU(x @ W1) @ W2

W1: (d_model × d_ff) = (8 × 32)   ← 4배 확장
W2: (d_ff × d_model) = (32 × 8)   ← 원래 크기로 축소
```

### 과정

```
입력: (2, 8) ← Attention 출력

Step 1: 확장 (8 → 32)
  hidden = x @ W1   (2, 32)
  → 8차원에서 표현 못하는 복잡한 패턴을 32차원에서 포착

Step 2: 비선형 활성화 (GELU/SiLU)
  hidden = GELU(hidden)   (2, 32)
  → 불필요한 뉴런을 억제, 중요한 특징만 활성화

Step 3: 축소 (32 → 8)
  output = hidden @ W2   (2, 8)
  → 활성화된 정보를 원래 차원으로 압축
```

### SwiGLU 변형 (LLaMA)

```
FFN_SwiGLU(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down

W_gate: (8 × 32)   ← 게이트 (어떤 정보를 통과시킬지)
W_up:   (8 × 32)   ← 정보 변환
W_down: (32 × 8)   ← 차원 축소

총 weight 3개 → FFN이 전체 모델 파라미터의 약 2/3 차지
```

### Add & Norm

```
ffn_output = LayerNorm(attention_output + FFN(attention_output))
```

---

## lm_head

### 역할

모델의 최종 hidden state를 vocab 크기의 logits 벡터로 변환합니다. 각 값은 해당 토큰이 다음에 올 가능성을 나타내는 점수(score)입니다.

### 구조

```
lm_head: Linear(d_model → vocab_size) = (8 → 7)
Weight: (vocab_size × d_model) = (7 × 8)
```

### 과정

```
입력: FFN 최종 출력 (2, 8) ← N개 layer를 거친 후

마지막 토큰 위치의 hidden state만 사용:
  h = [0.45, -0.23, 0.89, -0.12, 0.67, 0.34, -0.56, 0.78]   ← "오늘" 위치

logits = h @ lm_head_weight^T   (7,)

결과:
  [2.1,  0.3,  5.8,  1.2,  3.5,  0.1,  2.0]
  나는   오늘  학교에 갔다   밥을  먹었다  집에
               ↑ 가장 높은 점수 → "학교에"가 다음 토큰 후보
```

### Embedding과의 관계 (Weight Tying)

많은 모델(GPT-2, LLaMA 등)에서 lm_head의 weight를 Embedding 테이블과 공유합니다:

```
lm_head.weight == embedding.weight   (vocab_size × d_model)

이유: "토큰 → 벡터" 변환과 "벡터 → 토큰" 변환은 역관계
      같은 weight를 공유하면 파라미터 절약 + 일관성 향상
```

---

## 전체 데이터 흐름 (예제)

```
"나는 오늘" → tokens [0, 1]

1. Embedding:
   [0, 1] → [[0.12,-0.45,...], [0.89,-0.12,...]]     (2, 8)

2. Layer 0 - Attention:
   Q, K, V 계산 → Score → Softmax → 가중합              (2, 8)
   + Residual + LayerNorm

3. Layer 0 - FFN:
   확장(8→32) → GELU → 축소(32→8)                      (2, 8)
   + Residual + LayerNorm

4. Layer 1 - Attention:  (같은 구조, 다른 weight)
   ...                                                  (2, 8)

5. Layer 1 - FFN:
   ...                                                  (2, 8)

6. lm_head:
   마지막 위치 [0.45,-0.23,...] @ W → [2.1, 0.3, 5.8, 1.2, 3.5, 0.1, 2.0]
                                       ↓
7. Sampling:
   logits → softmax → [0.02, 0.003, 0.72, 0.007, 0.07, 0.002, 0.016]
   → "학교에" (72%) 선택
```

---

## 파라미터 규모 (LLaMA-7B 참고)

| 구성 요소 | Shape | 파라미터 수 | 비율 |
|----------|-------|-----------|------|
| Embedding | (32000 × 4096) | 131M | ~2% |
| Attention (per layer) | W_Q + W_K + W_V + W_O | 67M | |
| FFN (per layer) | W_gate + W_up + W_down | 135M | |
| Attention × 32 layers | | 2,147M | ~31% |
| FFN × 32 layers | | 4,295M | ~63% |
| lm_head | (32000 × 4096) | 131M (공유) | ~2% |
| **합계** | | **~6.7B** | |

---

## Sampling: Logits에서 다음 토큰 선택

### 전체 흐름

```
"나는 오늘" → [Prefill] → logits → [Sampling] → "학교에"
"학교에"    → [Decode]  → logits → [Sampling] → "갔다"
"갔다"      → [Decode]  → logits → [Sampling] → <eos>

결과: "나는 오늘 학교에 갔다"
```

### Logits → 확률 변환 (Softmax)

```
logits:        [2.1,   0.3,   5.8,   1.2,   3.5,   0.1,   2.0]
                                 ↓ softmax
probabilities: [0.02,  0.003, 0.72,  0.007, 0.07,  0.002, 0.016]
                               ↑ 72%
                "나는" "오늘" "학교에" "갔다" "밥을" "먹었다" "집에"
```

### Greedy (Argmax)

가장 높은 확률의 토큰을 선택합니다.

```
prob: [0.02, 0.003, 0.72, 0.007, 0.07, 0.002, 0.016]
                     ^^^^
선택: "학교에" (index 2, 72%)
```

- 항상 같은 결과 (결정적)
- 가장 빠르고 단순
- 반복적이고 단조로운 텍스트 생성 가능

### Temperature Scaling

logits를 temperature로 나누어 확률 분포를 조절합니다.

```
scaled_logits = logits / temperature
```

```
temp=0.5 (더 날카롭게, 확신 있게):
  prob: [0.005, 0.0002, 0.92, 0.001, 0.03, 0.0001, 0.004]
                         ^^^^ 92% → 거의 greedy

temp=1.0 (원본):
  prob: [0.02, 0.003, 0.72, 0.007, 0.07, 0.002, 0.016]

temp=2.0 (더 평탄하게, 다양하게):
  prob: [0.09, 0.05, 0.35, 0.07, 0.18, 0.04, 0.09]
                     ^^^^ 35% → 다른 토큰도 선택 가능
```

```
temp → 0: greedy와 동일
temp → ∞: 균등 분포 (완전 랜덤)
```

### Top-K Sampling

확률 상위 K개만 후보로 남기고 나머지를 제거합니다.

```
Top-K=3:
  원본:    [0.02, 0.003, 0.72, 0.007, 0.07, 0.002, 0.016]
  후보:    [  -,    -,   0.72,   -,   0.07,   -,    0.02 ]
                        학교에        밥을          집에
  재정규화: [0.89, 0.09, 0.02]  ← 이 3개 중에서 랜덤 선택
```

### Top-P (Nucleus) Sampling

누적 확률이 P를 넘을 때까지의 토큰만 후보로 사용합니다.

```
Top-P=0.9:
  확률 정렬: 학교에(0.72) → 밥을(0.07) → 나는(0.02) → 집에(0.016) → ...
  누적:      0.72          0.79         0.81          0.826

  → 갔다(0.007) 추가 → 누적 0.833 → 먹었다 추가 → 누적 0.835...
  → 0.9를 넘을 때까지 계속 추가

  결과: 상위 5~6개 토큰이 후보, 이 중 랜덤 선택
```

### 조합 (실제 사용)

```python
# LLaMA 등 실제 모델에서 흔한 조합
temperature = 0.7
top_p = 0.9

# 순서:
# 1) logits / temperature
# 2) softmax
# 3) top-p 필터링
# 4) 필터된 분포에서 랜덤 샘플링
```

### Decode 반복

선택된 토큰으로 다음 step 진행:

```
Step 0 (Prefill):
  Input: [0, 1] ("나는 오늘")
  → Embedding → Attention → FFN → lm_head → logits
  → sample → 2 ("학교에")

Step 1 (Decode):
  Input: [2] ("학교에") + KV cache
  → Embedding → Attention(+KV cache) → FFN → lm_head → logits
  → logits: [0.1, 0.05, 0.3, 4.8, 0.5, 0.2, 0.1]
  → sample → 3 ("갔다")

Step 2 (Decode):
  Input: [3] ("갔다")
  → ... → logits → sample → <eos>
  → 생성 종료

결과: "나는 오늘 학교에 갔다"
```

---

## C++ 코드에서의 구현

### Greedy (argmax)

```cpp
int argmax(const float* logits, size_t vocab_size) {
    return std::max_element(logits, logits + vocab_size) - logits;
}
```

### Temperature + Top-P Sampling

```cpp
int sample_top_p(const float* logits, size_t vocab_size,
                 float temperature, float top_p) {
    // 1) Temperature scaling
    std::vector<float> scaled(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        scaled[i] = logits[i] / temperature;
    }

    // 2) Softmax
    float max_val = *std::max_element(scaled.begin(), scaled.end());
    float sum = 0.0f;
    for (auto& v : scaled) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (auto& v : scaled) {
        v /= sum;
    }

    // 3) Sort by probability (descending)
    std::vector<std::pair<float, int>> prob_idx(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        prob_idx[i] = {scaled[i], (int)i};
    }
    std::sort(prob_idx.begin(), prob_idx.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // 4) Top-P filtering
    float cumsum = 0.0f;
    size_t cutoff = vocab_size;
    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += prob_idx[i].first;
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    // 5) Renormalize and sample
    float r = (float)rand() / RAND_MAX;
    cumsum = 0.0f;
    float renorm_sum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        renorm_sum += prob_idx[i].first;
    }
    for (size_t i = 0; i < cutoff; i++) {
        cumsum += prob_idx[i].first / renorm_sum;
        if (r <= cumsum) {
            return prob_idx[i].second;
        }
    }
    return prob_idx[0].second;
}
```

---

## Sampling 전략 비교

| 전략 | 다양성 | 품질 | 용도 |
|------|--------|------|------|
| Greedy | 없음 | 안정적 | 코드 생성, 사실 기반 |
| Temperature 낮음 (0.3) | 낮음 | 높음 | 정확한 답변 |
| Temperature 높음 (1.5) | 높음 | 불안정 | 창작, 브레인스토밍 |
| Top-K | 중간 | 중간 | 범용 |
| Top-P (0.9) | 적응적 | 높음 | 가장 널리 사용 |
| Temperature + Top-P | 적응적 | 높음 | LLaMA, GPT 등 실제 서비스 |
