# Transformer Sampling: 예제 기반 이해

## 예제 설정

```
프롬프트: "나는 오늘"
vocab: {0:"나는", 1:"오늘", 2:"학교에", 3:"갔다", 4:"밥을", 5:"먹었다", 6:"집에"}
```

## 전체 흐름

```
"나는 오늘" → [Prefill] → logits → [Sampling] → "학교에"
"학교에"    → [Decode]  → logits → [Sampling] → "갔다"
"갔다"      → [Decode]  → logits → [Sampling] → <eos>

결과: "나는 오늘 학교에 갔다"
```

## Step 1: Prefill

모든 입력 토큰을 한 번에 처리합니다.

```
Input tokens: [0, 1]  ("나는", "오늘")

→ Embedding → Attention → FFN → ... → lm_head

Output logits (마지막 토큰 위치):
  [2.1, 0.3, 5.8, 1.2, 3.5, 0.1, 2.0]
   나는  오늘 학교에 갔다  밥을  먹었다 집에
```

## Step 2: Logits → 확률 변환 (Softmax)

```
logits:        [2.1,   0.3,   5.8,   1.2,   3.5,   0.1,   2.0]
                                 ↓ softmax
probabilities: [0.02,  0.003, 0.72,  0.007, 0.07,  0.002, 0.016]
                               ↑ 72%
                "나는" "오늘" "학교에" "갔다" "밥을" "먹었다" "집에"
```

## Step 3: Sampling 전략

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

## Step 4: Decode (반복)

선택된 토큰으로 다음 step 진행:

```
Step 0 (Prefill):
  Input: [0, 1] ("나는 오늘")
  → logits → sample → 2 ("학교에")

Step 1 (Decode):
  Input: [2] ("학교에") + KV cache (or SSM state)
  → logits: [0.1, 0.05, 0.3, 4.8, 0.5, 0.2, 0.1]
  → prob:   [0.01, 0.01, 0.02, 0.85, 0.04, 0.02, 0.01]
  → sample → 3 ("갔다")

Step 2 (Decode):
  Input: [3] ("갔다")
  → logits → sample → <eos>
  → 생성 종료

결과: "나는 오늘 학교에 갔다"
```

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

## Sampling 전략 비교

| 전략 | 다양성 | 품질 | 용도 |
|------|--------|------|------|
| Greedy | 없음 | 안정적 | 코드 생성, 사실 기반 |
| Temperature 낮음 (0.3) | 낮음 | 높음 | 정확한 답변 |
| Temperature 높음 (1.5) | 높음 | 불안정 | 창작, 브레인스토밍 |
| Top-K | 중간 | 중간 | 범용 |
| Top-P (0.9) | 적응적 | 높음 | 가장 널리 사용 |
| Temperature + Top-P | 적응적 | 높음 | LLaMA, GPT 등 실제 서비스 |
