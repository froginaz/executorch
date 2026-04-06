# Transformer Sampling: Understanding Through Examples

## Example Setup

```
Prompt: "I went to"
vocab: {0:"I", 1:"went", 2:"to", 3:"school", 4:"eat", 5:"the", 6:"store", 7:"today"}
```

## End-to-End Flow

```
"I went to" → [Prefill] → logits → [Sampling] → "school"
"school"    → [Decode]  → logits → [Sampling] → "today"
"today"     → [Decode]  → logits → [Sampling] → <eos>

Result: "I went to school today"
```

## Step 1: Prefill

Process all input tokens at once.

```
Input tokens: [0, 1, 2]  ("I", "went", "to")

→ Embedding → Attention → FFN → ... → lm_head

Output logits (at last token position):
  [0.3, 0.1, 0.2, 5.8, 1.2, 3.5, 2.0, 0.5]
    I   went  to  school eat  the  store today
```

## Step 2: Logits → Probability (Softmax)

```
logits:        [0.3,   0.1,   0.2,   5.8,   1.2,   3.5,   2.0,   0.5]
                                       ↓ softmax
probabilities: [0.003, 0.002, 0.003, 0.72,  0.007, 0.07,  0.016, 0.004]
                                      ↑ 72%
                 I     went    to    school  eat    the    store  today
```

## Step 3: Sampling Strategies

### Greedy (Argmax)

Select the token with the highest probability.

```
prob: [0.003, 0.002, 0.003, 0.72, 0.007, 0.07, 0.016, 0.004]
                              ^^^^
Selected: "school" (index 3, 72%)
```

- Always produces the same result (deterministic)
- Fastest and simplest
- Can produce repetitive, monotonous text

### Temperature Scaling

Divide logits by temperature to reshape the probability distribution.

```
scaled_logits = logits / temperature
```

```
temp=0.5 (sharper, more confident):
  prob: [0.001, 0.0005, 0.001, 0.92, 0.001, 0.03, 0.004, 0.001]
                                ^^^^ 92% → nearly greedy

temp=1.0 (original):
  prob: [0.003, 0.002, 0.003, 0.72, 0.007, 0.07, 0.016, 0.004]

temp=2.0 (flatter, more diverse):
  prob: [0.06, 0.05, 0.06, 0.35, 0.07, 0.18, 0.09, 0.06]
                            ^^^^ 35% → other tokens become viable
```

```
temp → 0: equivalent to greedy
temp → ∞: uniform distribution (completely random)
```

### Top-K Sampling

Keep only the top K tokens by probability, discard the rest.

```
Top-K=3:
  original:     [0.003, 0.002, 0.003, 0.72, 0.007, 0.07, 0.016, 0.004]
  candidates:   [  -,     -,     -,   0.72,   -,   0.07,  0.016,   -  ]
                                      school       the    store
  renormalized: [0.89, 0.09, 0.02]  ← randomly sample from these 3
```

### Top-P (Nucleus) Sampling

Include tokens until their cumulative probability exceeds P.

```
Top-P=0.9:
  sorted by prob: school(0.72) → the(0.07) → store(0.016) → eat(0.007) → ...
  cumulative:     0.72           0.79        0.806           0.813

  → keep adding until cumulative >= 0.9
  → today(0.004) → I(0.003) → to(0.003) → cumulative 0.823...

  Result: top 5-6 tokens become candidates, randomly sample from them
```

### Combined Strategy (Real-World Usage)

```python
# Common combination used in LLaMA, GPT, etc.
temperature = 0.7
top_p = 0.9

# Order of operations:
# 1) logits / temperature
# 2) softmax
# 3) top-p filtering
# 4) random sampling from filtered distribution
```

## Step 4: Decode (Iterative)

Feed the selected token back and repeat:

```
Step 0 (Prefill):
  Input: [0, 1, 2] ("I went to")
  → logits → sample → 3 ("school")

Step 1 (Decode):
  Input: [3] ("school") + KV cache (or SSM state)
  → logits: [0.1, 0.05, 0.2, 0.3, 0.1, 0.5, 0.2, 4.8]
  → prob:   [0.01, 0.01, 0.01, 0.02, 0.01, 0.04, 0.01, 0.85]
  → sample → 7 ("today")

Step 2 (Decode):
  Input: [7] ("today")
  → logits → sample → <eos>
  → generation complete

Result: "I went to school today"
```

## C++ Implementation

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

## Sampling Strategy Comparison

| Strategy | Diversity | Quality | Use Case |
|----------|-----------|---------|----------|
| Greedy | None | Stable | Code generation, factual Q&A |
| Low Temperature (0.3) | Low | High | Precise answers |
| High Temperature (1.5) | High | Unstable | Creative writing, brainstorming |
| Top-K | Medium | Medium | General purpose |
| Top-P (0.9) | Adaptive | High | Most widely used |
| Temperature + Top-P | Adaptive | High | LLaMA, GPT production systems |
