# 🧑‍🏫 Knowledge Distillation: Teacher-Student Compression

## Table of Contents

- [[#1. 💡 Motivation: Compressing Soft Predictions|1. Motivation: Compressing Soft Predictions]]
- [[#2. 📐 Response-Based Distillation (Hinton et al.)|2. Response-Based Distillation (Hinton et al.)]]
  - [[#2.1 Soft Targets and Temperature Scaling|2.1 Soft Targets and Temperature Scaling]]
  - [[#2.2 The Distillation Loss|2.2 The Distillation Loss]]
  - [[#2.3 Dark Knowledge|2.3 Dark Knowledge]]
  - [[#2.4 💻 PyTorch: Distillation Training Loop|2.4 PyTorch: Distillation Training Loop]]
- [[#3. 🔗 Feature-Based Distillation|3. Feature-Based Distillation]]
  - [[#3.1 Intermediate Layer Matching (FitNets)|3.1 Intermediate Layer Matching (FitNets)]]
  - [[#3.2 Attention Transfer|3.2 Attention Transfer]]
  - [[#3.3 💻 PyTorch: Feature-Based Distillation|3.3 PyTorch: Feature-Based Distillation]]
- [[#4. ⚡ Distillation vs. Pruning|4. Distillation vs. Pruning]]
- [[#5. 📚 References|5. References]]

---

## 1. 💡 Motivation: Compressing Soft Predictions

A trained classifier doesn't just predict the correct class — its output distribution $p(y | x; T)$ encodes rich structural information: the model's *uncertainty* and the *similarity structure* of the classes. For example, a model trained on MNIST assigns small but nonzero probability to the digit "7" when shown a "1" — reflecting a genuine structural similarity that the hard one-hot label $y = 1$ throws away.

*Knowledge distillation* (Hinton, Vinyals, Dean 2015) exploits this: instead of training a small *student* network from hard labels, train it to match the *soft output distribution* of a large *teacher* network. The soft distribution carries more information per example than the hard label, enabling faster and more sample-efficient training of the student.

**Key contrast with pruning.** [[concepts/sparsity-pruning/classical-pruning|Pruning]] preserves the original architecture and removes weights. Distillation trains an entirely new (smaller) architecture to *mimic* the teacher's behavior. The student is not a subgraph of the teacher — it can have a completely different design.

| Dimension | Pruning | Distillation |
|-----------|---------|-------------|
| Student architecture | Sparse version of teacher | Free (any architecture) |
| Teacher needed at inference? | No | No |
| Requires teacher at train time? | No (magnitude); Yes (OBD/OBS) | Yes |
| Compression mechanism | Remove weights | Train smaller model |
| Best for | One architecture, high sparsity | Architecture flexibility |

---

## 2. 📐 Response-Based Distillation (Hinton et al.)

*Hinton, Vinyals, Dean (2015). "Distilling the Knowledge in a Neural Network." NeurIPS 2015 Deep Learning Workshop.*

### 2.1 Soft Targets and Temperature Scaling

Let $z_i$ be the *logits* of the teacher network for class $i$. The softmax at temperature $T$ is:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- At $T = 1$: standard softmax — the teacher's confident predictions.
- At $T > 1$ (*high temperature*): the distribution is *softer* — probabilities are more spread out across classes.

**Why raise the temperature?** At $T = 1$, the teacher often assigns $\approx 99\%$ probability to the correct class and $\approx 0\%$ to others — essentially a hard label with noise. At $T = 4$–$8$, the inter-class similarities become visible: a model trained on CIFAR-10 will assign noticeably higher probability to "cat" vs. "horse" for an image of a dog, even when both probabilities are small at $T = 1$. This *dark knowledge* is the signal distillation extracts.

### 2.2 The Distillation Loss

The student is trained with a combination of two objectives:

$$L_{distill} = (1 - \alpha)\, L_{hard}(\text{student}, y) + \alpha\, T^2\, \text{KL}(q^T_\text{teacher} \| q^T_\text{student})$$

- $L_{hard}$: standard cross-entropy with the ground-truth label $y$ (hard target).
- $\text{KL}(q^T_\text{teacher} \| q^T_\text{student})$: KL divergence between teacher and student soft distributions at temperature $T$.
- $T^2$ factor: compensates for the magnitude reduction of gradients at high temperature (gradients of the KL term scale as $1/T^2$, so multiplying by $T^2$ keeps the gradient magnitudes consistent with $L_{hard}$).

**Derivation of the $T^2$ factor.** The gradient of the soft cross-entropy w.r.t. student logit $a_i$ at temperature $T$ is:

$$\frac{\partial}{\partial a_i} \sum_j q_j^T \log p_j^T = \frac{1}{T}(p_i^T - q_i^T)$$

For the combined loss to balance correctly (neither term dominates), we need the KL term's gradients to be $O(1)$ like the hard CE term. Since KL gradients are $O(1/T)$ relative to $T=1$, multiplying by $T^2$ restores $O(T)$ — canceling with the $1/T$ factor and leaving $O(1)$ gradients overall.

> [!QUESTION] Exercise 1: Effect of temperature on the soft distribution
> *This exercise quantifies how temperature controls the information content of soft targets.*
>
> > **Prerequisites:** [[#2.1 Soft Targets and Temperature Scaling|2.1 Soft Targets and Temperature Scaling]]
>
> A 3-class teacher produces logits $z = (4.0, 1.0, 0.5)$.
>
> (a) Compute the softmax probabilities at $T = 1$, $T = 2$, and $T = 4$.
>
> (b) Compute the entropy $H(q^T) = -\sum_i q_i^T \log q_i^T$ at each temperature.
>
> (c) As $T \to \infty$, what does $q^T$ converge to, and what is its entropy?

> [!TIP]- Solution to Exercise 1
> **Key insight:** Higher temperature increases entropy — the soft distribution carries more information about inter-class relationships, which is the "dark knowledge."
>
> **(a)** At $T=1$: $z/T = (4, 1, 0.5)$, $e^z = (54.6, 2.72, 1.65)$, $Z = 58.97$. $q = (0.926, 0.046, 0.028)$.
>
> At $T=2$: $z/T = (2, 0.5, 0.25)$, $e^{z/T} = (7.39, 1.65, 1.28)$, $Z = 10.32$. $q = (0.716, 0.160, 0.124)$.
>
> At $T=4$: $z/T = (1, 0.25, 0.125)$, $e^{z/T} = (2.72, 1.28, 1.13)$, $Z = 5.13$. $q = (0.530, 0.250, 0.220)$.
>
> **(b)** Entropy: $T=1$: $H \approx 0.35$ bits. $T=2$: $H \approx 1.35$ bits. $T=4$: $H \approx 1.52$ bits.
>
> **(c)** As $T \to \infty$: $q_i^T = \exp(z_i/T) / \sum_j \exp(z_j/T) \to 1/K = 1/3$ for all $i$ (uniform distribution). Entropy $\to \log_2 3 \approx 1.585$ bits (maximum entropy for 3 classes). The logit differences become invisible — all inter-class structure is washed out.

### 2.3 Dark Knowledge

The term *dark knowledge* refers to the information encoded in the teacher's non-maximum class probabilities — the small but nonzero probabilities assigned to incorrect classes.

At $T = 1$, a model trained on MNIST might assign:
- $p(\text{"1"} | \text{image of "1"}) = 0.9999$
- $p(\text{"7"} | \text{image of "1"}) = 0.00005$
- $p(\text{"9"} | \text{image of "1"}) = 0.00003$

These tiny probabilities encode that "1" looks somewhat like "7" and "9" — structural similarity information that the hard label discards. At $T = 20$:
- $p(\text{"1"}) = 0.60$, $p(\text{"7"}) = 0.20$, $p(\text{"9"}) = 0.15$

Now the student can learn the similarity structure: "1" is more like "7" than like "5", a relation the hard label can never teach.

> [!INFO] Dark knowledge as a regularizer
> Hinton et al. showed that distillation from a teacher at temperature $T = 1$ alone (without hard labels) can train a student that *generalizes better* than the same student trained on hard labels from scratch. The soft targets act as a strong regularizer, preventing the student from overconfidently committing to incorrect-but-plausible classes.

### 2.4 💻 PyTorch: Distillation Training Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Hinton et al. distillation loss:
        L = (1 - alpha) * CE(student, hard) + alpha * T^2 * KL(teacher_soft || student_soft)

    Args:
        student_logits: (B, C) raw logits from student
        teacher_logits: (B, C) raw logits from teacher (no grad needed)
        hard_targets: (B,) integer class labels
        temperature: T; higher = softer distribution
        alpha: weight on the distillation term (1 - alpha on hard targets)

    Returns:
        scalar loss
    """
    # Hard target loss
    hard_loss = F.cross_entropy(student_logits, hard_targets)

    # Soft target loss at temperature T
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: sum(teacher * (log_teacher - log_student))
    # F.kl_div expects log-probs for input, probs for target
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

    # Scale by T^2 to compensate for gradient magnitude reduction
    return (1 - alpha) * hard_loss + alpha * temperature**2 * soft_loss


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = "cuda",
) -> float:
    student.train()
    teacher.eval()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        student_logits = student(inputs)
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        loss = distillation_loss(
            student_logits, teacher_logits, targets,
            temperature=temperature, alpha=alpha
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)
```

---

## 3. 🔗 Feature-Based Distillation

Response-based distillation only matches the *final output layer*. Feature-based methods transfer knowledge from *intermediate layers*, providing a richer training signal.

### 3.1 Intermediate Layer Matching (FitNets)

*Romero et al. (2015). "FitNets: Hints for Thin Deep Nets." ICLR 2015.*

FitNets matches intermediate activation maps ("hints") between teacher and student:

$$L_{hint} = \frac{1}{2} \|r(F_s(x)) - F_t(x)\|_F^2$$

where $F_s(x)$ and $F_t(x)$ are intermediate feature maps from the student and teacher, and $r(\cdot)$ is a *regressor* (learned linear or convolutional layer) that projects the student's features to the teacher's dimensionality.

**Two-stage training:**
1. **Hint training:** Train the student up to the hint layer by minimizing $L_{hint}$, using the teacher's intermediate features as supervision.
2. **Knowledge distillation:** Fine-tune the full student network using both soft targets ($L_{distill}$) and task loss ($L_{hard}$).

### 3.2 Attention Transfer

*Zagoruyko & Komodakis (2017). "Paying More Attention to Attention." ICLR 2017.*

Rather than matching raw feature maps, attention transfer matches *attention maps* — summary statistics of feature importance across spatial positions:

$$A(F) = \vec{\|F\|}^p \in \mathbb{R}^{H \times W}$$

where $F \in \mathbb{R}^{C \times H \times W}$ is a convolutional feature map, and $\|\cdot\|$ takes the norm over channels. The attention transfer loss:

$$L_{AT} = \sum_{l \in \text{layers}} \left\| \frac{A_l^s}{\|A_l^s\|_2} - \frac{A_l^t}{\|A_l^t\|_2} \right\|_2^2$$

Attention transfer is lighter than FitNets (no regressor, no dimension matching) and competitive on classification benchmarks.

### 3.3 💻 PyTorch: Feature-Based Distillation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class HintLoss(nn.Module):
    """
    FitNets-style hint loss: L = ||r(F_student) - F_teacher||^2
    The regressor r is a 1x1 conv that matches dimensions.
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.regressor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.regressor(student_feat)
        return F.mse_loss(projected, teacher_feat.detach())


def attention_map(feat: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Compute spatial attention map from a feature tensor.
    feat: (B, C, H, W) -> returns (B, H*W) normalized attention map.
    """
    attn = feat.pow(p).sum(dim=1)  # (B, H, W)
    attn = attn.view(attn.size(0), -1)  # (B, H*W)
    return F.normalize(attn, p=2, dim=1)


def attention_transfer_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
    p: int = 2,
) -> torch.Tensor:
    """Attention transfer loss summed over matched feature layers."""
    loss = torch.tensor(0.0, device=student_feats[0].device)
    for fs, ft in zip(student_feats, teacher_feats):
        as_ = attention_map(fs, p)
        at = attention_map(ft.detach(), p)
        loss += (as_ - at).pow(2).mean()
    return loss
```

---

## 4. ⚡ Distillation vs. Pruning

Distillation and pruning are complementary compression strategies, often combined:

**Distillation advantages:**
- Architecture flexibility: student can be a completely different design (MobileNet distilled from ResNet, DistilBERT from BERT).
- No sparse kernel requirements: student is dense, runs on standard hardware.
- Can transfer "reasoning style" not just weights.

**Pruning advantages:**
- Preserves the original architecture's inductive biases.
- Better at extreme compression ($>90\%$ parameter reduction) within the same architecture class.
- No teacher needed at inference time *or* training time (for magnitude pruning).

**Combined approaches:** DistilBERT (Sanh et al. 2020) combines distillation + magnitude pruning: first distill BERT-base → 66M parameters, then prune the student further.

> [!WARNING] Distillation does not always outperform direct training
> For structured pruning, Liu et al. (2019) showed that *training the pruned architecture from scratch* (without a teacher) matches or beats fine-tuning the pruned weights. By extension, distillation from a teacher to the pruned architecture often adds only marginal gains over random initialization of the pruned architecture — the architecture itself, not the knowledge transfer, is the dominant factor.

---

## 5. 📚 References

| Reference Name | Brief Summary | Link |
|----------------|---------------|------|
| Hinton, Vinyals, Dean (2015). "Distilling the Knowledge in a Neural Network" | Temperature-scaled soft targets; dark knowledge; KD training loss derivation | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| Romero et al. (2015). "FitNets" | Intermediate hint-layer matching; two-stage distillation | [arXiv:1412.6550](https://arxiv.org/abs/1412.6550) |
| Zagoruyko & Komodakis (2017). "Paying More Attention to Attention" | Attention map transfer; lightweight alternative to FitNets | [arXiv:1612.03928](https://arxiv.org/abs/1612.03928) |
| Sanh et al. (2019). "DistilBERT" | 40% smaller BERT via distillation + pruning; 97% of BERT performance | [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) |
| Gou et al. (2021). "Knowledge Distillation: A Survey" | Comprehensive taxonomy of response-based, feature-based, and relation-based KD | [arXiv:2006.05525](https://arxiv.org/abs/2006.05525) |
