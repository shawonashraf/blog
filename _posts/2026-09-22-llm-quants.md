---
title: "Understanding LLM Quantisation Methods"
date: 2026-09-22
permalink: /posts/2026-09-22-llm-quants
tags: 
    - llm

---


Large language models place substantial demands on memory, computation and data movement. Their parameters must be stored, their intermediate representations must be processed, and their attention histories must remain accessible throughout generation. Quantisation reduces these costs by representing selected numerical values with fewer bits. Its practical value is considerable: a model may fit on fewer accelerators, support more concurrent requests, or operate within the memory limits of a personal computer.

These benefits do not follow from bit width alone. Two models described as “4-bit” may use different numerical representations, retain different components at higher precision and require entirely different inference kernels. Their quality and performance can therefore differ substantially.

Understanding quantisation requires connecting three questions: **what is compressed, how the resulting numerical error is controlled, and how the compressed representation is executed**. This article develops that framework, explains the principal method families, and sets out a practical approach to evaluating them. It focuses on established techniques and their underlying principles rather than an exhaustive ranking of recent implementations.

## 1. What quantisation changes

The memory occupied by an LLM extends beyond its parameter tensors. During inference, it also includes intermediate activations, temporary workspaces and, for conventional autoregressive Transformers, a key–value cache. These components respond differently to quantisation.

**Weight quantisation** compresses the learned parameters. It directly reduces checkpoint size and the amount of weight data that must be transferred during execution. **Activation quantisation** reduces the precision of intermediate values and can enable low-precision matrix multiplication. **KV-cache quantisation** compresses the keys and values retained from earlier tokens, addressing a memory cost that grows with sequence length and concurrency.

The notation W4A16 denotes four-bit weights and 16-bit activations; W8A8 denotes eight-bit weights and activations. A label such as W4A8KV4 additionally specifies a four-bit cache. These labels remain incomplete specifications: they do not identify integer versus floating-point encoding, scaling granularity, accumulation precision, or operations retained at higher precision.

For an illustrative model with exactly eight billion parameters, the idealised weight payload is:

| Representation | Calculation | Weight payload |
| --- | --- | --- |
| 16 bits per parameter | 8 billion × 16 ÷ 8 | 16 GB |
| 8 bits per parameter | 8 billion × 8 ÷ 8 | 8 GB |
| 4 bits per parameter | 8 billion × 4 ÷ 8 | 4 GB |

These are decimal gigabytes, calculated before scales, zero-points, alignment and any higher-precision tensors. They are neither complete checkpoint sizes nor estimates of peak device memory. A model with a nominal four-gigabyte weight payload will require additional memory to run.

## 2. The numerical foundations

### Mapping continuous values to discrete levels

A uniform affine quantiser maps a real value \(x\) to an integer code:

\[
q=\operatorname{clip}\!\left(
\operatorname{round}(x/s)+z,\ q_{\min},\ q_{\max}
\right),
\qquad
\hat{x}=s(q-z).
\]

Here, \(s>0\) is the scale, \(z\) is an integer zero-point, and the integer bounds define the available codes. The reconstructed value \(\hat{x}\) approximates the original. Symmetric quantisation normally centres the representation around zero; asymmetric quantisation permits an offset to better fit an uneven range. [1]

As a simple example, consider a symmetric grid with scale 0.1. The value 0.26 rounds to code 3 and reconstructs as 0.3. Its absolute error is 0.04. Away from clipping, nearest rounding on a uniform grid has an absolute error no greater than half the step size. A value outside the representable range is clipped, potentially producing a much larger error.

This creates a fundamental compromise. A wider range accommodates extreme values but leaves fewer representable levels for the dense central part of the distribution. A narrower range improves resolution for ordinary values while sacrificing some extremes.

### Granularity and effective bit width

Quantisation parameters can be shared across an entire tensor, assigned to individual channels, or computed for smaller groups. Activation scales can also be computed separately for each token. More local scaling allows the representation to adapt to variation within a tensor, at the expense of metadata and additional implementation work. [1]

Suppose each group of \(g\) weights contains \(b\)-bit codes and one 16-bit scale. Ignoring all other overhead, the effective storage is:

\[
b_{\mathrm{effective}}=b+\frac{16}{g}.
\]

For four-bit weights, this gives 4.125 bits per weight at group size 128 and 4.5 bits at group size 32. Adding zero-points or other block parameters increases the total. This arithmetic explains why a nominal bit width is insufficient for comparing file sizes.

The relevant accuracy objective also extends beyond reconstructing individual weights. For a linear layer \(Y=WX\), a common target is:

\[
\min_{\hat{W}}\left\|WX-\hat{W}X\right\|_F^2.
\]

Here, the columns of \(X\) contain representative input activations. A weight perturbation matters according to the inputs it processes, which motivates methods that use activation statistics or layer reconstruction rather than weight error alone. GPTQ is a prominent example. [3]

### Static and dynamic activation scales

Static activation quantisation determines scales before deployment, typically from calibration data. Dynamic quantisation computes scales from the values encountered during inference. Static scales reduce runtime work; dynamic scales adapt to changing inputs but introduce reductions, conversions and metadata handling. Both approaches can use different granularities, so “dynamic” and “per-token” are related implementation choices rather than interchangeable definitions. [1]

## 3. Post-training quantisation and quantisation-aware training

Post-training quantisation, or PTQ, begins with an existing model and constructs a compressed representation. Simple methods use weight statistics alone. More sophisticated methods collect representative activations, search for scales and clipping thresholds, or compensate for reconstruction error.

PTQ should not be defined as universally excluding optimisation or gradients. OmniQuant, for example, optimises clipping and equivalent-transformation parameters through a calibration procedure while retaining the PTQ setting. The more useful distinction concerns what is updated, the objective being optimised, and the amount of training involved. [2]

Quantisation-aware training, or QAT, exposes trainable model parameters to quantisation effects during training or fine-tuning. A typical implementation inserts fake quantisation: values are rounded and clipped in the forward pass while remaining in floating-point tensors. Approximate gradients allow learning to proceed through these otherwise non-differentiable operations. [4]

A simplified straight-through estimator can be written as:

\[
\tilde{w}=w+\operatorname{stopgrad}\bigl(Q(w)-w\bigr),
\]

where \(Q\) includes quantisation and reconstruction. The forward value equals \(Q(w)\), while the simplified backward derivative with respect to \(w\) is one. Practical estimators may additionally mask gradients outside clipping bounds.

QAT can recover quality that PTQ fails to preserve, but it requires training resources and suitable data. It does not inherently require pretraining from scratch, and its training memory need not resemble the final compressed inference footprint. Four-bit activation quantisation also does not automatically imply that QAT is mandatory: transformation-based PTQ methods can support such configurations under appropriate conditions. [4, 14]

## 4. Weight quantisation: rounding, GPTQ and AWQ

### Round-to-nearest as a baseline

Round-to-nearest quantisation selects a scale and maps each weight independently to a nearby representable value. It is inexpensive and provides an essential reference point. Its limitation is that independent rounding does not account for how errors interact in a layer’s output.

Before adopting a more elaborate algorithm, practitioners should establish how much it improves on a carefully configured rounding baseline. The comparison should hold bit width, group size, tensor exclusions and evaluation settings constant.

### GPTQ: compensating for rounding error

GPTQ uses second-order information to preserve layer outputs while quantising weights sequentially. For the reconstruction objective above, the Hessian for a weight row is proportional to the input Gram matrix:

\[
H=2XX^\top.
\]

This is the Hessian of the layer’s quadratic reconstruction problem, rather than the full language-model training loss. When GPTQ rounds a weight or column, it adjusts the remaining unquantised weights to compensate for the resulting output error. Block processing and a Cholesky-based formulation make this procedure practical for large models. [3]

The reference implementation stabilises the computation by adding damping to the Hessian diagonal, scaled relative to its mean diagonal value. Damping addresses poor conditioning and numerical instability; it is not a universal, dimensionless constant that can be applied unchanged to every raw Hessian. [5]

GPTQ’s main attraction is accurate low-bit weight compression without full-model retraining. Calibration coverage, grouping and implementation choices remain relevant, and its reconstruction objective does not guarantee preservation of every downstream behaviour.

### AWQ: protecting influential channels through scaling

Activation-aware Weight Quantisation, or AWQ, uses activation magnitudes to identify channels whose weight errors are especially consequential. It protects those channels through rescaling before quantisation. With a positive diagonal matrix \(S\):

\[
Wx=(WS)(S^{-1}x).
\]

Before quantisation, this transformation leaves the linear operation unchanged. AWQ searches for scales that improve the quantised result, increasing selected weight-channel magnitudes while inversely scaling the associated activations. Under group quantisation, this can reduce those channels’ effective error after compensation, provided the shared quantisation step does not increase excessively. [6]

The original method combines scale search with weight clipping. Its protection mechanism does not require retaining the salient weights in a separate floating-point representation. “Activation-aware” describes the information used to quantise the weights; it does not mean that activations themselves must be stored at four bits.

AWQ and GPTQ pursue different error-control strategies. Neither method name establishes a universal advantage for coding, reasoning or chat. Such a conclusion requires a matched experiment on the model and workload being deployed.

## 5. Quantising activations: outliers, LLM.int8() and SmoothQuant

Activations are input-dependent and may contain channels with unusually large magnitudes. If one shared scale must accommodate these outliers, ordinary values receive relatively coarse resolution. Two influential methods illustrate different responses.

### LLM.int8(): separate treatment for outliers

LLM.int8() combines vector-wise eight-bit quantisation with a higher-precision path for outlier feature dimensions. Most of the matrix multiplication uses eight-bit representations, while the numerically sensitive contribution is computed separately and combined with the result. [7]

This is a deliberate mixed-precision design. Its significance is that a small subset of difficult values need not dictate the precision of the entire operation. Its practical benefit nevertheless depends on the cost of identifying, extracting and processing that subset.

### SmoothQuant: redistribute the dynamic range

SmoothQuant uses an equivalent channel-scaling transformation to reduce activation outliers while allowing weights to absorb more of the dynamic range. For the column-vector convention used here, its structure is again \(Wx=(WS)(S^{-1}x)\), but the scales target weight-and-activation quantisation. [8]

A representative channel scale is:

\[
s_j=
\frac{\bigl(\max |X_j|\bigr)^\alpha}
{\bigl(\max |W_{:,j}|\bigr)^{1-\alpha}},
\]

with suitable handling of zero-valued channels. Activation maxima are collected over calibration inputs. The parameter \(\alpha\) controls how strongly the transform reduces activation magnitude relative to its effect on weights.

SmoothQuant was developed to enable W8A8 integer inference. Its central contribution is the redistribution of quantisation difficulty between tensors whose product must remain unchanged. AWQ uses related algebra, but optimises a different target: preserving important channels under weight-only quantisation.

The transformation remains exact only before quantisation, apart from floating-point rounding. The final approximation must still be evaluated, especially when deployment inputs differ from the calibration distribution.

## 6. Floating-point and non-uniform representations

Integer codes are only one way to allocate a limited number of representable values. Floating-point formats allocate resolution across magnitudes through an exponent, while codebook-based representations can place levels according to a target distribution.

| Representation | Bits per element, excluding shared metadata | Numerical structure |
| --- | --- | --- |
| FP32 | 32 | 1 sign, 8 exponent and 23 fraction bits |
| FP16 | 16 | 1 sign, 5 exponent and 10 fraction bits |
| BF16 | 16 | 1 sign, 8 exponent and 7 fraction bits |
| INT8 | 8 | 256 integer codes interpreted using quantisation parameters |
| INT4 | 4 | 16 integer codes interpreted using quantisation parameters |
| FP8 E4M3 / E5M2 | 8 | Floating-point encodings with different range–precision trade-offs |
| NF4 | 4 | 16 non-uniform codebook levels with block scaling |

BF16 retains FP32's exponent width, giving it a broadly similar dynamic range, while FP16 allocates more bits to the fraction. BF16 therefore offers greater range, and FP16 offers finer relative precision for normal values within its range. This is a distinction between range and precision, rather than a general claim that one format represents small values more accurately. [9]

### FP8: range and precision in eight bits

Two widely used FP8 encodings are E4M3 and E5M2. Their names specify the exponent and fraction-bit counts, in addition to a sign bit. In the encodings proposed by Micikevicius and colleagues, E4M3 has a maximum finite magnitude of 448; E5M2 extends this to 57,344 while providing fewer fraction bits. Exact special-value behaviour depends on the format variant. [9]

FP8 still requires an appropriate scaling strategy. Values can overflow, underflow or round to coarse approximations, and higher-precision accumulation remains a separate implementation choice. Evidence that a particular FP8 configuration matches a baseline on selected evaluations does not make conversion mathematically lossless.

FP8 is therefore best understood as a numerical format within a quantisation recipe. The recipe also includes scaling, tensor exclusions, calibration or dynamic statistics, and execution support.

### NF4 and QLoRA: efficient adaptation

NormalFloat 4, or NF4, uses **16 representable levels** selected for normally distributed weights. It is a non-uniform codebook representation, rather than a conventional four-bit format with exponent and fraction fields. Blocks of weights are scaled and mapped to those levels. [10]

QLoRA uses a frozen, quantised base model while training low-rank adapters. Its design includes NF4, double quantisation of quantisation constants, and paged optimisers to manage memory spikes. Computation can use dequantised values at higher precision, so four-bit storage does not imply four-bit arithmetic throughout the training process.

This makes QLoRA an adaptation method with a compressed base model. It is distinct from QAT that updates model parameters to prepare a specified deployment quantiser. Adapter training also does not establish which inference representation will be fastest after fine-tuning. Merging and re-quantising an adapter can change the approximation and should be evaluated as a new deployment candidate.

### MXFP4 and NVFP4: block-scaled floating point

Very small floating-point values rely heavily on shared scales. MXFP4 combines E2M1 elements with a power-of-two scale shared across 32 values. NVIDIA’s NVFP4 uses E2M1 elements with an E4M3 scale for each 16-value block and an additional per-tensor FP32 scale in its described inference representation. [11]

The NVFP4 block scale alone raises effective storage to:

\[
4+\frac{8}{16}=4.5\text{ bits per value},
\]

before the tensor scale and other overhead. These formats demonstrate why the encoding of scales can be as consequential as the encoding of individual elements. Their acceleration depends on compatible hardware and kernels; neither a four-bit label nor a favourable result on one model establishes a universal accuracy or speed guarantee.

## 7. GGUF and block quantisation in llama.cpp

GGUF is a model file format containing tensor data and metadata. It can contain tensors with different numerical encodings; it is not itself a single quantisation algorithm. The llama.cpp runtime provides execution across several hardware backends, including CPU and GPU configurations. File representation, quantisation policy and execution backend are therefore separate parts of the deployment. [12, 13]

Two block formats illustrate the storage arithmetic. In Q4_0, a block holds 32 four-bit codes plus a 16-bit scale: 18 bytes for 32 weights, or 4.5 bits per weight. In Q4_K, a 256-weight super-block holds 128 bytes of codes, 12 bytes of packed sub-block scale/minimum metadata, and two 16-bit super-block parameters. The total is 144 bytes, also 4.5 bits per weight. [13]

Equal storage does not make the schemes equivalent. Q4_K’s hierarchical parameters represent local variation differently. A model-level preset such as Q4_K_M may additionally assign different tensor types across the model, so its overall footprint need not equal the payload rate of a single Q4_K tensor. [13, 20]

An importance matrix, collected from representative activations, can guide the quantisation objective towards errors that matter more during execution. Conceptually, a diagonal approximation weights input-channel errors by their mean squared activations. This does not imply that every important weight is automatically allocated extra bits: importance can influence the choice of codes within a fixed representation. [20]

Some low-bit IQ formats use codebook-based encodings, while others use non-linear scalar levels. The specific format matters; the entire IQ family should not be described as one uniform vector-quantisation algorithm. [13]

## 8. Rotations and vector quantisation

Channel scaling changes the magnitude of coordinates. An orthogonal rotation changes the coordinate system while preserving vector length. If \(R^\top R=I\), then:

\[
Wx=(WR^\top)(Rx).
\]

A suitable rotation can distribute a large coordinate across several dimensions, making the representation less dominated by isolated outliers. Applying this idea throughout a Transformer requires respecting residual connections, normalisation and attention; arbitrary rotations cannot simply be inserted everywhere.

**QuaRot** uses structured rotations, including randomised Hadamard transforms, to make low-bit weights, activations and KV caches more tractable. Its four-bit approach combines transformations with quantisation machinery; it should not be characterised as universally calibration-free round-to-nearest conversion. The original implementation includes a GPTQ-based weight-quantisation route. [14]

**QuIP and QuIP#** focus on extreme weight compression. QuIP uses incoherence processing to make weight and curvature structure more favourable for quantisation. QuIP# combines randomised Hadamard processing with vector quantisation using structured lattice codebooks, alongside fine-tuning to improve fidelity. Vector quantisation selects a code for a small vector jointly, allowing it to exploit structure that independent scalar rounding misses. [15, 16]

**SpinQuant** learns rotations to improve the quantised model rather than relying entirely on fixed random choices. Its optimisation targets the variation in quantisation quality that different rotations can produce. [17]

These approaches broaden the design space beyond selecting scales. They also introduce engineering questions: which transforms can be folded into weights, which must execute online, and whether the resulting representation has efficient kernels on the target device.

## 9. KV-cache quantisation and long-context inference

Weight compression provides a largely fixed saving. A conventional KV cache grows with the number of retained tokens and active sequences. For a uniform Transformer configuration, its idealised payload is:

\[
M_{\mathrm{KV}}=
2BLTH_{\mathrm{KV}}D_{\mathrm{head}}p,
\]

where \(B\) is the number of sequences, \(L\) the layer count, \(T\) the retained tokens per sequence, \(H_{\mathrm{KV}}\) the KV-head count, \(D_{\mathrm{head}}\) the head dimension, and \(p\) the bytes per cached element. The leading factor accounts for both keys and values.

For an illustrative configuration with one sequence, 32 layers, eight KV heads, head dimension 128 and 8,192 retained tokens, a 16-bit cache occupies exactly 1 GiB. At 131,072 tokens it occupies 16 GiB. An ideal eight-bit payload halves these figures; an ideal four-bit payload quarters them. Scales, residual high-precision regions, allocator behaviour and temporary buffers reduce the realised saving.

This calculation is architecture-specific. Grouped-query attention uses fewer KV heads than query heads. Sliding-window attention, latent-attention designs and heterogeneous layer configurations require different accounting. Parameter count alone cannot determine cache size.

**KIVI** shows why keys and values may benefit from different quantisation granularities. Its design quantises keys per channel and values per token, with a higher-precision residual region for recent entries. The method supports very low-bit cache storage in its evaluated settings, but its results do not imply uniform robustness across architectures or context lengths. [18]

**QServe** studies a combined W4A8KV4 system. Its SmoothAttention component addresses key-cache outliers through a function-preserving rescaling of queries and keys. QServe also addresses the execution overhead of low-bit representations, illustrating that cache compression and serving efficiency must be designed together. [19]

Cache evaluation should include long-context retrieval and tasks requiring information from distant positions. A small short-context perplexity change is insufficient evidence that a compressed cache preserves long-context behaviour.

## 10. Ternary and binary models

Extreme low-bit models move beyond compressing conventional checkpoints towards learning with restricted representations. BitNet b1.58 uses ternary weights drawn from \(\{-1,0,+1\}\) in its targeted linear layers. The name refers to \(\log_2 3\approx1.585\), the information needed to distinguish three equally likely states under ideal coding. [21]

This figure is not a guarantee of complete model storage. Scales, packing, higher-precision components and runtime buffers still contribute. Nor does success after quantisation-aware training imply that an arbitrary BF16 checkpoint will retain its capabilities if its weights are directly rounded to three values.

Binary and ternary representations can simplify arithmetic, but exploiting them efficiently requires suitable packing and kernels. Their significance lies in the possibility of training models whose learned representations suit inexpensive computation from the outset. Their deployment properties must be evaluated as a complete model-and-runtime design.

## 11. Why fewer bits do not guarantee faster inference

Quantisation affects both memory traffic and computation. During low-batch autoregressive decoding, substantial weight data may be read to produce each token. Reducing that traffic can improve speed. During prompt processing, or prefill, many tokens are processed together, increasing weight reuse and changing the balance between memory and compute. Long-context attention can shift the bottleneck again.

An effective quantised kernel unpacks codes, applies scales and performs matrix multiplication with limited intermediate data movement. Marlin, for example, coordinates low-bit weight loading and dequantisation with tensor-core computation. Its design shows why storage savings need carefully engineered execution to become runtime gains. [22]

A less suitable path may materialise large dequantised tensors, launch many small operations, or fall back to a slower kernel for unsupported shapes. Consequently, a smaller checkpoint can deliver disappointing latency.

Benchmark prefill and decoding separately. Record prompt and output lengths, batch size, concurrency, device model, software version, kernel path and cache precision. For interactive systems, time to first token and inter-token latency are often more informative than aggregate tokens per second. For services, throughput should be measured at an acceptable latency target.

## 12. Choosing and evaluating a quantisation method

The following comparison identifies useful starting points. It is a synthesis of the mechanisms described above, rather than a universal ranking or a hardware-compatibility guarantee.

| Primary objective | Methods or configurations to investigate | Main evaluation question |
| --- | --- | --- |
| Reduce weight memory with limited conversion effort | RTN, GPTQ, AWQ | Does the compressed checkpoint preserve the required tasks? |
| Accelerate matrix multiplication with low-precision activations | SmoothQuant, FP8, supported W4A8 schemes | Does the actual kernel improve performance at the intended batch size? |
| Run across local CPU and GPU resources | Appropriate GGUF encodings with llama.cpp | Which tensor mix fits memory and provides acceptable latency? |
| Fine-tune under a constrained memory budget | QLoRA and NF4 | Does adapter quality survive the final inference conversion? |
| Reduce memory at long contexts or high concurrency | FP8/integer KV-cache schemes, KIVI-style methods | Are retrieval and generation preserved at deployment context lengths? |
| Explore aggressive weight-and-activation compression | QuaRot, SpinQuant, QAT | Are quality gains worth calibration, training and runtime complexity? |
| Use block-scaled four-bit floating point | Supported MXFP4 or NVFP4 configurations | Does the complete hardware/software path deliver the expected benefit? |

A disciplined evaluation begins with a reproducible higher-precision baseline. Preserve the checkpoint, tokenizer, prompt template, decoding settings and evaluation data. Record quality and runtime measurements before changing precision.

Next, identify the dominant constraint. If weights prevent the model from fitting, weight compression is the immediate priority. If long contexts exhaust memory, calculate and profile the cache. If latency remains poor after the model fits, inspect execution rather than assuming that another reduction in bit width will help.

Use calibration data that covers the intended workload, including relevant languages, code, document structure and sequence lengths. Keep calibration separate from evaluation. Calibration that represents typical traffic should also be supplemented with evaluation of rare but consequential inputs.

Quality assessment should extend beyond perplexity. Useful measurements include task accuracy, executable code correctness, numerical reasoning, instruction adherence, structured-output validity and long-context retrieval. For reasoning workloads, hold generation budgets constant and inspect changes in response length and termination as well as final-answer accuracy.

Report uncertainty where sampling is involved. Small apparent gains or losses may reflect evaluation variance. Avoid tables that assign a fixed “accuracy drop” to a method across unrelated models and tasks, and state whether a reported change is relative percent or percentage points.

Finally, select the configuration on measured quality, peak memory and runtime together. A larger quantised model may outperform a smaller higher-precision model, but the comparison is meaningful only under a clearly stated resource budget. Model size, architecture, training quality and workload all affect the result.

Quantisation ultimately allocates limited numerical precision across a computation. Rounding chooses nearby levels; GPTQ compensates for output error; AWQ and SmoothQuant redistribute sensitivity; rotations change the coordinate system; codebooks encode local structure; and QAT adapts trainable parameters to the eventual representation. A successful deployment combines an appropriate error-control method with a format and kernel that suit the hardware, then verifies that the resulting model still performs the work required of it.

## References

1. PyTorch. [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/).
2. Shao et al. [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137).
3. Frantar et al. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323).
4. PyTorch. [Quantization-Aware Training for Large Language Models with PyTorch](https://pytorch.org/blog/quantization-aware-training/).
5. IST-DASLab. [GPTQ reference implementation](https://github.com/IST-DASLab/gptq/blob/main/gptq.py).
6. Lin et al. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978).
7. Dettmers et al. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339).
8. Xiao et al. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).
9. Micikevicius et al. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433).
10. Dettmers et al. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
11. NVIDIA. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/?p=102000).
12. ggml. [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
13. ggml-org. [llama.cpp](https://github.com/ggml-org/llama.cpp), [tensor encoding schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes), and [block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).
14. Ashkboos et al. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456), and [reference implementation](https://github.com/spcl/QuaRot).
15. Chee et al. [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304).
16. Tseng et al. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396).
17. Liu et al. [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406).
18. Liu et al. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750).
19. Lin et al. [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532).
20. ggml-org. [llama.cpp quantisation tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), [importance-matrix tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), and [importance-matrix implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp).
21. Ma et al. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764).
22. IST-DASLab. [Marlin: FP16 × INT4 inference kernel](https://github.com/IST-DASLab/marlin).

*Numerical memory examples are illustrative calculations, not benchmark measurements. Linked software documentation can evolve independently of the research papers.*
