---
tags:
  - finetuning
Date: 16-06-2025 10:31
---

## QLoRA

---

## QLoRA (Quantized Low-Rank Adaptation)

Concept: [[QLoRA]] is an extension of [[LoRA]] also called as LoRA 2.0, that further optimizes memory usage by combining it with quantization techniques. Specifically, QLoRA quantizes the base LLM's weights to a very low precision (e.g., 4-bit) while still performing LoRA fine-tuning on top.

How it Works:
 
QLoRA uses 4-bit NormalFloat (NF4) quantization, which is a data type optimized for normally distributed data, providing better precision than standard 4-bit integers.

The key innovation is "double quantization," where the quantization constants themselves are quantized, saving even more memory.

During fine-tuning, the 4-bit quantized base model is loaded into memory. However, to perform gradient calculations, the 4-bit weights are de-quantized to a higher precision (e.g., BF16) only when needed for specific computations. The LoRA adaptation matrices (A and B) are trained in a higher precision (e.g., FP16 or BF16), while the gradients are computed with respect to these high-precision LoRA weights.

Benefits:

- Extremely Low Memory Footprint: Allows fine-tuning of very large LLMs (e.g., 65B parameters) on a single consumer GPU (e.g., RTX 3090/4090).
    
- Maintains Performance: Despite the aggressive quantization, QLoRA often achieves comparable performance to full 16-bit LoRA fine-tuning.
    
- Democratizes Fine-tuning: Makes it possible for researchers and developers with limited hardware to fine-tune state-of-the-art LLMs.
    

