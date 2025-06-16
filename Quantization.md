---
tags: 
Date: "16-06-2025 10:24"
---

## Quantization

---
#### 2.1 Model Quantization

Concept: This specifically refers to the application of quantization techniques to the weights, biases, and sometimes activations within the neural network structure of an LLM. The goal is to reduce the precision of these numerical representations, making the model lighter and faster.
 
#### 2.2 Full Precision (FP32) and Half Precision (FP16/BF16)

Concept: These refer to the data types used to represent numbers in computer memory and during computations.

- Full Precision (FP32 - 32-bit Floating Point): This is the standard data type for most deep learning training. It uses 32 bits to represent a floating-point number, offering a wide range and high precision. While precise, it's memory-intensive and computationally demanding.
    

- Representation: 1 sign bit, 8 exponent bits, 23 mantissa bits.
    

- Half Precision (FP16 - 16-bit Floating Point): This uses 16 bits to represent a floating-point number. It significantly reduces memory usage and can speed up computations on hardware that supports it (e.g., modern GPUs with Tensor Cores). However, it has a smaller range and less precision, which can sometimes lead to numerical instability during training if not managed carefully.
    

- Representation: 1 sign bit, 5 exponent bits, 10 mantissa bits.
    

- BFloat16 (BF16 - Brain Floating Point): Another 16-bit floating-point format, designed specifically for deep learning by Google. Unlike FP16, BF16 retains the 8-bit exponent of FP32 while reducing the mantissa to 7 bits. This gives it the same dynamic range as FP32, which is crucial for training stability, while still offering the memory and speed benefits of a 16-bit format.
    

- Representation: 1 sign bit, 8 exponent bits, 7 mantissa bits.
    

Relevance to LLMs: Training and inference with LLMs often utilize mixed precision (e.g., training in BF16/FP16 for speed while keeping master weights in FP32) to balance performance and memory constraints.

#### 2.3 Calibration

Concept: Calibration is a crucial step in Post-Training Quantization (PTQ). It involves running a small representative subset of the original dataset through the full precision model before quantization. During this process, the ranges (min/max values) of activations for each layer are observed and recorded. These observed ranges are then used to determine the optimal scaling factors and zero-points for quantizing the activations of each layer, minimizing information loss.

Why Calibrate?

Without proper calibration, naive quantization can lead to significant accuracy drops because the fixed quantization ranges might not accurately represent the distribution of actual activation values during inference.

#### 2.4 Modes of Quantization

- Post-Training Quantization (PTQ):  
    Concept: This method quantizes an LLM after it has been fully trained in full precision. It's the simplest and most common form of quantization because it doesn't require re-training or access to the training dataset (only a calibration dataset, if applicable, for activations).  
    Pros: Easy to implement, no training required, compatible with pre-trained models.  
    Cons: Can lead to a noticeable drop in accuracy, especially for lower bit-widths (e.g., 4-bit), as the model was not "aware" of quantization during its training.
    
- Quantization-Aware Training (QAT):  
    Concept: This method simulates the effects of quantization during the training process. Quantization and de-quantization operations are inserted into the model's computational graph. The model "learns" to be robust to the precision limitations during training, leading to significantly higher accuracy compared to PTQ at the same bit-width.  
    Pros: Achieves much higher accuracy for a given bit-width compared to PTQ, as the model learns to compensate for quantization errors.  
    Cons: Requires access to the original training data and the ability to modify the training pipeline. More complex to implement.
    

#### 2.5 Symmetric Quantization

Concept: In symmetric quantization, the range of values is symmetrical around zero. This means the minimum and maximum values for a given tensor (e.g., weights or activations) have the same absolute magnitude (e.g., -127 to +127 for an 8-bit signed integer). The zero-point is always set to zero.

Formula (simplified for signed integers):

q=round(x/S)

x=$q∗S$

where $S = \frac{{maxabsvalue}}{{quantrange}/2}$ (e.g., 127 for 8-bit signed)

Example: If your float range is [−1.0,1.0] and you quantize to 8-bit signed integers ([−127,127]), the scaling factor S would be calculated such that 1.0 maps to 127 and −1.0 maps to −127.

Use Cases: Often used for weights and activations where the distribution is centered around zero.

#### 2.6 Asymmetric Quantization

Concept: In asymmetric quantization, the range of values is not necessarily symmetrical around zero. A non-zero offset (zero-point) is used to map the floating-point zero to a specific integer value. This allows the quantization range to more closely match the actual distribution of the data, especially when the data is not centered around zero.

Formula (simplified for unsigned integers):

$q=round(x/S)+Z$

$x=(q−Z)∗S$

where $S = \frac{{maxvalue} - {minvalue}}{{quantrange}}$ and $Z = {round}(-{minvalue} / S)$

Example: If your float range is [0.0,2.0] and you quantize to 8-bit unsigned integers ([0,255]), the zero-point Z would be 0, and the scaling factor S would map 0.0 to 0 and 2.0 to 255. If your range was [−0.5,1.5], the zero-point would shift to align 0.0 with an integer in the 0-255 range.

Use Cases: Particularly effective for activations, which often have asymmetric distributions (e.g., ReLU outputs are always non-negative).

