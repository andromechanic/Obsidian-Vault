---
tags: 
Date: "16-06-2025 10:30"
---

## LoRA

---
## LoRA (Low-Rank Adaptation)

Concept: LoRA is a Parameter-Efficient Fine-Tuning (PEFT) technique that significantly reduces the number of trainable parameters during fine-tuning of large models like LLMs and Stable Diffusion models. Instead of fine-tuning all the weights of the pre-trained model, LoRA injects a small number of trainable rank decomposition matrices into the transformer layers.

How it Works:

For a pre-trained weight matrix W0​ (e.g., a query, key, value, or output projection matrix in a Transformer's attention mechanism), LoRA introduces two small, trainable matrices, A and B, such that their product BA approximates the weight update ΔW.

So, the effective weight matrix becomes W′=W0​+BA.

- W0​ is the original, frozen pre-trained weight matrix.
    
- A is a low-rank matrix of shape (d,r), where d is the input dimension and r is the chosen rank (r≪min(din​,dout​)).
    
- B is a low-rank matrix of shape (r,dout​), where dout​ is the output dimension.
    

Only the matrices A and B are trained during fine-tuning. The rank r is a crucial hyperparameter; a smaller r means fewer trainable parameters but might limit expressiveness.

Benefits:

- Massive Reduction in Trainable Parameters: This is the primary advantage, making fine-tuning much more memory and computationally efficient.
    
- Faster Training: Fewer parameters mean faster gradient computations and updates.
    
- Reduced Storage for Fine-tuned Models: You only need to store the small A and B matrices for each fine-tuned task, rather than a full copy of the large base model.
    
- No Inference Latency: During inference, the learned ΔW=BA can be merged with W0​ offline, so there's no additional computational overhead at inference time.
    


