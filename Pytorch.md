---
tags:
  - "#workshops/pytorch"
Date: 15-01-2025 14:19
---

## Pytorch

---

1. Importing Pytorch

```python
import torch
```

2. Finding GPUs

```python
print(torch.cuda.is_available())    #prints True if GPU
torch.cuda.get_device_name(0)       #prints name of GPU
```

3. Tensor

```python
a = torch.tensor([1.0, 2.0, 3],dtype=torch.float32)
a.ndim
```
4. Accessing elements using positions.
`tensor_vaiable [row][column]`


```python
c = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(c[1][0])    #prints 4
c[:, 1]           #prints tensor([2., 5.])
c[0,: ]           #prints tensor([1., 2., 3.])
c.shape           #prints torch.Size([2, 3])
```


```python
# empty
emp = torch.empty(2,3)
#zeros
torch.zeros(4,5)
#ones
torch.ones(3,4)
```

5. Manual seeding while using random - It keeps the random initialization the same how much ever times the cell is run
```python
#manual_seed
torch.manual_seed(42)
#rand
torch.rand(2,3)
```


```python
#arange
torch.arange(5, 15, 3)   #prints tensor([ 5,  8, 11, 14])
#linspace
torch.linspace(0, 10, 5)   #prints tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
#eye (identity matrix)
torch.eye(3)
#diagonal
torch.diag(torch.tensor([1,2,3,5]))
#full
torch.full((4, 5), 7)
```

6. Reshape can change the shapes of tensors and flatten will make it into a 1D format

```python
arr2 = torch.rand(3,6)
arr2.reshape(3,2,3)
#flatten
arr2.flatten()
t1_empty = torch.empty_like(arr2)   #zero tensor with arr2s dimension
#ones_like
torch.ones_like(t1)
#rand_like
torch.rand_like(t1)
```


7.  Opeartions
```python
a = torch.tensor([1, 2, 3]) 
b = torch.tensor([4, 5, 6]) 
scalar = 2 

# Arithmetic Operations 
# Scalar addition 
result_add_scalar = a + scalar 
print("Scalar Addition:", result_add_scalar) # tensor([3, 4, 5])
# Scalar subtraction 
result_sub_scalar = a - scalar 
print("Scalar Subtraction:", result_sub_scalar) # tensor([-1, 0, 1])
# Scalar multiplication 
result_mul_scalar = a * scalar
print("Scalar Multiplication:", result_mul_scalar) # tensor([2, 4, 6]) 
# Scalar division 
result_div_scalar = a / scalar 
print("Scalar Division:", result_div_scalar) # tensor([0.5, 1.0, 1.5]) 
# Element-wise Operations 
# Element-wise addition 
result_add = a + b 
print("Element-wise Addition:", result_add) # tensor([5, 7, 9]) 
# Element-wise subtraction 
result_sub = a - b 
print("Element-wise Subtraction:", result_sub) # tensor([-3, -3, -3]) 
# Element-wise multiplication 
result_mul = a * b 
print("Element-wise Multiplication:", result_mul) # tensor([4, 10, 18]) 
# Element-wise division 
result_div = a / b 
print("Element-wise Division:", result_div) # tensor([0.25, 0.4, 0.5]) 


# Statistics Operations 
# Mean 
mean_val = a.mean() 
print("Mean:", mean_val) # tensor(2.) 
# Sum 
sum_val = a.sum() 
print("Sum:", sum_val) # tensor(6) 
# Standard deviation s
td_val = a.std() 
print("Standard Deviation:", std_val) # tensor(1.) 
# Min
min_val = a.min() 
print("Min:", min_val) # tensor(1) 
# Max 
max_val = a.max() 
print("Max:", max_val) # tensor(3) 


# Matrix Operations
# Creating matrices 
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]]) 
# Matrix multiplication 
result_matmul = torch.matmul(matrix_a, matrix_b) 
print("Matrix Multiplication:\n", result_matmul) # tensor([[19, 22], # [43, 50]])
# Transpose 
result_transpose = matrix_a.T 
print("Transpose:\n", result_transpose) # tensor([[1, 3], # [2, 4]]) 
# Inverse 
matrix_inv = torch.inverse(matrix_a.float()) 
print("Inverse:\n", matrix_inv) # tensor([[-2.0, 1.0], # [ 1.5, -0.5]])
# Determinant 
det_val = torch.det(matrix_a.float()) 
print("Determinant:", det_val) # tensor(-2.0000) 
# Eigenvalues and Eigenvectors 
eigenvalues, eigenvectors = torch.eig(matrix_a.float(), eigenvectors=True) 
print("Eigenvalues:\n", eigenvalues) 
print("Eigenvectors:\n", eigenvectors)
```

8. Important functions
```python
#sigmoid
torch.sigmoid(arr)
#softmax
torch.softmax(arr,dim=1)
#relu
torch.relu(arr)

```

9. Setup GPU in Pytorch
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.tensor([1,2,3], device = device)
```

10. [[Autograd]]
```python
# required for the gradient tracking
Z = torch.tensor(2.0, requires_grad=True)
```


```python
```


```python
```