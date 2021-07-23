import torch 
from typing import Tuple

torch.manual_seed(0)

print("## PART 1 ##")
## Ex 1.1
def sgd_factorise(A: torch.Tensor, rank : int, num_epochs =1000, lr =0.01):
    m = A.shape[0]
    n = A.shape[1]
    U = torch.rand(m, rank)
    V = torch.rand(n, rank)
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A[r][c] - U[r] @ V[c].t()
                U[r] = U[r] + lr * e * V[c] 
                V[c] = V[c] + lr * e * U[r]
    return [U, V]

## Ex 1.2
A = torch.tensor([[0.3374, 0.6005, 0.1735],[3.3359, 0.0492, 1.8374],[2.9407, 0.5301, 2.2620]])
print("A: ",A)
[U, V] = sgd_factorise(A, 2)
print("U: ", U)
print("V: ", V)
R = U@V.t()
print("R: ", R)
loss = torch.nn.functional.mse_loss(R, A, reduction="sum")
print("Loss: ", loss)

print()
print("## PART 2 ##")

## Ex 2.1
(U, S, V) = torch.svd(A)
print("S: ", S)
S[-1] = 0
print("S: ", S)
SVD = U @ torch.diag(S) @ V.t()
print("R: ", SVD)
loss = torch.nn.functional.mse_loss(SVD, A, reduction="sum")
print("Loss: ", loss)

print()
print("## PART 3 ##")

## Ex 3.1
def sgd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank : int, num_epochs =1000, lr =0.01):
    m = A.shape[0]
    n = A.shape[1]
    U = torch.rand(m, rank)
    V = torch.rand(n, rank)
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                if(M[r][c]):
                    e = A[r][c] - U[r] @ V[c].t()
                    U[r] = U[r] + lr * e * V[c] 
                    V[c] = V[c] + lr * e * U[r]
    return [U, V]

## Ex 3.2
M = torch.tensor([[1, 1, 1],[0, 1, 1],[1, 0, 1]])
[U, V] = sgd_factorise_masked(A, M, 2)
print("U: ", U)
print("V: ", V)
R2 = U@V.t()
print("A: ", A)
print("R: ", R2)
loss = torch.nn.functional.mse_loss(R2, A, reduction="sum")
print("Loss: ", loss)

R2[1][0] = A[1][0]
R2[2][1] = A[2][1]
print(R2)
loss = torch.nn.functional.mse_loss(R2, A, reduction="sum")
print("Loss: ", loss)