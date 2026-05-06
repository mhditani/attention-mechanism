import torch
import torch.nn.functional as F

Q = torch.rand((3,4))
K = torch.rand((3,4))  
V = torch.rand((3,4))

# Compute the attention scores
attentin_scores = Q @ (K.T)  # (3,4) @ (4,3) -> (3,3)

# normalize the attention scores
attention_weights = F.softmax(attentin_scores, dim=-1)  # (3,3)

# Compute the output
output = attention_weights @ V  # (3,3) @ (3,4)  

# print the Attention Weights
print(f'Attention Weights: {attention_weights}')

# print the output
print(f'Attention Scores: {output}')    


