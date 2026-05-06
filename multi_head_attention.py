import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedd_size, heads):
        super(MultiHeadAttention, self).__init__()
        
        self.heads = heads
        self.head_dims = embedd_size // heads
        
        self.W_q = nn.Linear(embedd_size, embedd_size)
        self.W_k = nn.Linear(embedd_size, embedd_size)
        self.W_v = nn.Linear(embedd_size, embedd_size)
        
        self.fc_out = nn.Linear(embedd_size, embedd_size)
        
    def forward(self, X, mask=None):
        batch_size, seq_length, embedd_size = X.shape
        
        # Compute Q, K, V matrices
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
        
        # split into multiple heads
        Q = Q.view(batch_size, seq_length, self.heads, self.head_dims).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.heads, self.head_dims).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.heads, self.head_dims).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dims ** 0.5)
        print(f'Attention Scores before masking: \n{attention_scores}')
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        print(f'Attention Scores after masking: {attention_scores}')
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embedd_size)
        out = self.fc_out(out)
        
        return out
    
    
    
seq_length = 3    
X = torch.rand((1, 3, 8))  # (batch_size, seq_length, embedd_size)
mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0)  # Example mask for sequence length of 3  

attention_layer = MultiHeadAttention(embedd_size=8, heads=2)

output = attention_layer(X, mask=mask)

print(output)      
    
    
