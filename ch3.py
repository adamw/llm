import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) # nn.Linear(d_in, d_out, bias=False).weight.T 
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # print("x", x.shape) # tokens x d_in
        # print("W_key", self.W_key.shape) # d_in x d_out
        keys = x @ self.W_key 
        # print("keys", keys.shape) # tokens x d_out (token -> key)
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T
        # print("queries", queries.shape) # tokens x d_out
        # print("keys.T", keys.T.shape) # d_out x tokens
        # print("attn_scores", attn_scores.shape) # tokens x tokens
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # print("attn_weights", attn_weights.shape) # tokens x tokens
        # print("values", values.shape) # tokens x d_out
        context_vec = attn_weights @ values 
        # print("context_vec", context_vec.shape) # tokens x d_out
        return context_vec
    
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        context_vec = attn_weights @ values
        return context_vec

d_in = 3
d_out = 2

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

# torch.manual_seed(123)
# sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        _, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
                
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec
    
# batch = torch.stack((inputs, inputs), dim=0)   
# print(batch.shape)

# torch.manual_seed(123)
# context_length = batch.shape[1]
# ca = CausalAttention(d_in, d_out, context_length, 0.5)
# context_vecs = ca(batch)
# print("context_vecs.shape", context_vecs.shape)
# print(context_vecs)

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)    

# batch = torch.stack((inputs, inputs), dim=0)   

# torch.manual_seed(123)
# context_length = batch.shape[1]
# d_in, d_out = 3, 1 # exercise 3.2 -> d_out from 2 to 1 (output is 2-dim since we have two heads)
# mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
# context_vecs = mha(batch)

# print(context_vecs)
# print("context_vecs.shape", context_vecs.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x) # b x num_tokens x d_out
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # unroll: b x num_tokens x d_out -> b x num_tokens x num_heads x head_dim
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # b x num_tokens x num_heads x head_dim -> b x num_heads x num_tokens x head_dim
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
                
        # queries: b x num_heads x num_tokens x head_dim
        # keys.transpose(2,3): b x num_heads x head_dim x num_tokens
        # attn_scores: b x num_heads x num_tokens x num_tokens
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # attn_weights: b x num_heads x num_tokens x num_tokens
        # values: b x num_heads x num_tokens x head_dim
        # attn_weights @ values: b x num_heads x num_tokens x head_dim
        # (attn_weights @ values).transpose(1, 2): b x num_tokens x num_heads x head_dim
        context_vec = (attn_weights @ values).transpose(1, 2) # b x num_tokens x num_heads x head_dim
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    

# batch = torch.stack((inputs, inputs), dim=0)   

# torch.manual_seed(123)
# batch_size, context_length, d_in = batch.shape
# d_out = 2
# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
# context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape", context_vecs.shape)