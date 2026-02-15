GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
#

# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)
# print(batch)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)

# 4.2

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift
    
# 4.3

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x+0.044715*torch.pow(x, 3))))
    
# import matplotlib.pyplot as plt
# gelu, relu = GELU(), nn.ReLU()

# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8,3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)
# out = ffn(x)
# print(out.shape)

# 4.4

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x
    
# layer_sizes = [3, 3, 3, 3, 3, 1]
# simple_input = torch.tensor([[1., 0., -1.]])
# torch.manual_seed(123)
# model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
# torch.manual_seed(123)
# model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

# def print_gradients(model, x):
#     output = model(x)
#     target = torch.tensor([[0.]])
#     loss = nn.MSELoss()
#     loss = loss(output, target)
#     loss.backward()

#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# print("Without shortcut")
# print_gradients(model_without_shortcut, simple_input)
# print("With shortcut")
# print_gradients(model_with_shortcut, simple_input)            

# 4.5

from ch3 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
# torch.manual_seed(123)
# x = torch.rand(2, 4, 768)
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)

# 4.6

class GPTModel(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.debug = debug
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        if self.debug: print("[DEBUG] in_idx", in_idx.shape)
        tok_embeds = self.tok_emb(in_idx)
        if self.debug: print("[DEBUG] tok_embeds", tok_embeds.shape)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        if self.debug: print("[DEBUG] pos_embeds", pos_embeds.shape)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        if self.debug: print("[DEBUG] dropped", x.shape)
        x = self.trf_blocks(x)
        if self.debug: print("[DEBUG] after transformers", x.shape)
        x = self.final_norm(x)
        if self.debug: print("[DEBUG] after norm", x.shape)
        logits = self.out_head(x)
        if self.debug: print("[DEBUG] logits", x.shape)
        return logits
    
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M, debug=True)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# Exercise 4.2
GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_LARGE = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_XL = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def model_stats(cfg, prefix):
    model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{prefix} Total number of parameters: {total_params:,}")

    # Exercise 4.1
    total_params_att = sum(p.numel() for trf in model.trf_blocks for p in trf.att.parameters())
    print(f"{prefix} Total number attention parameters: {total_params_att:,}")

    total_params_ff = sum(p.numel() for trf in model.trf_blocks for p in trf.ff.parameters())
    print(f"{prefix} Total number feed forward parameters: {total_params_ff:,}")

    total_size_bytes = total_params*4 # float32 - 4 bytes
    total_size_mb = total_size_bytes / (1024*1024)
    print(f"{prefix} Total size of the model: {total_size_mb:.2f} MB")
    print("")

# model_stats(GPT_CONFIG_124M, "124M  ") 
# model_stats(GPT_CONFIG_MEDIUM, "MEDIUM") # 406M 
# model_stats(GPT_CONFIG_LARGE, "LARGE ") # 838M
# model_stats(GPT_CONFIG_XL, "XL    ") # 1637M 

# 4.7

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print("encoded_tensor.shape:", encoded_tensor.shape)

# model.eval()
# out = generate_text_simple(model, encoded_tensor, 6, GPT_CONFIG_124M["context_length"])
# print("Output:", out)
# print("Output length:", len(out[0]))

# decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)