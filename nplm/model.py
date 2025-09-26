# model.py

# implemented with help from chatgpt and this medium article:
# https://medium.com/@dahami/a-neural-probabilistic-language-model-breaking-down-bengios-approach-4bf793a84426
# and this other implementation article: https://naturale0.github.io/2021/02/04/Understanding-Neural-Probabilistic-Language-Model

import torch
import torch.nn as nn
import torch.nn.functional as F


class NPLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim):
        super().__init__()

        # Embedding lookup matrix (C in the paper)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        input_dim = context_size * embed_dim  # (n-1) * m

        # Hidden transformation (Hx + d)
        self.hidden = nn.Linear(input_dim, hidden_dim)

        # Output transformations: Wx and Uh
        self.W = nn.Linear(input_dim, vocab_size, bias=False)
        self.U = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Single shared bias (b)
        self.b = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        # Lookup embeddings: (batch, context, embed_dim)
        embeds = self.embeddings(x)

        # Flatten: (batch, context*embed_dim)
        x_flat = embeds.view(embeds.size(0), -1)

        # Hidden path: Uh
        h = torch.tanh(self.hidden(x_flat))
        hidden_out = self.U(h)

        # Linear shortcut path: Wx
        linear_out = self.W(x_flat)

        # Combine with single bias
        y = hidden_out + linear_out + self.b

        return y