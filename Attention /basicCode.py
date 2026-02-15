import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        
        self.query_layer = nn.Linear(embed_dim, embed_dim)

        #initialising the key layer
        self.key_layer = nn.Linear(embed_dim, embed_dim)

        #initialising the value layer
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        #scaling factor for the attention scores
        self.scale_layer = embed_dim ** 0.5

    def forward(self, x):
        #x is the input to the attention layer, which is the output of the encoder
        Q = self.query_layer(x)
        #k is the output of the key layer
        K = self.key_layer(x)
        #v is the output of the value layer
        V = self.value_layer(x)

        #calculating the dot product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_layer
        #applying softmax to get the attention weights and converting the weights into probabilities
        attention_weights = torch.softmax(scores, dim=-1)

        #calculating the weighted sum of the value vectors
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
    

