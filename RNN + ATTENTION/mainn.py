import torch
import torch.nn as nn
import torch.nn.functional as F

#coding the gru based rnn which is the encoder
class encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, emb_dim, n_layers=1):
        super(encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers = n_layers,
            batch_first = True
        )

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden
    
#coding the attention based decoder
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)
    

#decoder with attention
class decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, emb_dim, attention, n_layers=1):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(
            emb_dim + hidden_dim,
            hidden_dim,
            num_layers = n_layers,
            batch_first = True
        )

        # Fix: output is concat of (output, context) => hidden_dim * 2
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)

        embedded = self.embedding(input)

        attention_weights = self.attention(hidden[-1], encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)

        # context vector is the weighted sum of the encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context), dim=1))

        return prediction, hidden, attention_weights
        

#full seq2seq model
class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)

            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs
    

#initialising the model and running it
INPUT_DIM = 10
OUTPUT_DIM = 10
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
HIDDEN_DIM = 64

#now we will run the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn = Attention(HIDDEN_DIM)
    enc = encoder(INPUT_DIM, HIDDEN_DIM, ENC_EMB_DIM)
    dec = decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_EMB_DIM, attn)

    model = seq2seq(enc, dec, device).to(device)

    batch_size = 2
    src_len = 5
    trg_len = 6

    src = torch.randint(0, INPUT_DIM, (batch_size, src_len)).to(device)
    trg = torch.randint(0, OUTPUT_DIM, (batch_size, trg_len)).to(device)

    outputs = model(src, trg)
    print("outputs shape:", outputs.shape)



