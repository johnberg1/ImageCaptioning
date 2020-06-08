import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=300):
        super(EncoderCNN, self).__init__()

        # get the pretrained densenet model
        
        resnet = torchvision.models.resnet152(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        # self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # add another fully connected layer
        # self.fulCon = nn.Linear(in_features=1024, out_features=embed_size)

        # # dropout layer
        # self.dropout = nn.Dropout(p=0.5)

        # # activation layers
        # self.prelu = nn.PReLU()

    def forward(self, images):

        # # get the embeddings from the densenet
        # densenet_outputs = self.dropout(self.prelu(self.densenet(images)))

        # # pass through the fully connected
        # embeddings = self.fulCon(densenet_outputs)

        # return embeddings
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
#     def fine_tune(self):
# #        for p in self.densenet.parameters():
# #            p.requires_grad = False
#         for c in list(self.densenet.children())[0][:6]:
#             for p in c.parameters():
#                 p.requires_grad = False

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        pretrainedEmbeds = np.loadtxt('embeds300.txt', delimiter=',')
        self.embed.weight.data.copy_(torch.from_numpy(pretrainedEmbeds))
        self.embed.weight.requires_grad = False
        
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):
        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)

        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).to(device)

        # embed the captions
        captions_embed = self.embed(captions)

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features.to(device), (hidden_state, cell_state))

            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.fc_out(hidden_state)

            # build the output tensor
            outputs[:, t, :] = out

        return outputs