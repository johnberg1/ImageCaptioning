import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from codec import EncoderCNN, DecoderRNN
from loaderTensor import ImageDataset
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "C:/pr_files/neural/"
tr = 0.7
val = 0.1
dataset_test = ImageDataset(root_dir = root, trainPercent = tr, valPercent = val , 
                            flag = 'test',transform = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST), transforms.ToTensor()]))

vocab_size = 1004
dataloader_test  = DataLoader(dataset_test, batch_size=64, shuffle=False)
wordC = pd.read_hdf(root + "eee443_project_dataset_train.h5", 'word_code')
wordC = wordC.to_dict('split')
wordDict = dict(zip(wordC['data'][0], wordC['columns']))

# encoderDir = 'C:/Users/user/Desktop/EE443 Neural/Final Project/imgNet/models/encoder1578752311.pth'
decoderDir = 'C:/pr_files/models/decoder-2.pth'

inputOutputPair = []
decoder = DecoderRNN(embed_size=300, hidden_size=256, vocab_size=1004).to(device)
encoder = EncoderCNN(300).to(device)
decoder.load_state_dict(torch.load(decoderDir))
# encoder.load_state_dict(torch.load(encoderDir))
criterion = nn.CrossEntropyLoss().to(device)
lossLog = []
transform = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST), transforms.ToTensor()])

oppenheim = Image.open("oppenheim.png")
oppenheim = transform(oppenheim)
oppenheim = oppenheim.view(1,3,224,224)
#test the model  

with torch.no_grad():
    # set the evaluation mode
    encoder.eval()
    decoder.eval()

    val_images = oppenheim
    
    # val_captions = testData["caption"]
    # cap = val_captions[0]          
    # caption = cap.cpu().detach().numpy()
    
    # define the captions
    # captions_target = val_captions[:, 1:].long().to(device)
    # captions_train = val_captions[:, :-1].long().to(device)

    # Move batch of images and captions to GPU if CUDA is available.
    val_images = val_images.to(device)

    # Pass the inputs through the CNN-RNN model.
    captions_train = torch.from_numpy(np.arange(16)).view(1,16).long().to(device)
    val_features = encoder(val_images)
    val_outputs = decoder(val_features, captions_train)
    # if i %100 == 0:
    #     print(i)
    #     captionOut = val_outputs[0].cpu().detach().numpy().argmax(axis = 1)
    #     captionStr = [wordDict[i] for i in caption]
    #     captionOutStr = [wordDict[i] for i in captionOut]
    #     img = val_images[0]
    #     inputOutputPair.append([img, captionStr, captionOutStr])
    
    # val_loss = criterion(val_outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
    # lossLog.append(val_loss)

def show(i):
    plt.imshow(inputOutputPair[i][0].permute(1,2,0).cpu())
    print(" ".join(inputOutputPair[i][2]))