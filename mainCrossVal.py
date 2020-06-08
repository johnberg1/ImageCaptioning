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
import numpy as np
# load data
root = "C:/pr_files/neural/"
#totalList = sorted(os.listdir(root+"images/"), key=len)
#lens = [int(len(totalList)*0.7), int(len(totalList)*0.1), len(totalList) - (int(len(totalList)*0.1) + int(len(totalList)*0.7))]
#train, test, val = torch.utils.data.dataset.random_split(totalList, lens)
tr = 0.05
val = 0.01
dataset_tr   = ImageDataset(root_dir = "C:/pr_files/", trainPercent = tr, valPercent = val , flag = 'train', transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((224,224),interpolation=Image.NEAREST),
                                                                                           transforms.ToTensor()]))
#dataset_test = ImageDataset(root_dir = root, trainPercent = tr, valPercent = val , flag = 'test',transform = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST), transforms.ToTensor()]))
dataset_val  = ImageDataset(root_dir = "C:/pr_files/", trainPercent = tr, valPercent = val  , flag = 'validation',transform = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                                                                         transforms.ToTensor(), transforms.Normalize(mean = [0.489, 0.456, 0.406], std = [0.229, 0.224,0.225])]))
wordC = pd.read_hdf(root + "eee443_project_dataset_train.h5", 'word_code')
wordC = wordC.to_dict('split')
wordDict = dict(zip(wordC['data'][0], wordC['columns']))
dataloader_val   = DataLoader(dataset_val, batch_size=50, shuffle=True)

valBatchSize = 2**np.arange(2,8)
avrValLoss = []
#cross validate across decoder hidden layer size
for j in valBatchSize:
    dataloader_train = DataLoader(dataset_tr, batch_size=int(j), shuffle=True)
    
    print('Batch Size', j)
    # Initializations for training
    losses = list()
    val_losses = list()
    decoder = DecoderRNN(embed_size=300, hidden_size=256, vocab_size=1004).to(device)
    #encoder = EncoderCNN(300).to(device)
    # encoder.fine_tune()
    criterion = nn.CrossEntropyLoss().to(device)
    
    vocab_size = 1004
    # encoder_lr = 0.008
    # encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 # lr=encoder_lr)
    decoder_lr = 0.005
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                 lr=decoder_lr)
    
    # Training starts
    for epoch in range(1, 3):
        print("Epoch:",epoch)
        for i, data in enumerate(dataloader_train):
            # zero the gradients
            decoder.zero_grad()
            # encoder.zero_grad()
    
            # set decoder and encoder into train mode
            # encoder.train()
            decoder.train()
    
            # Obtain the batch.
            images = data["image"]
            captions = data["caption"]
            # make the captions for targets and teacher forcer
            captions_target = captions[:, 1:].long().to(device)
            captions_train = captions[:, :captions.shape[1]-1].long().to(device)
    
            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
    
            # Pass the inputs through the CNN-RNN model.
            outputs = decoder(images, captions_train)
    
            # Calculate the batch loss
            trainLoss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
    
            # Backward pass
            trainLoss.backward()
    
            # Update the parameters in the optimizer
            # encoder_optimizer.step()
            decoder_optimizer.step()
            if i % 200 == 0:
                print('Batch ', i)
           
    
    #        # - - - Validate - - -
    #        # turn the evaluation mode on
    #        if i % 50 == 0:
    #            with torch.no_grad():
    #
    #                # set the evaluation mode
    #                decoder.eval()
    #
    #                # get the validation images and captions
    #                dataVal = next(iter(dataloader_val))
    #                val_images = dataVal["image"]
    #                img = val_images[0]
    #                val_captions = dataVal["caption"]
    #                cap = val_captions[0]
    #                
    #                caption = cap.cpu().detach().numpy()
    #                captionStr = [wordDict[i] for i in caption]
    #                print(captionStr)
    #                # define the captions
    #                captions_target = val_captions[:, 1:].long().to(device)
    #                captions_train = val_captions[:, :-1].long().to(device)
    #
    #                # Move batch of images and captions to GPU if CUDA is available.
    #                val_images = val_images.to(device)
    #
    #                # Pass the inputs through the CNN-RNN model.
    ##                val_features = encoder(val_images)
    #                val_outputs = decoder(val_images, captions_train)
    #                
    #                captionOut = val_outputs[0].cpu().detach().numpy().argmax(axis = 1)
    #                captionOutStr = [wordDict[i] for i in captionOut]
    #                print("\n", captionOutStr)
    #                # Calculate the batch loss.
    #                val_loss = criterion(val_outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
    #                print("Loss of ", i, "th batch: ", val_loss.item(), sep="")
    #                                
    #                val_losses.append(val_loss.item())
    #                losses.append(trainLoss.item())
    #                np.save("train_lossesWOEncoder", np.array(losses))    
    #                np.save("val_lossesWOEncoder", np.array(val_losses))
        # append the validation loss and training loss
        
            
    print("\nSaving the model")
#    time_stamp = str(int(__import__("time").time()))
    torch.save(decoder.state_dict(), "./models/decoderCrossValBatchSize"+ str(j) + ".pth")
    #    torch.save(encoder.state_dict(), "./models/encoderWOEncoder"+ time_stamp + ".pth")
    
    for i, valData in enumerate(dataloader_val):
        # - - - Validate - - -
        # turn the evaluation mode on
        with torch.no_grad():
        
            # set the evaluation mode
            decoder.eval()
        
            # get the validation images and captions
            dataVal = valData
            val_images = dataVal["image"]
            img = val_images[0]
            val_captions = dataVal["caption"]
            cap = val_captions[0]
            
            caption = cap.cpu().detach().numpy()
#            captionStr = [wordDict[i] for i in caption]
#            print(captionStr)
            # define the captions
            captions_target = val_captions[:, 1:].long().to(device)
            captions_train = val_captions[:, :-1].long().to(device)
        
            # Move batch of images and captions to GPU if CUDA is available.
            val_images = val_images.to(device)
        
            # Pass the inputs through the CNN-RNN model.
        #               val_features = encoder(val_images)
            val_outputs = decoder(val_images, captions_train)
            
            captionOut = val_outputs[0].cpu().detach().numpy().argmax(axis = 1)
            captionOutStr = [wordDict[i] for i in captionOut]
            print("\n", captionOutStr)
            # Calculate the batch loss.
            val_loss = criterion(val_outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
            print("Loss of ", i, "th batch: ", val_loss.item(), sep="")
                            
            val_losses.append(val_loss.item())
            #losses.append(trainLoss.item())
    avrValLoss.append( sum(val_losses)/len(val_losses))

plt.plot(valBatchSize,avrValLoss)
plt.xlabel('Batch Size')
plt.ylabel('Cross Entropy Loss')
plt.title('Validation set Cross Entropy Loss \n with different Batch Sizes')
#np.save("train_lossesWOEncoder", np.array(losses))    
np.save("crossVal_batchSize", np.array(val_losses))

