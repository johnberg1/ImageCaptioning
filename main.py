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

# load data
root = "C:/pr_files/neural/"
#totalList = sorted(os.listdir(root+"images/"), key=len)
#lens = [int(len(totalList)*0.7), int(len(totalList)*0.1), len(totalList) - (int(len(totalList)*0.1) + int(len(totalList)*0.7))]
#train, test, val = torch.utils.data.dataset.random_split(totalList, lens)
tr = 0.7
val = 0.1
dataset_tr   = ImageDataset(root_dir = "C:/pr_files/", trainPercent = tr, valPercent = val , flag = 'train', transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((224,224),interpolation=Image.NEAREST),
                                                                                           transforms.ToTensor()]))
dataset_test = ImageDataset(root_dir = "C:/pr_files/", trainPercent = tr, valPercent = val , flag = 'test',transform = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST), transforms.ToTensor()]))
dataset_val  = ImageDataset(root_dir = "C:/pr_files/", trainPercent = tr, valPercent = val  , flag = 'validation',transform = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                                                                         transforms.ToTensor(), transforms.Normalize(mean = [0.489, 0.456, 0.406], std = [0.229, 0.224,0.225])]))

dataloader_train = DataLoader(dataset_tr, batch_size=64, shuffle=True)
dataloader_train2 = DataLoader(dataset_tr, batch_size=32, shuffle=True)
dataloader_test  = DataLoader(dataset_test, batch_size=1, shuffle=False)
dataloader_val   = DataLoader(dataset_val, batch_size=50, shuffle=True)

# Initializations for training
losses = list()
val_losses = list()
decoder = DecoderRNN(embed_size=300, hidden_size=256, vocab_size=1004).to(device)
# encoder = EncoderCNN(300).to(device)
# encoder.fine_tune()
criterion = nn.CrossEntropyLoss().to(device)

vocab_size = 1004
# encoder_lr = 0.008
# encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             # lr=encoder_lr)
decoder_lr = 0.008
decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

wordC = pd.read_hdf(root + "eee443_project_dataset_train.h5", 'word_code')
wordC = wordC.to_dict('split')
wordDict = dict(zip(wordC['data'][0], wordC['columns']))
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
        features = images
        captions = data["caption"]
        # make the captions for targets and teacher forcer
        captions_target = captions[:, 1:].long().to(device)
        captions_train = captions[:, :captions.shape[1]-1].long().to(device)

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)

        # Pass the inputs through the CNN-RNN model.
        # features = encoder(images)
        outputs = decoder(features, captions_train)

        # Calculate the batch loss
        trainLoss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))

        # Backward pass
        trainLoss.backward()

        # Update the parameters in the optimizer
        # encoder_optimizer.step()
        decoder_optimizer.step()

       

        # - - - Validate - - -
        # turn the evaluation mode on
        if i % 50 == 0:
            with torch.no_grad():

                # set the evaluation mode
                # encoder.eval()
                decoder.eval()

                # get the validation images and captions
                dataVal = next(iter(dataloader_val))
                val_images = dataVal["image"]
                img = val_images[0]
                val_captions = dataVal["caption"]
                cap = val_captions[0]
                
                caption = cap.cpu().detach().numpy()
                captionStr = [wordDict[i] for i in caption]
                print(captionStr)
                # define the captions
                captions_target = val_captions[:, 1:].long().to(device)
                captions_train = val_captions[:, :-1].long().to(device)

                # Move batch of images and captions to GPU if CUDA is available.
                val_images = val_images.to(device)
                val_features = val_images
                # Pass the inputs through the CNN-RNN model.
                # val_features = encoder(val_images)
                val_outputs = decoder(val_features, captions_train)
                
                captionOut = val_outputs[0].cpu().detach().numpy().argmax(axis = 1)
                captionOutStr = [wordDict[i] for i in captionOut]
                print("\n", captionOutStr)
                # Calculate the batch loss.
                val_loss = criterion(val_outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
                print("Loss of ", i, "th batch: ", val_loss.item(), sep="")
                                
                val_losses.append(val_loss.item())
                
                
                losses.append(trainLoss.item())
                np.save("train_losses", np.array(val_losses))    
                np.save("val_losses", np.array(val_losses))
        # append the validation loss and training loss
    
        
        
#        losses.append(loss.item())
#
#        # save the losses
#        np.save("losses", np.array(losses))
#        np.save("val_losses", np.array(val_losses))
#
#        # Get training statistics.
#        stats = "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f" % (epoch, num_epochs, i_step, total_step, loss.item(), val_loss.item())
#
#        # Print training statistics (on same line).
#        print("\r" + stats, end="")
#        sys.stdout.flush()
#
#    # Save the weights. if epoch % save_every == 0:
        # print("\nSaving the model")
        # torch.save(decoder.state_dict(), os.path.join("./models", "decoder-%d.pth" % epoch))
        # torch.save(encoder.state_dict(), os.path.join("./models", "encoder-%d.pth" % epoch))
