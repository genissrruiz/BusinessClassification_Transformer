# In this file, the training process is implemented in the train_epoch function. 
# The train_epoch function takes the model, optimizer, data_loader, loss_history, and device as input. 

# Importing libraries.
import torch
import torch.nn.functional as F

# Train the model in order to classify images with text annotations.
def train_epoch(model, optimizer, data_loader, loss_history,device="cuda"):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data_img, data_txt, txt_mask, target) in enumerate(data_loader):
        data_img = data_img.to(device)
        data_txt = data_txt.to(device)
        txt_mask = txt_mask.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data_img, data_txt, txt_mask), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data_img)) + '/' + '{:5}'.format(total_samples) +
                 ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
