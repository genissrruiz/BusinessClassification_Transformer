# In this file we define the ConTextTransformer model and the ConTextDataset class. We also define the training and testing loops.

# Importing libraries.
import time,os,json
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import *
from test import *
from utils.utils import *

import wandb

import torchvision
from torch.utils.data import Dataset

from einops import rearrange

import fasttext
import fasttext.util

# Check if CUDA is available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)


# Define the ConTextTransformer model in order to classify images with text annotations.
class ConTextTransformer(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()

        # ResNet152 pretrained model.
        resnet152 = torchvision.models.resnet152(pretrained=True)
        modules=list(resnet152.children())[:-2]
        self.resnet152=nn.Sequential(*modules)

        # Freezing the ResNet152 layers.
        for param in self.resnet152.parameters():
            param.requires_grad = False
        self.num_cnn_features = 64  # 8x8
        self.dim_cnn_features = 2048
        self.dim_fasttext_features = 300

        # Unfreezing the last layers of the ResNet152.
        for param in self.resnet152[-1][-1].parameters():
            param.requires_grad = True
        for param in self.resnet152[-1][-2].parameters():
            param.requires_grad = True
        for param in self.resnet152[-1][-3].parameters():
            param.requires_grad = True

        # Positional embedding, CNN feature to embedding and FastText feature to embedding layers.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_cnn_features + 1, dim))
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim)
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True, activation = 'gelu')
        encoder_norm = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        # Using a MLP head to classify the images with GELU activation function.
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    # Forward pass of the model with image, text and mask inputs (if you want to use it).
    def forward(self, img, txt, mask=None):
        x = self.resnet152(img)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.cnn_feature_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x2 = self.fasttext_feature_to_embedding(txt.float())
        x = torch.cat((x,x2), dim=1)

        #tmp_mask = torch.zeros((img.shape[0], 1+self.num_cnn_features), dtype=torch.bool)
        #mask = torch.cat((tmp_mask.to(device), mask), dim=1)
        #x = self.transformer(x, src_key_padding_mask=mask)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


# Define the ConTextDataset class in order to load the images and text annotations.
class ConTextDataset(Dataset):
    def __init__(self, json_file, root_dir, root_dir_txt,no_text_dir, train=True, transform=None):

        # Load the JSON file with the image names and labels.
        with open(json_file) as f:
            data = json.load(f)
        self.train = train
        self.root_dir = root_dir
        self.root_dir_txt = root_dir_txt
        self.no_text_dir = no_text_dir
        os.makedirs(self.no_text_dir, exist_ok=True)
        self.transform = transform
        if (self.train):
            self.samples = data['train']
        else:
            self.samples = data['test']

        # Load the FastText model for the text embeddings in English.
        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.fasttext = fasttext.load_model('cc.en.300.bin')
        self.dim_fasttext = self.fasttext.get_dimension()
        self.max_num_words = 64

    # Get the length of the dataset.
    def __len__(self):
        return len(self.samples)

    # Get the image, text, text mask and target label.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.samples[idx][0]+'.jpg')
        image = Image.open(img_name).convert('RGB')
        original_image = image.copy()
        if self.transform:
            image = self.transform(image)

        text = np.zeros((self.max_num_words, self.dim_fasttext))
        text_mask = np.ones((self.max_num_words,), dtype=bool)
        text_name = os.path.join(self.root_dir_txt, self.samples[idx][0]+'.json')
        with open(text_name) as f:
            data = json.load(f)

        words = []
        if 'textAnnotations' in data.keys():
            for i in range(1,len(data['textAnnotations'])):
                word = data['textAnnotations'][i]['description']
                if len(word) > 2: words.append(word)

        words = list(set(words))
        for i,w in enumerate(words):
            if i>=self.max_num_words: break
            text[i,:] = self.fasttext.get_word_vector(w)
            text_mask[i] = False
        
        if len(words) == 0:
            no_text_image_path = os.path.join(self.no_text_dir, f"{self.samples[idx][0]}.jpg")
            original_image.save(no_text_image_path)
        
        target = self.samples[idx][1] - 1

        return image, text, text_mask, target



if __name__ == "__main__":
    
    # Using WandB in order to visualize the training process.
    wandb.login()
    wandb.init(project="xnap-project")

    # Load the dataset and the dataloaders.
    json_file = './splits/split_1.json'
    img_dir = "./data/JPEGImages/"
    txt_dir = "./ocr_labels/"
    no_text_dir = "./no_text_images"

    # Define the data transformations for the training and testing sets.
    input_size = 256
    data_transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data_transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Load the dataset and the dataloaders.
    train_set = ConTextDataset(json_file, img_dir, txt_dir, no_text_dir, True, data_transforms_train)
    test_set  = ConTextDataset(json_file, img_dir, txt_dir, no_text_dir, False, data_transforms_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

    # Define the model, optimizer and scheduler.
    N_EPOCHS = 25
    start_time = time.time()

    model = ConTextTransformer(image_size=input_size, num_classes=28, channels=3, dim=256, depth=2, heads=4, mlp_dim=512)
    model.to(device)
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.AdamW(params_to_update, lr=0.0001)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)

    train_loss_history, test_loss_history = [], []
    best_acc = 0.

    # Training and testing loops.
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_loss = train_epoch(model, optimizer, train_loader, train_loss_history)
        acc = evaluate(model, test_loader, test_loss_history)

        # Log the results to WandB.
        wandb.log({"epoch": epoch, "train_loss": train_loss, "accuracy": acc})

        if acc>best_acc: torch.save(model.state_dict(), 'all_best_split1_resnet152.pth')
        scheduler.step()

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

