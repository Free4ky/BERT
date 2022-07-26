from torch import nn
from transformers import BertModel
import torch
from torch.optim import Adam
from tqdm import tqdm
from dataclass import Dataset
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import numpy as np
import random
import pandas as pd
from copy import deepcopy


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

    def train(self, model, train_data, val_data, learning_rate, epochs):

        torch.cuda.empty_cache()

        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.0

        self.valLoss = []
        self.trainLoss = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train, val = Dataset(train_data), Dataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(
            train, batch_size=4, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.type(torch.LongTensor)
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.type(torch.LongTensor)
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            if (total_acc_val > best_acc):
                best_acc = total_acc_val
                best_model_wts = deepcopy(model.state_dict())

            self.valLoss.append(total_loss_val / len(val_data))
            self.trainLoss.append(total_loss_train / len(train_data))

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                    | Val Loss: {total_loss_val / len(val_data): .3f} \
                    | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        model.load_state_dict(best_model_wts)

    def showTrainingResult(self):
        fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
        epochs = range(1, len(self.valLoss) + 1)

        plt.plot(epochs, self.valLoss, color='red', label='Validation loss')
        plt.plot(epochs, self.trainLoss, color='blue', label='Train loss')
        plt.title('Training result')
        plt.legend()
        plt.show()


def evaluate(model, test_data):

    # model.eval()
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.type(torch.LongTensor)
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def checkInput(model):
    print('Чтобы завершить введите 0')
    answers = ["Hmm, let me see... It is: ",
               "Oh, i'm pretty sure that it is: ",
               "At least i can try... It is: "]
    labels = {0: 'educational',
              1: 'other'
              }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.eval()
    model.to(device)

    while True:
        with torch.no_grad():
            userInput = input("Введите тестовую строку: ")
            df = pd
            if userInput == '0':
                return
            d = {'category': ['educational'], 'text': [userInput]}
            df = pd.DataFrame(data=d)
            test = Dataset(df)
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
            for test_input, test_label in test_dataloader:
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                result = output.argmax(dim=1)
                print(
                    f'{userInput}\n{random.choice(answers)}{labels[result.item()]}')
