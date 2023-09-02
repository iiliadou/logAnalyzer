import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, learning_rate, epochs):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)

        if torch.cuda.is_available():
            self.model = model.cuda()
            self.criterion = self.criterion.cuda()

    def train(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=2)

        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].to(self.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

                batch_loss = self.criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(self.device)
                    mask = val_input['attention_mask'].to(self.device)
                    input_id = val_input['input_ids'].squeeze(1).to(self.device)

                    output = self.model(input_id, mask)

                    batch_loss = self.criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            train_loss_avg = total_loss_train / len(self.train_dataset)
            train_acc_avg = total_acc_train / len(self.train_dataset)
            val_loss_avg = total_loss_val / len(self.val_dataset)
            val_acc_avg = total_acc_val / len(self.val_dataset)

            print(
                f"Epoch: {epoch_num + 1} | Train Loss: {train_loss_avg:.3f} | Train Accuracy: {train_acc_avg:.3f} | "
                f"Val Loss: {val_loss_avg:.3f} | Val Accuracy: {val_acc_avg:.3f}"
            )