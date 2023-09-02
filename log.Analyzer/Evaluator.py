import torch
from sklearn.metrics import precision_recall_fscore_support

class Evaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.THRESHOLD = 0.7

        if torch.cuda.is_available():
            self.model = model.cuda()

    def evaluate(self):
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=2)

        true_labels = []
        predicted_labels = []
        total_acc_test = 0

        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(self.device)
                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

                predicted = output.argmax(dim=1)
                true_labels.extend(test_label.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                acc = (predicted == test_label).sum().item()
                total_acc_test += acc

        accuracy = total_acc_test / len(self.test_dataset)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )

        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1_score:.3f}")

    # This would be the implementation to test on the whole log and not only on the 512 length chunks
    def evaluate_whole_log(self):
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=2)

        total_log_acc = 0
        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(self.device)
                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_log_acc += acc
            return total_log_acc / len(self.test_dataset)
