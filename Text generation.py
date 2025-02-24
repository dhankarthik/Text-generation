import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import RegexpTokenizer

class TextDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.long)
        return x, y

def load_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def tokenize_words(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    return tokens

def convert_text_to_numbers(text, chars):
    char_to_num = dict((c, i) for i, c in enumerate(chars))
    return [char_to_num[char] for char in text]

def generate_text(model, chars, num_to_char, pattern, vocab_len):
    for i in range(1000):
        x = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0)
        x = x.view(-1, 100, 1)
        x = x / float(vocab_len)
        prediction = model(x)
        index = torch.argmax(prediction)
        result = num_to_char[index.item()]
        seq_in = [num_to_char[value] for value in pattern]
        print(result, end="")
        pattern.append(index.item())
        pattern = pattern[1:len(pattern)]

def main():
    file_path = "C:/Users/aeaka/PYTHON CODES/frankenstein.txt"
    text = load_data(file_path)
    processed_inputs = tokenize_words(text)
    chars = sorted(list(set(processed_inputs)))
    vocab_len = len(chars)
    char_to_num = dict((c, i) for i, c in enumerate(chars))
    num_to_char = dict((i, c) for i, c in enumerate(chars))
    seq_length = 100
    x_data = []
    y_data = []
    for i in range(0, len(processed_inputs) - seq_length, 1):
        in_seq = processed_inputs[i:i + seq_length]
        out_seq = processed_inputs[i + seq_length]
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])
    dataset = TextDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(256, vocab_len)

        def forward(self, x):
            x = x.view(-1, 100, 1)
            h0 = torch.zeros(1, x.size(0), 256).to(x.device)
            c0 = torch.zeros(1, x.size(0), 256).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(2):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    start = np.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]
    print("Random Seed:")
    print(" ", ''.join([num_to_char[value] for value in pattern]), "")
    generate_text(model, chars, num_to_char, pattern, vocab_len)

if __name__ == "__main__":
    main()
