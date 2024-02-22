import numpy as np
#from giza_actions.action import Action, action
import torch
from src.data_preprocessing.baseline_lstm_preprocessing import get_train_test
from src.data_preprocessing.data_handlers import load_data
from src.models.BasicLSTM import BasicLSTM
from torch.utils.data import DataLoader, TensorDataset
has_cuda = torch.cuda.is_available()
device = "cpu"
if has_cuda:
    device = "cuda"

#@action(name="Train LSTM")
def train(input_size, n_features=1, hidden_size=8, n_epochs=100, lr=0.001, batch_size=8):
    df = load_data()
    X_train, y_train, X_test, y_test = get_train_test(df, window=input_size)
    model = BasicLSTM(n_features, hidden_size)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            outputs = model.forward(X_batch.unsqueeze(-1).to(device)).squeeze()
            optimizer.zero_grad()

            loss = criterion(outputs, y_batch.to(device))
            # @TODO compute average loss instead of displaying the last one

            loss.backward()

            optimizer.step()
        if epoch % 2 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


if __name__ == "__main__":
    train(24)