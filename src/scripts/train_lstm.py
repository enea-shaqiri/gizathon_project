import os.path

import numpy as np
#from giza_actions.action import Action, action
import torch
from dotenv import load_dotenv, find_dotenv

from src.data_preprocessing.baseline_lstm_preprocessing import get_train_test
from src.data_preprocessing.data_handlers import load_data
from src.models.BasicLSTM import BasicLSTM
from torch.utils.data import DataLoader, TensorDataset
has_cuda = torch.cuda.is_available()
load_dotenv(find_dotenv())
device = "cpu"
if has_cuda:
    device = "cuda"

#@task(name='Convert To ONNX')
def convert_to_onnx(model, input_size, filename):
    dummy_input = torch.randn(1, input_size).unsqueeze(-1).to(device, dtype=torch.float64)
    path = os.path.join(os.environ["ONNX_DIR"], filename)
    torch.onnx.export(model, dummy_input, path, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},)
    print(f"Model has been converted to ONNX and saved as {path}")

#@task(name="Training!")
def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs):
    best_loss, patience, valid_losses, train_losses, best_model = np.inf, 0, [], [], None
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            # The input has form (batch_size, sequence_length, number_of_features)
            outputs = model.forward(X_batch.unsqueeze(-1).to(device)).squeeze()
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch.unsqueeze(-1).to(device)).squeeze()
                loss = criterion(outputs, y_batch.to(device))
                valid_loss += loss.item()
        running_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {running_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        train_losses.append(running_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_loss - 0.1:
            best_loss, patience = valid_loss, 0
            best_model = model
        else:
            patience += 1
        # Stop after 10 epochs without validation loss improvement
        if patience >= 10:
            print("Early stopping")
            return best_model, train_losses, valid_losses
    return best_model, train_losses, valid_losses

#@action(name="Train LSTM")
def main(input_size, n_features=1, hidden_size=8, n_epochs=40, lr=0.001, batch_size=16, filename="lstm_model.onnx"):
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test(df, window=input_size)
    model = BasicLSTM(n_features, hidden_size)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    best_model, train_losses, valid_losses = train(model, train_loader, valid_loader, criterion, optimizer, n_epochs)
    convert_to_onnx(best_model, input_size, filename)



if __name__ == "__main__":
    main(24, n_epochs=800)