import os.path

import numpy as np
from giza_actions.action import Action, action
from giza_actions.task import task
import torch
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv, find_dotenv

from config.fc_config import configs
from gizathon_project.src.data_preprocessing.fc_preprocessing import get_train_test_task
from gizathon_project.src.data_preprocessing.data_handlers import load_data
from gizathon_project.src.models.BasicFC import BasicFC

has_cuda = torch.cuda.is_available()
load_dotenv(find_dotenv())
device = "cpu"
if has_cuda:
    device = "cuda"

@task(name='Convert To ONNX')
def convert_to_onnx(model, input_size, filename):
    dummy_input = torch.randn(1, input_size).to(device, dtype=torch.float32)
    path = os.path.join(os.environ["ONNX_DIR"], filename)
    torch.onnx.export(model, dummy_input, path, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, )
    print(f"Model has been converted to ONNX and saved as {path}")

@task(name="Training!")
def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs):
    best_loss, patience, valid_losses, train_losses, best_model = np.inf, 0, [], [], None
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            outputs = model.forward(X_batch.to(device)).squeeze()
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch.to(device))
            if outputs[0] == outputs[1] == outputs[2]:
                pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch.to(device)).squeeze()
                if len(outputs.shape) == 0:
                    y_batch = y_batch[0]
                loss = criterion(outputs, y_batch.to(device))
                valid_loss += loss.item()
        running_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {running_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        train_losses.append(running_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_loss - 0.3:
            best_loss, patience = valid_loss, 0
            best_model = model
        else:
            patience += 1
        # Stop after 4 epochs without validation loss improvement
        if patience >= 20:
            print("Early stopping")
            return best_model, train_losses, valid_losses
    return best_model, train_losses, valid_losses

@action(name="Train Fully Connected", log_prints=True)
def main():
    df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_task(df, window=configs["lookback_window"])
    model = BasicFC(len(X_train[0]), configs["hidden_size_1"], configs["hidden_size_2"], configs["hidden_size_3"])
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], weight_decay=1e-5)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=configs["batch_size"], shuffle=False)
    best_model, train_losses, valid_losses = train(model, train_loader, valid_loader, criterion, optimizer, configs["n_epochs"])
    convert_to_onnx(best_model, len(X_train[0]), configs["filename"])



if __name__ == "__main__":
    main()
