


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dummy time series data: 100 samples, sequence length 10, 1 feature


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=2, bidirectional=True, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            bidirectional=bidirectional)
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
def run(X, y, new = False, num_epochs = 20):
    
    try:
        if new:
            raise FileNotFoundError
        model = LSTMModel(input_size=4, hidden_size=50, num_layers=1, num_classes=2)
        model.load_state_dict(torch.load('slidingwindowclaudebackend/pickle_saves/modellen/lstm_trained_model.pth'))

    except FileNotFoundError:
        # Data preparation
        X_tensor = torch.tensor(X, dtype=torch.float32)  # (n, feature_length)
        y_tensor = torch.tensor(y, dtype=torch.long)       # (n,)
        X_tensor = X_tensor.unsqueeze(1)  # (n, 1, feature_length)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Instantiate model with adjustments:
        model = LSTMModel(input_size=4, hidden_size=50, num_layers=1, num_classes=2, bidirectional=True, dropout=0.5)
        
        # Compute class weights:
        num_total = len(y)
        num_pos = (y == 1).sum()  # make sure y is a numpy array
        num_neg = num_total - num_pos
        weight_for_0 = num_total / (2 * num_neg)
        weight_for_1 = num_total / (2 * num_pos)
        class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        print("Starting training...")
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, targets in loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            
            avg_loss = epoch_loss / len(loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        # Save the model after training
        torch.save(model.state_dict(), 'slidingwindowclaudebackend/pickle_saves/modellen/lstm_trained_model.pth')

        # Later, load the model:
        # loaded_model = LSTMModel(input_size=4, hidden_size=50, num_layers=1, num_classes=2)
        # loaded_model.load_state_dict(torch.load('lstm_trained_model.pth'))
        # loaded_model.eval()  # Set to evaluation mode for inference

    return model
