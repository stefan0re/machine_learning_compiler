import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_and_save_dataset():

    # load and preprocess the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Export to csv:

    # Create a DataFrame with the data and feature names
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add the target column (species)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Save to CSV
    df.to_csv("iris.csv", index=False)

    return X_train, X_test, y_train, y_test


# Define the model
class BasicNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(BasicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size, bias=True)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size, bias=True)
        self.fc3 = nn.Linear(hidden2_size, output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # no activation here because CrossEntropyLoss expects raw logits
        return self.fc3(x)


# save weights and biases to file
def save_model(model, filename="model.torchpp"):
    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                flat = param.detach().numpy().flatten()
                line = ",".join(str(x) for x in flat)
                f.write(line + "\n")


def test_model(model):
    # Evaluation mode
    model.eval()

    # Disable gradient computation (faster & safer for inference)
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Model setup
a = 1 # batch size
b = 4 # Iris features
c = 64
d = 16
e = 3 # Iris species

X_train, X_test, y_train, y_test = load_and_save_dataset()

model = BasicNet(b, c, d, e)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

save_model(model)

test_model(model)

# Print the einsum-notation for the model
print(f"a: {a}"); 
print(f"b: {b}"); 
print(f"c: {c}"); 
print(f"d: {d}"); 
print(f"e: {e}"); 
print(f"[[[b,a],[c,b]->[c,a]],[d,c]->[d,a]],[e,d]->[e,a]"); 
