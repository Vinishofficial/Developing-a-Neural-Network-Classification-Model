# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="704" height="837" alt="image" src="https://github.com/user-attachments/assets/f5dc5c85-9218-4db2-885d-472eb28fa136" />


## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4: 
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM

### Name: VINISH RAJ R

### Register Number: 212223230243

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 =nn.Linear(input_size,32)
        self.fc2 =nn.Linear(32, 16)
        self.fc3 =nn.Linear(16, 8)
        self.fc4 =nn.Linear(8, 4)
    def forward(self, x):
      #Include your code here
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x    
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
train_model(model, train_loader, criterion, optimizer, epochs=100)
```

### Dataset Information
<img width="1246" height="249" alt="image" src="https://github.com/user-attachments/assets/cdbc57a4-8407-46d1-8171-43082cef6a7d" />



### OUTPUT

## Confusion Matrix

<img width="702" height="589" alt="image" src="https://github.com/user-attachments/assets/95f9f792-accb-4448-a4bb-fd778a09e505" />




## Classification Report
<img width="552" height="446" alt="image" src="https://github.com/user-attachments/assets/5b14decc-f466-4fc7-a086-9afde6a60e6a" />



### New Sample Data Prediction
<img width="372" height="102" alt="image" src="https://github.com/user-attachments/assets/6a935d50-39f8-48f0-b76b-0647af566070" />




## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
