import pyautogui
import cv2
import numpy as np
import time


# Define the coordinates of the area you want to capture
x1, y1 = 100, 100  # Top-left corner
x2, y2 = 500, 500  # Bottom-right corner

# Create a window to display the captured screenshots
cv2.namedWindow("Screenshot", cv2.WINDOW_NORMAL)

# Continuously capture the screen image of the specific area every 1 second
while True:
    # Capture the screen image of the specific area
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    
    # Convert the screenshot to an OpenCV image
    screenshot_cv2 = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Display the screenshot in a window
    cv2.imshow("Screenshot", screenshot_cv2)
    
    # Wait for 1 second before capturing the next screenshot
    time.sleep(1)
    
    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows when the program ends
cv2.destroyAllWindows()



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define a Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the network, loss function, and optimizer
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(net.state_dict(), 'mnist_cnn.pth')

