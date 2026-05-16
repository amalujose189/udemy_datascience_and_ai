import torch
from torchvision import datasets,transforms # mnsit,cfiar10,svhn datasets
from torch.utils.data import DataLoader # datset loader for shuffling and batching ,loading data in parallel using multiprocessing workers 
import torch.nn as nn # for building neural network
import torch.nn.functional as F # for activation functions and other functional operations


#define transformation
transform=transforms.Compose([ #CONBAIN multiple transformation in one
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)) #normalize the data to have mean 0.5 and std 0.5 for each channel MEAN AND STD DEV
])
#LOAD THE DATASET
train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform) #download the data if not present and apply transformation
#root='./data' where the data is stored
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform= transform)
#CREATE DATA LOADERS
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False)
# Now you can use train_loader and test_loader in your training loop

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

#define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten() #flatten the input image from 28x28 to 784 means 2d to 1d
        self.fc1=nn.Linear(28*28,128) #input layer to hidden layer flattened input size is 28*28=784 and hidden layer has 128 neurons
        self.fc2=nn.Linear(128,64) #hidden layer to hidden layer output size is 128 and hidden layer has 64 neurons
        self.fc3=nn.Linear(64,10) #hidden layer to output layer 64 converted to 10 output classes (digits 0-9)

    def forward(self,x):
        x=self.flatten(x) #flatten the input image
        x=F.relu(self.fc1(x)) #apply relu activation function after first layer
        x=F.relu(self.fc2(x)) #apply relu activation function after second layer
        x=self.fc3(x) #output layer (no activation function here because we will use CrossEntropyLoss which applies softmax)
        return x
# Create an instance of the model
model=NeuralNetwork()
print(model)
# Define the loss function and optimizer
criterion=nn.CrossEntropyLoss() #cross entropy loss for multi-class classification #tend different output to the true labels and calculate the loss
optimizer=torch.optim.Adam(model.parameters(),lr=0.001) #Adam optimizer with learning rate of 0.001

#training loop
def train(model,train_loader,criterion,optimizer,epochs=5):
    model.train() #set the model to training mode
    for epoch in range(epochs): 
        running_loss=0.0 #initialize running loss for each epoch    
        for images,labels in train_loader:#iterate through the training data    
            optimizer.zero_grad() #zero the gradients
            outputs=model(images) #forward pass
            loss=criterion(outputs,labels) #calculate the loss
            loss.backward() #backward pass
            optimizer.step() #update the weights
            running_loss += loss.item()#accumulate the loss for each batch #calculate the average loss for the epoch and print it
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
# Call the training function
train(model,train_loader,criterion,optimizer)

#evaluation loop
def evaluation_model(model,test_loader): 
    model.eval() #set the model to evaluation mode
    correct=0
    total=0
    with torch.no_grad(): #disable gradient calculation for evaluation efficiency
        for images,labels in test_loader:
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1) #first one don't care 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100*correct/total:.2f}%")
# Call the evaluation function
evaluation_model(model,test_loader)
#save the model
torch.save(model.state_dict(),'mnist_model.pth') #save the model parameters to a file named 'mnist_model.pth'

#reload the model
loaded_model=NeuralNetwork() #create a new instance of the model
loaded_model.load_state_dict(torch.load('mnist_model.pth')) #load the saved model parameters into the new model instance
#verify the loaded model performance    
evaluation_model(loaded_model,test_loader) #evaluate the loaded model on the test dataset to verify it works correctly

#update optimizer for the loaded model
optimizer=torch.optim.Adam(loaded_model.parameters(),lr=0.001) #create a new optimizer for the loaded model parameters  
train(loaded_model,train_loader,criterion,optimizer) #continue training the loaded model for additional epochs
evaluation_model

