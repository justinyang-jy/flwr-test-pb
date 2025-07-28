"""fl-test-1: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.models as models
import random


class ResNet18(nn.Module):
    """ResNet-18 model for CIFAR-10 classification"""

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Use ResNet-18 without pretrained weights and modify for CIFAR-10
        self.resnet = models.resnet18(weights=None)
        # Modify first conv layer for CIFAR-10 (32x32 instead of 224x224)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for smaller input size
        # Modify final layer for CIFAR-10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Alias for backward compatibility
Net = ResNet18


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, alpha: float = 0.3):
    """Load partition CIFAR10 data with non-IID Dirichlet distribution."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            min_partition_size=10,
            self_balancing=True,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_random_local_epochs(min_epochs: int, max_epochs: int) -> int:
    """Get random number of local epochs for client heterogeneity."""
    return random.randint(min_epochs, max_epochs)


def initialize_control_variates(net):
    """Initialize control variates for SCAFFOLD algorithm."""
    control_variates = []
    for param in net.parameters():
        control_variates.append(torch.zeros_like(param.data).cpu().numpy())
    return control_variates


def update_control_variates(old_weights, new_weights, global_weights, control_variates, lr):
    """Update client control variates for SCAFFOLD."""
    updated_control_variates = []
    
    for i, (old_w, new_w, global_w, c_i) in enumerate(zip(old_weights, new_weights, global_weights, control_variates)):
        # Convert to tensors for computation
        old_w_tensor = torch.tensor(old_w)
        new_w_tensor = torch.tensor(new_w)
        global_w_tensor = torch.tensor(global_w)
        c_i_tensor = torch.tensor(c_i)
        
        # SCAFFOLD control variate update: c_i^+ = c_i - c + (1/K*lr) * (x_i - y_i)
        # where x_i is old weights, y_i is new weights, K is local steps
        option_1 = c_i_tensor - (1.0 / lr) * (new_w_tensor - old_w_tensor)
        updated_control_variates.append(option_1.cpu().numpy())
    
    return updated_control_variates


def scaffold_train(net, trainloader, epochs, device, global_weights, control_variates, global_control_variates, lr=1.0):
    """Train with SCAFFOLD algorithm using control variates."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    # Store initial weights
    initial_weights = get_weights(net)
    
    net.train()
    running_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # SCAFFOLD gradient correction
            with torch.no_grad():
                for i, param in enumerate(net.parameters()):
                    if param.grad is not None:
                        # Apply control variate correction
                        c_i = torch.tensor(control_variates[i]).to(device)
                        c_global = torch.tensor(global_control_variates[i]).to(device)
                        param.grad.data += c_i - c_global
            
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1
    
    final_weights = get_weights(net)
    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    
    # Update control variates
    updated_control_variates = update_control_variates(
        initial_weights, final_weights, global_weights, control_variates, lr
    )
    
    return final_weights, avg_loss, updated_control_variates
