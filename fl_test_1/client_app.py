"""fl-test-1: A Flower / PyTorch app with SCAFFOLD client."""

import torch
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_test_1.task import (
    Net, 
    get_weights, 
    load_data, 
    set_weights, 
    test, 
    train,
    get_random_local_epochs,
    initialize_control_variates,
    scaffold_train
)


# Define Flower Client with SCAFFOLD support
class ScaffoldClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs_range):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs_range = local_epochs_range
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # SCAFFOLD-specific state
        self.control_variates = initialize_control_variates(self.net)
        self.client_id = None

    def fit(self, parameters, config):
        """Train the model using SCAFFOLD algorithm."""
        # Set model parameters
        set_weights(self.net, parameters)
        
        # Get configuration
        scaffold_lr = config.get("scaffold_lr", 1.0)
        server_round = config.get("server_round", 1)
        
        # Get global control variates from server
        global_control_variates = config.get("global_control_variates", None)
        if global_control_variates is None:
            # Initialize if not provided
            global_control_variates = [np.zeros_like(cv) for cv in self.control_variates]
        else:
            # Convert from list format
            global_control_variates = [np.array(cv) for cv in global_control_variates]
        
        # Variable local epochs for client heterogeneity
        local_epochs = get_random_local_epochs(
            self.local_epochs_range[0], 
            self.local_epochs_range[1]
        )
        
        print(f"Client training for {local_epochs} epochs in round {server_round}")
        
        # Get current global weights for SCAFFOLD
        global_weights = [param.copy() for param in parameters]
        
        # Train with SCAFFOLD
        updated_weights, train_loss, updated_control_variates = scaffold_train(
            self.net,
            self.trainloader,
            local_epochs,
            self.device,
            global_weights,
            self.control_variates,
            global_control_variates,
            scaffold_lr
        )
        
        # Update client control variates
        self.control_variates = updated_control_variates
        
        # Prepare metrics including control variates
        metrics = {
            "train_loss": train_loss,
            "local_epochs": local_epochs,
            "control_variates": [cv.tolist() for cv in self.control_variates]
        }
        
        return (
            updated_weights,
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the validation set."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        # Additional metrics for analysis
        metrics = {
            "accuracy": accuracy,
            "val_loss": loss,
        }
        
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get configuration parameters
    alpha = context.run_config.get("alpha", 0.05)  # More aggressive non-IID
    local_epochs_min = context.run_config.get("local-epochs-min", 1)
    local_epochs_max = context.run_config.get("local-epochs-max", 5)
    
    # Load data with extreme non-IID distribution
    trainloader, valloader = load_data(partition_id, num_partitions, alpha)
    
    # Create client with variable local epochs
    local_epochs_range = (local_epochs_min, local_epochs_max)
    
    print(f"Client {partition_id}: Loaded {len(trainloader.dataset)} training samples, "
          f"{len(valloader.dataset)} validation samples")
    print(f"Client {partition_id}: Local epochs range: {local_epochs_range}")
    
    # Return SCAFFOLD Client instance
    return ScaffoldClient(net, trainloader, valloader, local_epochs_range).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
