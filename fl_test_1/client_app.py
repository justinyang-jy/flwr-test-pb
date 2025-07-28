"""fl-test-1: A Flower / PyTorch app with SCAFFOLD client."""

import torch
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_test_1.task import Net, get_weights, load_data, set_weights, test, train


class ScaffoldClient(NumPyClient):
    """SCAFFOLD client with simplified control variate handling."""
    
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # Initialize client control variates
        self.control_variates = None

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Initialize control variates if needed
        if self.control_variates is None:
            self.control_variates = [np.zeros_like(param) for param in parameters]
        
        # Check if this is a SCAFFOLD round
        use_scaffold = config.get("use_scaffold", False)
        
        if use_scaffold:
            print(f"SCAFFOLD Client training (Round {config.get('server_round', 0)})")
            # For now, use regular training but prepare for SCAFFOLD
            initial_weights = get_weights(self.net)
            train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
            final_weights = get_weights(self.net)
            
            # Simple SCAFFOLD control variate update
            # c_i^+ = c_i - (1/K) * (x_final - x_initial)
            K = self.local_epochs * len(self.trainloader)
            updated_control_variates = []
            for i, (init_w, final_w, c_i) in enumerate(zip(initial_weights, final_weights, self.control_variates)):
                # Simplified SCAFFOLD update
                c_new = c_i - (1.0 / max(K, 1)) * (final_w - init_w)
                updated_control_variates.append(c_new)
            
            self.control_variates = updated_control_variates
            
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {
                    "train_loss": train_loss,
                    "scaffold_round": config.get('server_round', 0),
                },
            )
        else:
            # Regular FedAvg training
            print("Regular FedAvg training")
            train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss},
            )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config.get("alpha", 0.3)
    local_epochs = context.run_config.get("local-epochs", 1)
    
    # Load data with non-IID distribution
    trainloader, valloader = load_data(partition_id, num_partitions, alpha)
    
    print(f"SCAFFOLD Client {partition_id}: Loaded {len(trainloader.dataset)} training samples, "
          f"{len(valloader.dataset)} validation samples with alpha={alpha}")

    # Return SCAFFOLD Client instance
    return ScaffoldClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
