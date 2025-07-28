"""fl-test-1: A Flower / PyTorch app with SCAFFOLD strategy."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from fl_test_1.task import Net, get_weights, initialize_control_variates


class ScaffoldStrategy(Strategy):
    """SCAFFOLD federated learning strategy with control variates."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        scaffold_lr: float = 1.0,
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.scaffold_lr = scaffold_lr
        
        # SCAFFOLD-specific state
        self.global_control_variates: Optional[NDArrays] = None
        self.client_control_variates: Dict[str, NDArrays] = {}
        
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters and control variates."""
        if self.initial_parameters is not None:
            # Initialize global control variates
            initial_ndarrays = parameters_to_ndarrays(self.initial_parameters)
            self.global_control_variates = [np.zeros_like(param) for param in initial_ndarrays]
            return self.initial_parameters
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[str, FitIns]]:
        """Configure the next round of training."""
        config = {
            "server_round": server_round,
            "scaffold_lr": self.scaffold_lr,
        }
        
        # Add global control variates to config
        if self.global_control_variates is not None:
            config["global_control_variates"] = [cv.tolist() for cv in self.global_control_variates]
        
        fit_ins = FitIns(parameters, config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        return [(client.cid, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[str, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
            
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        return [(client.cid, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, FitRes]],
        failures: List[Union[Tuple[str, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using SCAFFOLD algorithm."""
        if not results:
            return None, {}

        # Extract weights and control variates from results
        weights_results = []
        control_variates_results = []
        num_examples_total = 0
        
        for client_id, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((weights, fit_res.num_examples))
            num_examples_total += fit_res.num_examples
            
            # Extract client control variates from metrics
            if "control_variates" in fit_res.metrics:
                client_cv = [np.array(cv) for cv in fit_res.metrics["control_variates"]]
                control_variates_results.append((client_id, client_cv))

        # Weighted average of model parameters
        aggregated_weights = [
            np.zeros_like(weights_results[0][0][i]) for i in range(len(weights_results[0][0]))
        ]
        
        for weights, num_examples in weights_results:
            for i, layer_weights in enumerate(weights):
                aggregated_weights[i] += layer_weights * (num_examples / num_examples_total)

        # Update global control variates (SCAFFOLD server update)
        if control_variates_results and self.global_control_variates is not None:
            # Average client control variates
            num_clients = len(control_variates_results)
            new_global_cv = [np.zeros_like(cv) for cv in self.global_control_variates]
            
            for client_id, client_cv in control_variates_results:
                # Store client control variates
                self.client_control_variates[client_id] = client_cv
                
                # Accumulate for global average
                for i, cv in enumerate(client_cv):
                    new_global_cv[i] += cv / num_clients
            
            # Update global control variates
            self.global_control_variates = new_global_cv

        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            train_losses = [fit_res.metrics.get("train_loss", 0.0) for _, fit_res in results]
            metrics_aggregated["train_loss_avg"] = sum(train_losses) / len(train_losses)
            metrics_aggregated["num_clients"] = len(results)

        return ndarrays_to_parameters(aggregated_weights), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[str, EvaluateRes]],
        failures: List[Union[Tuple[str, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Aggregate accuracy and loss
        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        losses = [evaluate_res.loss for _, evaluate_res in results]
        
        metrics_aggregated = {
            "accuracy": sum(accuracies) / len(accuracies),
            "loss": sum(losses) / len(losses),
        }
        
        return sum(losses) / len(losses), metrics_aggregated

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for training."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        # No centralized evaluation in this implementation
        return None


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    scaffold_lr = context.run_config.get("scaffold-lr", 1.0)
    
    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define SCAFFOLD strategy
    strategy = ScaffoldStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        scaffold_lr=scaffold_lr,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
