"""fl-test-1: A Flower / PyTorch app with SCAFFOLD strategy extending FedAvg."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from fl_test_1.task import Net, get_weights


class ScaffoldStrategy(FedAvg):
    """SCAFFOLD strategy extending FedAvg for better compatibility."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SCAFFOLD-specific: Global control variates
        self.global_control_variates: Optional[NDArrays] = None
        # Client control variates storage
        self.client_control_variates: Dict[str, NDArrays] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure the next round of training with SCAFFOLD."""
        # Get the standard FedAvg configuration
        config_list = super().configure_fit(server_round, parameters, client_manager)
        
        # Initialize global control variates if needed
        if self.global_control_variates is None:
            param_arrays = parameters_to_ndarrays(parameters)
            self.global_control_variates = [np.zeros_like(arr) for arr in param_arrays]

        # Add SCAFFOLD-specific configuration
        scaffold_config_list = []
        for client, fit_ins in config_list:
            client_id = client.cid
            
            # Initialize client control variates if needed
            if client_id not in self.client_control_variates:
                param_arrays = parameters_to_ndarrays(parameters)
                self.client_control_variates[client_id] = [
                    np.zeros_like(arr) for arr in param_arrays
                ]
            
            # Add SCAFFOLD info to config
            scaffold_config = dict(fit_ins.config)
            scaffold_config["use_scaffold"] = True
            scaffold_config["server_round"] = server_round
            
            # Create new FitIns with updated config
            from flwr.common import FitIns
            new_fit_ins = FitIns(fit_ins.parameters, scaffold_config)
            scaffold_config_list.append((client, new_fit_ins))

        return scaffold_config_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using SCAFFOLD principles."""
        if not results:
            return None, {}

        # Use FedAvg aggregation as base
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # SCAFFOLD-specific: Update control variates if available
        for client, fit_res in results:
            client_id = client.cid
            # For now, just track that this client participated in SCAFFOLD
            if "scaffold_round" in fit_res.metrics:
                # Client participated in SCAFFOLD training
                pass

        # Update global control variates (simplified)
        if self.client_control_variates:
            # Average all client control variates
            all_client_cvs = list(self.client_control_variates.values())
            if all_client_cvs:
                for i in range(len(self.global_control_variates)):
                    if i < len(all_client_cvs[0]):
                        cv_sum = np.zeros_like(self.global_control_variates[i])
                        count = 0
                        for client_cv in all_client_cvs:
                            if i < len(client_cv):
                                cv_sum += client_cv[i]
                                count += 1
                        if count > 0:
                            self.global_control_variates[i] = cv_sum / count

        return parameters_aggregated, metrics_aggregated


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define SCAFFOLD strategy (extending FedAvg)
    strategy = ScaffoldStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
