import torch
import torch.nn as nn
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from tefl.task import INPUT_SIZE, TEFederatedModel

app = ServerApp()


# model version with backbone only since head is not shared between nodes
class BackboneOnlyModel(nn.Module):
    HIDDEN_SIZE = TEFederatedModel.HIDDEN_SIZE
    DROPOUT = TEFederatedModel.DROPOUT

    def __init__(self, input_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
        )

    def forward(self, x):
        return self.backbone(x)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]

    global_model = BackboneOnlyModel(INPUT_SIZE)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg 
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final backbone to disk
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_backbone.pt")
