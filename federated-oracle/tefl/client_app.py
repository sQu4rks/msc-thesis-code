import os

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tefl.task import (
    create_model,
    load_data,
    partition_to_node,
    test as test_fn,
    train as train_fn,
)

# Directory to save heads after training
HEADS_DIR = os.path.join(os.path.dirname(__file__), "..", "heads")

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    node_name = partition_to_node(partition_id)

    # Create model for this node
    model = create_model(node_name)

    # Load backbone weights from server
    server_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_backbone_state_dict(server_state_dict)

    # Restore local head from context.state 
    if "head_state" in context.state:
        head_state = context.state["head_state"].to_torch_state_dict()
        current_state = model.state_dict()
        for k, v in head_state.items():
            if k.startswith('head.'):
                current_state[k] = v
        model.load_state_dict(current_state)

    # Get device
    device = torch.device("cpu")
    model.to(device)

    # Load data
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, batch_size)

    # Train
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # save head to context.state for next round
    head_state_dict = {k: v for k, v in model.state_dict().items() if k.startswith('head.')}
    context.state["head_state"] = ArrayRecord(head_state_dict)

    # Return only backbone weights for aggregation
    backbone_state_dict = model.get_backbone_state_dict()
    model_record = ArrayRecord(backbone_state_dict)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Get node info
    partition_id = context.node_config["partition-id"]
    node_name = partition_to_node(partition_id)

    # Create model for this node
    model = create_model(node_name)

    # Load backbone weights from server
    server_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_backbone_state_dict(server_state_dict)

    # get local head from context
    if "head_state" in context.state:
        head_state = context.state["head_state"].to_torch_state_dict()
        current_state = model.state_dict()
        for k, v in head_state.items():
            if k.startswith('head.'):
                current_state[k] = v
        model.load_state_dict(current_state)

    # Get device and send model to device
    device = torch.device("cpu")
    model.to(device)

    # Load data
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, batch_size)

    # Evaluate
    eval_loss, eval_mae = test_fn(model, valloader, device)

    # Save head to disk - overwrite previous saved
    os.makedirs(HEADS_DIR, exist_ok=True)
    head_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith('head.')}
    head_path = os.path.join(HEADS_DIR, f"{node_name}_head.pt")
    torch.save(head_state_dict, head_path)

    # Return metrics
    metrics = {
        "eval_loss": eval_loss,
        "eval_mae": eval_mae,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
