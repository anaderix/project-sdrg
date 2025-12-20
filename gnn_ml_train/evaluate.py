import torch

def evaluate(model, dataset, indices):
    model.eval()
    correct = 0

    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            logits = model(data)          # [num_edges]
            pred = logits.argmax().item() # single edge
            correct += int(pred == data.y.item())

    return correct / len(indices)
