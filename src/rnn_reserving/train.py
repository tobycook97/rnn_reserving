import torch 

def train_epoch(model, dataloader, optimizer, criterion):

  model.train()
  total_loss = 0
  num_batches = 0

  # create mask to avoid the padding!
  for batch in dataloader:
    inputs, lengths, targets = batch
    optimizer.zero_grad()
    predictions = model(inputs, lengths)

    mask = torch.zeros_like(predictions, dtype=torch.bool)
    for i, length in enumerate(lengths):
      mask[i, :length] = True

    valid_preds = predictions[mask]
    valid_targets = targets[mask]

    loss = criterion(valid_preds, valid_targets)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    num_batches += 1

  return total_loss / num_batches


def eval_epoch(model, dataloader, criterion):

  model.eval()
  total_loss = 0
  num_batches = 0

  # create mask to avoid the padding!
  with torch.no_grad():

    for batch in dataloader:
      inputs, lengths, targets = batch

      predictions = model(inputs, lengths)

      mask = torch.zeros_like(predictions, dtype=torch.bool)

      for i, length in enumerate(lengths):
        mask[i, :length] = True

      valid_preds = predictions[mask]
      valid_targets = targets[mask]
      loss = criterion(valid_preds, valid_targets)

      total_loss += loss.item()
      num_batches += 1

  return total_loss / num_batches
