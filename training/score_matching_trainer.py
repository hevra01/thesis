import torch


class ScoreMatchingTrainer:
    def __init__(self, sde, optimizer, criterion, device='cuda'):
        self.sde = sde
        # for convenience: since the score net is already in the sde
        self.model = sde.score_net  
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, x):
        """
        Performs a single training step on a batch.

        Args:
            x (torch.Tensor): A batch of input data

        Returns:
            float: The loss value for this batch
        """
        x = x.to(self.device)
        self.model.train()

        # Sample random timesteps for each example
        t = torch.rand((x.size(0),), device=self.device)

        # Add noise using forward SDE and get the target noise
        x_noisy, eps = self.sde.solve_forward_sde(x, t, return_eps=True)

        # Predict the noise using the model
        pred_score = self.model(x_noisy, t)

        # Compute loss between predicted and true noise
        loss = self.criterion(pred_score, eps)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train_epoch(self, dataloader):
        """
        Trains the model for one full epoch over the dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The training data

        Returns:
            float: Average loss over the epoch
        """
        total_loss = 0.0
        total_batches = 0
        self.model.train()  # Set model to training mode once per epoch

        for batch in dataloader:
            print(batch.keys())
            exit()
            # Unpack batch if it’s a (data, label) pair
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            # Call the per-batch training step
            loss = self.train_step(batch)

            total_loss += loss
            total_batches += 1

        return total_loss / total_batches

