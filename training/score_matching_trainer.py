import torch


class ScoreMatchingTrainer:
    def __init__(self, sde, optimizer, criterion, device='cuda'):
        self.sde = sde
        # for convenience: since the score net is already in the sde
        self.model = sde.score_net  
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, batch):
        """
        Performs a single training step on a batch.

        Args:
            batch (dict): A batch of input data

        Returns:
            float: The loss value for this batch
        """

        # Sample random timesteps for each example
        t = torch.rand((batch.size(0),), device=self.device)
        

        # Add noise using forward SDE and get the target noise
        x_noisy, eps = self.sde.solve_forward_sde(batch, t, return_eps=True)

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
        self.model.train()  # Set model to training mode once per epoch
        total_loss = 0.0
        total_batches = 0

        # batch[0] is the data, batch[1] is the label
        for batch in dataloader:
            print(batch["image"][0])
            print(batch["image"][0].shape)

            # Call the per-batch training step
            loss = self.train_step(batch["image"])

            total_loss += loss
            total_batches += 1

        return total_loss / total_batches

