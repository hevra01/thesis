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

        # we are only using the images for training, the model is learning to predict the noise. 
        # Has nothing to do with the labels.
        clean_images = batch["image"].to(self.device)

        # Sample random timesteps for each example
        t = torch.rand((clean_images.size(0),), device=self.device)
        
        # Add noise using forward SDE and get the target noise
        noisy_images, eps = self.sde.solve_forward_sde(clean_images, t, return_eps=True)

        # since we are using an MLP as the score net, we need to flatten the input
        noisy_images_flat = noisy_images.view(noisy_images.size(0), -1)
        eps_flat = eps.view(eps.size(0), -1)


        # Predict the noise using the model.
        pred_score = self.model(noisy_images_flat, t)

        # Compute loss between predicted and true noise
        # remember that the model predicts the non-scaled noise.
        # so when we compute the loss, we compare it to also the epsilon noise and not the 
        # sigma(t) * eps. also, the model predicts the noise that needs to be removed 
        # from the data to bring it closer to the data distribution.
        # So we need to take the negative of the noise.
        loss = self.criterion(pred_score, -eps_flat)

        # Clear old gradients
        self.optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update parameters
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

            # Call the per-batch training step
            loss = self.train_step(batch)

            total_loss += loss
            total_batches += 1

        return total_loss / total_batches

