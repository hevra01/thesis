import torch


class Diffusion:
    def __init__(self, sde, optimizer, criterion, sampling_transform=None, device='cuda'):
        self.sde = sde
        # for convenience: since the score net is already in the sde
        self.model = sde.score_net  
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.sampling_transform = sampling_transform

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

    def sample(self, num, batch_size=128, timesteps=1000, sample_shape=None, stochastic=True, sampling_transform=True):
        """
        The sample function generates synthetic data samples by solving the reverse diffusion process. 
        It starts with random noise and iteratively refines it using the reverse stochastic differential 
        equation (SDE) or reverse ordinary differential equation (ODE), depending on the stochastic argument.

        Args:
            num (int): The number of samples to generate. The function will keep generating samples in batches until this number is reached.
            batch_size (int): The batch size for sampling
            timesteps (int): The number of timesteps for the reverse diffusion process.
            sample_shape (tuple): The shape of the samples to generate. e.g. for mnist, (1, 28, 28).
            stochastic (bool): Whether to use stochastic sampling or not: solve_reverse_sde or solve_reverse_ode.
            sampling_transform (bool): Determines whether to apply the optional sampling_transform to the generated samples.

        Returns:
            torch.Tensor: The generated samples
        """

        # this list will store the generated samples
        samples = []

        # Set the model to evaluation mode
        self.model.eval() 

        # Disable gradient computation
        with torch.no_grad():  
            # len(samples) represents the number of batches, not the total number of samples generated so far.
            while len(samples) * batch_size < num:
                # this is to safely handle the last batch
                iter_batch_size = min(batch_size, num - batch_size * len(samples))

                if sample_shape is not None:
                    noise_shape = (iter_batch_size,) + tuple(sample_shape)
                else:
                    noise_shape = (iter_batch_size,)

                # mapping from the gaussian noise to target distribution. 
                noise = torch.randn(noise_shape).to(self.device)
                # attention: here we are really assuming that we 
                # are using an mlp hence we are flattening the data.
                noise_flat = noise.view(noise.size(0), -1)
                # Solve the reverse diffusion process
                if stochastic:
                    sample = self.sde.solve_reverse_sde(x_start=noise_flat, steps=timesteps)
                else:
                    sample = self.sde.solve_reverse_ode(x_start=noise, steps=timesteps)
                samples.append(sample.cpu())

        samples = torch.cat(samples)
        # apply the optional sampling transform if it is provided
        if sampling_transform is not None:
            samples = sampling_transform(samples)
        return samples