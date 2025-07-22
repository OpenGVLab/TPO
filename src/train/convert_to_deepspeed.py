import torch


# Load the normal checkpoint
checkpoint = torch.load('normal_checkpoint.pt')

import deepspeed

# Initialize the DeepSpeed model
model = your_model_class() # Your model definition
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    checkpoint=checkpoint
)

deepspeed.save_checkpoint('deepspeed_checkpoint', 0)