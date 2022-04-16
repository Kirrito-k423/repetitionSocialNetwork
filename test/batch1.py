import torch
from torch.utils.data import DataLoader, TensorDataset


# Create the dataset with N_SAMPLES samples
N_SAMPLES, D_in, H, D_out = 10000, 1000, 100, 10

x = torch.randn(N_SAMPLES, D_in)
y = torch.randn(N_SAMPLES, D_out)

# Define the batch size and the number of epochs
BATCH_SIZE = 64
N_EPOCHS = 2

# Use torch.utils.data to create a DataLoader
# that will take care of creating batches
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model, loss and optimizer
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Get the dataset size for printing (it is equal to N_SAMPLES)
dataset_size = len(dataloader.dataset)

# Loop over epochs
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Loop over batches in an epoch using DataLoader
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):

        print(id_batch)
        print(x_batch.size())
        print(y_batch.size())
        y_batch_pred = model(x_batch)

        loss = loss_fn(y_batch_pred, y_batch)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Every 100 batches, print the loss for this batch
        # as well as the number of examples processed so far
        if id_batch % 100 == 0:
            loss, current = loss.item(), (id_batch + 1) * len(x_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")
