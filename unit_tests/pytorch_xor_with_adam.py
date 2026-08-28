import torch

x = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
])

y = torch.tensor([[0.], [1.], [1.], [0.]])

model = torch.nn.Sequential(
    torch.nn.Linear(2, 2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid(),
)

with torch.no_grad():
    model[0].weight[:] = torch.tensor([
        [0.364816, 0.859271],
        [0.419038, 0.481612],
    ])
    model[0].bias[:] = torch.tensor([0.067053, 0.837133])

    model[2].weight[:] = torch.tensor([
        [-0.837809, 0.644721],
    ])
    model[2].bias[:] = torch.tensor([-0.881249])

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
loss_fn = torch.nn.MSELoss(reduction="sum")

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    print(i + 1, loss.item())
