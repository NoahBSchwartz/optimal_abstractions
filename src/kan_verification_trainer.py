# ============================================================================
# TRAINING KANS
# ============================================================================

import numpy as np
import scipy as sc
from sklearn.metrics import r2_score
import src.kan_verification_grapher as kan_verification_grapher
import ssl
from torch import autograd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.custom_fastkan import FastKAN
from tqdm import tqdm

# 1. Duffing Oscillator
class LBFGS_duffing_oscillator(torch.optim.LBFGS):
    def __init__(self, params, lr=1, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32):
        super().__init__(params, lr=lr, history_size=history_size, tolerance_grad=tolerance_grad, 
                         tolerance_change=tolerance_change)
        self.tolerance_ys = tolerance_ys

def compute_jacobian_duffing_oscillator(model, x):
    def ret_sum(x):
        return model(x)
    jacb = autograd.functional.jacobian(ret_sum, x, create_graph=True)
    jacb = jacb[:, :, :, 0]
    jacb = jacb.sum(dim=-1).unsqueeze(-1) 
    return jacb

def compute_ode_duffing_oscillator(y_pred, x, device):
    def ode(init_cond, d):
        return init_cond[:, 1], init_cond[:, 0] - init_cond[:, 0]**3 - d*init_cond[:, 1]
    grad = ode(y_pred, x[:, 3])
    grad = torch.tensor([[dxdt, dydt] for dxdt, dydt in zip(grad[0], grad[1])], device=device).unsqueeze(-1) 
    return grad

def compute_forward_integrals_duffing_oscillator(dataset, duration, t_eval, device):
    def system_cont(t, x, d):
        dxdt = [x[1], x[0] - x[0]**3 - d*x[1]]
        return dxdt 
    sol_set = []
    for init_cond in dataset.detach().cpu().numpy():
        t, x0, y0, d = init_cond[0], init_cond[1], init_cond[2], init_cond[3]
        sol = sc.integrate.solve_ivp(system_cont, [0, duration], (x0, y0), args=(d,), t_eval=t_eval)
        sol_set.extend(np.array(sol.y.T))
    sol_set = torch.tensor(np.array(sol_set), dtype=torch.float32, device=device)
    return sol_set

# Loss Function
def ode_residuals_duffing_oscillator(model, coords, steps, duration, t_eval, device, alpha=1.0, beta=1.0, gamma=1.0):
    coords = coords.clone().detach().requires_grad_(True)
    y_pred = model(coords) 
    y_gt = compute_forward_integrals_duffing_oscillator(coords[::steps], duration, t_eval.cpu().numpy(), device)
    kan_autograd = compute_jacobian_duffing_oscillator(model, coords)
    kan_ode_grad = compute_ode_duffing_oscillator(y_pred, coords, device)
    init_kan_grad = y_pred[::steps]
    init_ode_grad = coords[::steps][:, 1:3]
    dyn_loss = torch.mean((kan_autograd - kan_ode_grad)**2)
    integral_loss = torch.mean((y_pred - y_gt)**2)
    init_loss = torch.mean((init_kan_grad - init_ode_grad)**2)
    batch_loss = alpha*dyn_loss + beta*integral_loss + beta*init_loss
    return batch_loss

def train_loop_duffing_oscillator(model, dataset, steps, duration, t_eval, device):
    optimizer = LBFGS_duffing_oscillator(model.parameters(), lr=1, history_size=10, tolerance_grad=1e-32, 
                     tolerance_change=1e-32, tolerance_ys=1e-32)
    iter_ = 10
    for i in range(iter_):
        def closure():
            optimizer.zero_grad()
            loss = ode_residuals_duffing_oscillator(model, dataset, steps, duration, t_eval, device, alpha=0.2, beta=0.6, gamma=0.2)
            loss.backward()
            return loss
        optimizer.step(closure)
        if i % 1 == 0:
            current_loss = closure().item()
            print(f"Iteration {i}, Loss: {current_loss}")

# Create the dataset for the model
def train_duffing_oscillator():
    device = "cpu"
    s_size = 50
    x_sample = np.random.uniform(-1, 1, size=s_size)
    y_sample = np.random.uniform(-1, 1, size=s_size)
    d_sample = np.random.uniform(0.1, 0.5, size=s_size)
    x_sample = torch.tensor(x_sample, dtype=torch.float32, device=device)
    y_sample = torch.tensor(y_sample, dtype=torch.float32, device=device)
    d_sample = torch.tensor(d_sample, dtype=torch.float32, device=device)
    duration = 1
    time_interval = 0.2
    steps = int(duration / time_interval)
    print(f"Steps: {steps}")
    t_eval = torch.linspace(0, duration, steps=steps)
    dataset = torch.stack([x_sample, y_sample, d_sample], dim=1).to(device)
    dataset = dataset.repeat_interleave(steps, dim=0)
    t_eval_set = t_eval.T.repeat(s_size).view(-1, 1).to(device)
    dataset = torch.cat([t_eval_set, dataset], dim=1)
    dataset.requires_grad = True
    print(f"Dataset shape: {dataset.shape}")
    test_size = 5
    test_dataset = dataset[(s_size-test_size)*steps:]
    dataset = dataset[:(s_size-test_size)*steps]
    print(f"Train dataset shape: {dataset.shape}, Test dataset shape: {test_dataset.shape}")

    # Base Update and Layernorm must be set to false for verificaton 
    kan_duffing = FastKAN(layers_hidden=[4, 15, 2], grid_min=-2., grid_max=2., num_grids=15, use_base_update = False, use_layernorm=False)
    print("Starting training...")

    train_loop_duffing_oscillator(kan_duffing, dataset, steps, duration, t_eval, device)
    print("\nTest set predictions:")
    kan_verification_grapher.graph_ode_results(test_dataset, kan_duffing, steps, t_eval, duration)
    print("\nTrain set predictions:")
    kan_verification_grapher.graph_ode_results(dataset, kan_duffing, steps, t_eval, duration)

    print("\nComputing error metrics on test set:")
    sol_map = []
    for init_conds in test_dataset[::steps].detach().cpu().numpy():
        sol, _ = kan_verification_grapher.system_eq_dis(init_conds[1:], t_eval.numpy(), duration)
        sol_map.extend(sol)
    pred = kan_duffing(test_dataset).detach().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (test dataset): {res_score}")
    print("\nComputing error metrics on train set:")
    sol_map = []
    for init_conds in dataset[::steps].detach().cpu().numpy():
        sol, _ = kan_verification_grapher.system_eq_dis(init_conds[1:], t_eval.numpy(), duration)
        sol_map.extend(sol)
    pred = kan_duffing(dataset).detach().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (train dataset): {res_score}")
    return kan_duffing

# 2. Dampened Pendulum
def compute_jacobian_dampened_pendulum(model, x):
    def ret_sum(x):
        return model(x)
    jacb = autograd.functional.jacobian(ret_sum, x, create_graph=True)
    jacb = jacb[:, :, :, 0]
    jacb = jacb.sum(dim=-1).unsqueeze(-1)
    return jacb

def compute_ode_dampened_pendulum(y_pred, x, device):
    def ode(init_cond, a, b, c, d):
        return -a*init_cond[:, 0] + b*init_cond[:, 1] + c*init_cond[:, 2], \
               (d - a)*init_cond[:, 0] - 2*b*init_cond[:, 1], \
               d*init_cond[:, 0] + b*init_cond[:, 1] - c*init_cond[:, 2]

    grad = ode(y_pred, x[:, 4], x[:, 5], x[:, 6], x[:, 7])
    grad = torch.tensor([[dxdt, dydt, dzdt] for dxdt, dydt, dzdt in zip(grad[0], grad[1], grad[2])], 
                        device=device).unsqueeze(-1) 
    return grad

def compute_forward_integrals_dampened_pendulum(dataset, duration, t_eval, device):
    def system_cont(t, x, a, b, c, d):
        dxdt = [-a*x[0] + b*x[1] + c*x[2], 
                (d - a)*x[0] - 2*b*x[1], 
                d*x[0] + b*x[1] - c*x[2]]
        return dxdt 
    
    sol_set = []
    for init_cond in dataset.detach().cpu().numpy():
        t, x0, y0, z0, a, b, c, d = init_cond[0], init_cond[1], init_cond[2], init_cond[3], \
                                    init_cond[4], init_cond[5], init_cond[6], init_cond[7]
        sol = sc.integrate.solve_ivp(system_cont, [0, duration], (x0, y0, z0), 
                                     args=(a, b, c, d), t_eval=t_eval)
        sol_set.extend(np.array(sol.y.T))
    
    sol_set = torch.tensor(np.array(sol_set), dtype=torch.float32, device=device)
    return sol_set

def ode_residuals_dampened_pendulum(model, coords, steps, duration, t_eval, device, alpha=1.0, beta=1.0, gamma=1.0):
    coords = coords.clone().detach().requires_grad_(True)
    y_pred = model(coords)
    y_gt = compute_forward_integrals_dampened_pendulum(coords[::steps], duration, t_eval, device)
    kan_autograd = compute_jacobian_dampened_pendulum(model, coords)
    kan_ode_grad = compute_ode_dampened_pendulum(y_pred, coords, device) 
    init_kan_grad = y_pred[::steps]
    init_ode_grad = coords[::steps][:, 1:4]

    dyn_loss = torch.mean((kan_autograd - kan_ode_grad)**2)
    integral_loss = torch.mean((y_pred - y_gt)**2)
    init_loss = torch.mean((init_kan_grad - init_ode_grad)**2)
    
    batch_loss = alpha*dyn_loss + beta*integral_loss + gamma*init_loss
    
    return batch_loss

def train_loop_dampened_pendulum(model,  dataset, steps, duration, t_eval, device, iterations=2):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for i in range(iterations):
        optimizer.zero_grad()
        loss = ode_residuals_dampened_pendulum(model, dataset, steps, duration, t_eval, device)
        loss.backward()
        optimizer.step()
        
        if i % 1 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
    
    return model

def system_eq_dis_dampened_pendulum(cond_input, t_eval, time):
    def system_cont(t, x, a, b, c, d):
        dxdt = [-a*x[0] + b*x[1] + c*x[2], 
                (d - a)*x[0] - 2*b*x[1], 
                d*x[0] + b*x[1] - c*x[2]]
        return dxdt
    
    x0, y0, z0, a, b, c, d = cond_input
    sol = sc.integrate.solve_ivp(system_cont, [0, time], (x0, y0, z0), 
                                args=(a, b, c, d), t_eval=t_eval)
    return sol.y.T, sol.t.T

def train_dampened_pendulum():
    device = "cpu"
    s_size = 30
    x_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    y_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    z_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    a_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    b_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    c_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    d_sample = torch.tensor(np.random.uniform(0, 1, size=s_size), dtype=torch.float32, device=device)
    duration = 1
    time_interval = 0.2
    steps = int(duration / time_interval)
    print(f"Number of steps: {steps}")
    t_eval = torch.linspace(0, duration, steps=int(steps))
    dataset = torch.stack([x_sample, y_sample, z_sample, a_sample, b_sample, c_sample, d_sample], dim=1).to(device)
    dataset = dataset.repeat_interleave(steps, dim=0)
    t_eval_set = t_eval.T.repeat(s_size).view(-1, 1).to(device)

    dataset = torch.cat([t_eval_set, dataset], dim=1)
    dataset.requires_grad = True
    print(f"Dataset shape: {dataset.shape}")
    test_size = 5
    test_dataset = dataset[(s_size-test_size)*steps:]
    dataset = dataset[:(s_size-test_size)*steps]
    print(f"Train dataset shape: {dataset.shape}, Test dataset shape: {test_dataset.shape}")

    kan_dampened_pendulum = FastKAN(
        layers_hidden=[8, 24, 3],
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=15,
        use_base_update=False,
        use_layernorm=False
    )
    kan_dampened_pendulum.to(device)

    print(f"Jacobian shape: {compute_jacobian_dampened_pendulum(kan_dampened_pendulum, dataset).shape}")
    print("Training model...")
    kan_dampened_pendulum = train_loop_dampened_pendulum(kan_dampened_pendulum,  dataset, steps, duration, t_eval, device, iterations=10)

    print("\nTest set predictions:")
    kan_verification_grapher.graph_ode_results(test_dataset, kan_dampened_pendulum, steps, t_eval, duration)
    print("\nTrain set predictions:")
    kan_verification_grapher.graph_ode_results(dataset, kan_dampened_pendulum, steps, t_eval, duration)

    print("\nEvaluating test set performance:")
    sol_map = []
    for init_conds in test_dataset[::steps].detach().cpu().numpy():
        sol, _ = system_eq_dis_dampened_pendulum(init_conds[1:], t_eval, duration)
        sol_map.extend(sol)

    pred = kan_dampened_pendulum(test_dataset).detach().cpu().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (test_dataset): {res_score}")

    print("\nEvaluating train set performance:")
    sol_map = []
    for init_conds in dataset[::steps].detach().cpu().numpy():
        sol, _ = system_eq_dis_dampened_pendulum(init_conds[1:], t_eval, duration)
        sol_map.extend(sol)

    pred = kan_dampened_pendulum(dataset).detach().cpu().numpy()
    res_score = r2_score(sol_map, pred)
    print(f"Overall R2-Score (train_dataset): {res_score}")
    return kan_dampened_pendulum

# 3. MNIST
def train_mnist():
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data_mnist", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data_mnist", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = FastKAN([28 * 28, 16, 10], use_base_update=False, use_layernorm=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
    return model

# 4. CIFAR10
def train_cifar10():
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data_cifar10", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.CIFAR10(
        root="./data_cifar10", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = FastKAN([3072, 16, 10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 3072).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 3072).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
    return model