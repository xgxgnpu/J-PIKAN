# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import scipy.io
import time
from collections import OrderedDict

from scipy.interpolate import griddata

# Import Jacobi models
from KAN_nn.jacobi_a2b2 import JacobiPINN2
from KAN_nn.jacobi_a1b1 import JacobiPINN1

# Check CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)


class PhysicsInformedNN:
    def __init__(self, X_data, UV_data, X_f, layers, nu, nn_model, degree, epoch_LBFGS):
        self.epoch_LBFGS = epoch_LBFGS
        
        # Record training history
        self.loss_history = []
        self.error_history = []
        self.iter_history = []
        
        # Data processing
        self.X_data = torch.tensor(X_data, requires_grad=True).float().to(device)
        self.UV_data = torch.tensor(UV_data).float().to(device)
        self.X_f = torch.tensor(X_f, requires_grad=True).float().to(device)
        self.nu = nu
        
        # Initialize model
        self.dnn = nn_model(layers, degree).to(device)
        self.total_params = sum(p.numel() for p in self.dnn.parameters())
        
        # Optimizers
        self.optimizer_adam = optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer_lbfgs = optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0,
            max_iter=self.epoch_LBFGS,
            max_eval=self.epoch_LBFGS,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1e-7,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0

    def net_uvp(self, x):
        
        uvp = self.dnn(x)
        
        u = uvp[:,0:1]
        v = uvp[:,1:2]
        p = uvp[:,2:3]
        
        return u,v,p
    
    def net_f(self, x):
        """Compute PDE residuals"""
        u, v, p = self.net_uvp(x)
        
        # Compute partial derivatives
        u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2:3]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
        
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        v_t = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 2:3]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1:2]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0][:, 1:2]
        
        # Navier-Stokes equations
        f_u = u_t + (u*u_x + v*u_y) + p_x - self.nu*(u_xx + u_yy)
        f_v = v_t + (u*v_x + v*v_y) + p_y - self.nu*(v_xx + v_yy)
        f_p = u_x + v_y
        
        return f_u, f_v, f_p

    def loss_func(self):
        """Compute total loss"""
        # Predictions
        u_pred, v_pred, _ = self.net_uvp(self.X_data)
        UV_pred = torch.cat([u_pred, v_pred], dim=1)
        
        # Data loss
        loss_data = torch.mean((UV_pred - self.UV_data)**2)
        
        # PDE residual loss
        f_u, f_v, f_p = self.net_f(self.X_f)
        loss_pde = torch.mean(f_u**2 + f_v**2 + f_p**2)
        
        return loss_data + loss_pde
    
    
    def train_adam(self, epochs, X_test, UV_test):
        print("=====================Adam started===========================")
        self.dnn.train()
        for epoch in range(epochs):
            self.optimizer_adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_adam.step()

            if epoch % 100 == 0:
                error = self.compute_error(X_test, UV_test)
                print(f"Epoch {epoch}, Loss: {loss.item():.5e}, Error: {error:.5e}")
                self.loss_history.append(loss.item())
                self.error_history.append(error)
                self.iter_history.append(epoch)

    def train_lbfgs(self, X_test, UV_test, epoch_ADAM):
        print("=====================L-BFGS started===========================")
        self.dnn.train()
        
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss = self.loss_func()
            loss.backward()
            if self.iter % 100 == 0:
                error = self.compute_error(X_test, UV_test)
                print(f"Iter {self.iter}, Loss: {loss.item():.5e}, Error: {error:.5e}")
                self.loss_history.append(loss.item())
                self.error_history.append(error)
                self.iter_history.append(self.iter + epoch_ADAM)
            self.iter += 1
            return loss
            
        self.optimizer_lbfgs.step(closure)

    def predict(self, X):
        """Model prediction"""
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.dnn.eval()
        with torch.no_grad():
            return self.net_uvp(X)

    def compute_error(self, X_test, UV_test):
        """Compute prediction error"""
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        UV_test = torch.tensor(UV_test, dtype=torch.float32).to(device)
        
        u_pred, v_pred, _ = self.predict(X_test)
        UV_pred = torch.cat([u_pred, v_pred], dim=1)
        
        error = torch.norm(UV_test - UV_pred, 2) / torch.norm(UV_test, 2)
        return error.item()


def load_and_process_data(data_path, n_f=10000):
    """
    Load and process Navier-Stokes equation data
    """
    # Load data
    data = scipy.io.loadmat(data_path)
    
    # Get grid data
    x = data['x'].flatten()
    y = data['y'].flatten()
    t = data['t'].flatten()
    
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    u = data['u']
    v = data['v']
    p = data['p']
    
    nx, ny, nt = u.shape
    
    # Initial condition data
    idx_t0 = 0
    X_ic = np.hstack((
        X[:, :, idx_t0].flatten()[:, None],
        Y[:, :, idx_t0].flatten()[:, None],
        T[:, :, idx_t0].flatten()[:, None]
    ))
    UV_ic = np.hstack((
        u[:, :, idx_t0].flatten()[:, None],
        v[:, :, idx_t0].flatten()[:, None]
    ))
    
    # Boundary condition data
    # Left boundary
    X_bc_left = np.hstack((
        X[0, :, :].flatten()[:, None],
        Y[0, :, :].flatten()[:, None],
        T[0, :, :].flatten()[:, None]
    ))
    UV_bc_left = np.hstack((
        u[0, :, :].flatten()[:, None],
        v[0, :, :].flatten()[:, None]
    ))
    
    # Right boundary
    X_bc_right = np.hstack((
        X[-1, :, :].flatten()[:, None],
        Y[-1, :, :].flatten()[:, None],
        T[-1, :, :].flatten()[:, None]
    ))
    UV_bc_right = np.hstack((
        u[-1, :, :].flatten()[:, None],
        v[-1, :, :].flatten()[:, None]
    ))
    
    # Bottom boundary
    X_bc_bottom = np.hstack((
        X[:, 0, :].flatten()[:, None],
        Y[:, 0, :].flatten()[:, None],
        T[:, 0, :].flatten()[:, None]
    ))
    UV_bc_bottom = np.hstack((
        u[:, 0, :].flatten()[:, None],
        v[:, 0, :].flatten()[:, None]
    ))
    
    # Top boundary
    X_bc_top = np.hstack((
        X[:, -1, :].flatten()[:, None],
        Y[:, -1, :].flatten()[:, None],
        T[:, -1, :].flatten()[:, None]
    ))
    UV_bc_top = np.hstack((
        u[:, -1, :].flatten()[:, None],
        v[:, -1, :].flatten()[:, None]
    ))
    
    # Merge all boundary and initial condition data
    X_data = np.vstack([X_ic, X_bc_left, X_bc_right, X_bc_bottom, X_bc_top])
    print("X_data", X_data.shape)
    UV_data = np.vstack([UV_ic, UV_bc_left, UV_bc_right, UV_bc_bottom, UV_bc_top])
    print("UV_data", UV_data.shape)
    
    # Sample interior points
    lb = np.array([x.min(), y.min(), t.min()])
    ub = np.array([x.max(), y.max(), t.max()])
    X_f = lb + (ub - lb) * lhs(3, n_f)
    
    # Prepare test data
    X_test = np.hstack((
        X.flatten()[:, None],
        Y.flatten()[:, None],
        T.flatten()[:, None]
    ))
    UV_test = np.hstack((
        u.flatten()[:, None],
        v.flatten()[:, None]
    ))
    P_test = p.flatten()[:, None]
    
    return X_data, UV_data, X_f, X_test, UV_test, P_test, lb, ub


def main():
    # Set parameters
    equation = 'NS_degree4_jacobi'
    data_path = 'data/NS_Taylor_Green_mu_0.01.mat'
    nu = 0.01  # Viscosity coefficient
    n_f = 2000  # Number of collocation points

    # Load and process data
    print("Loading and processing data...")
    X_data, UV_data, X_f, X_test, UV_test, P_test, lb, ub = load_and_process_data(data_path, n_f)
    print(f"Data loading completed!")
    print(f"Training data points: {len(X_data)}")
    print(f"Collocation points: {len(X_f)}")
    print(f"Test points: {len(X_test)}")

    # Define Jacobi model dictionary
    models_dict = OrderedDict({
        'Jacobi(alpha=2,beta=2)': JacobiPINN2,
        'Jacobi(alpha=1,beta=1)': JacobiPINN1,
    })

    # Store results
    results = {
        'training_time': {},
        'l2_error': {},
        'predictions': {},
        'loss_history': {},
        'error_history': {},
        'iter_history': {}
    }

    # Training parameters
    epoch_ADAM = 2000
    epoch_LBFGS = 20000

    degree = 4
    size = 20
    hidden_layer = 4
    layers = [3] + [size] * hidden_layer + [3]
    print(f"Network architecture: {layers}")

    # Train all Jacobi models
    for model_name, model_class in models_dict.items():
        print(f"\nTraining {model_name} model...")

        start_time = time.time()
        model = PhysicsInformedNN(X_data, UV_data, X_f, layers, nu, 
                                 model_class, degree, epoch_LBFGS)

        print(f"{model_name} total parameters: {model.total_params}")

        # Training
        model.train_adam(epoch_ADAM, X_test, UV_test)
        model.train_lbfgs(X_test, UV_test, epoch_ADAM)

        # Record results
        training_time = time.time() - start_time
        results['training_time'][model_name] = training_time

        # Prediction and error calculation
        UV_pred = model.predict(X_test)
        error = model.compute_error(X_test, UV_test)
        results['l2_error'][model_name] = error

        # Store other results
        results['predictions'][model_name] = UV_pred
        results['loss_history'][model_name] = model.loss_history
        results['error_history'][model_name] = model.error_history
        results['iter_history'][model_name] = model.iter_history

    # Load data for visualization
    data = scipy.io.loadmat(data_path)
    x = data['x'].flatten()
    y = data['y'].flatten()
    t = data['t'].flatten()
    X, Y = np.meshgrid(x, y)
    u = data['u']
    v = data['v']
    p = data['p']

    # Save prediction data
    file_name = f'{equation}_model_predictions_size_{size}_hidden_layer_{hidden_layer}_degree_{degree}.npz'
    np.savez(file_name,
             predictions=results['predictions'],
             l2_error=results['l2_error'])

    # Prediction visualization
    t_idx = 50
    t_val = t[t_idx]

    # Get ground truth at this time
    u_true = u[:, :, t_idx]
    v_true = v[:, :, t_idx]
    p_true = p[:, :, t_idx]

    # Create grid points
    x_grid = np.linspace(x.min(), x.max(), 256)
    y_grid = np.linspace(y.min(), y.max(), 256)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    # Create figure for each model with u, v, p subplots
    for model_name, pred_tuple in results['predictions'].items():
        fig = plt.figure(figsize=(20, 6))
        
        # Extract and reshape predictions
        u_pred = pred_tuple[0].cpu().numpy()
        v_pred = pred_tuple[1].cpu().numpy()
        p_pred = pred_tuple[2].cpu().numpy()
        
        # Reshape to 3D array [nx, ny, nt]
        nx = ny = int(np.sqrt(u_pred.shape[0] / len(t)))
        u_pred_3d = u_pred.reshape(nx, ny, len(t))
        v_pred_3d = v_pred.reshape(nx, ny, len(t))
        p_pred_3d = p_pred.reshape(nx, ny, len(t))
        
        # Extract predictions at specific time
        u_pred_t = u_pred_3d[:, :, t_idx]
        v_pred_t = v_pred_3d[:, :, t_idx]
        p_pred_t = p_pred_3d[:, :, t_idx]
        
        # Compute relative errors
        error_u = np.linalg.norm(u_true - u_pred_t) / np.linalg.norm(u_true)
        error_v = np.linalg.norm(v_true - v_pred_t) / np.linalg.norm(v_true)
        error_p = np.linalg.norm(p_true - p_pred_t) / np.linalg.norm(p_true)
        
        # u component
        plt.subplot(131)
        c = plt.pcolor(X_mesh, Y_mesh, u_pred_t, cmap='rainbow')
        plt.colorbar(c)
        plt.title(f'u - Error: {error_u:.3e}')
        plt.xlabel('x', fontsize=12, fontweight='bold')
        plt.ylabel('y', fontsize=12, fontweight='bold')
        plt.grid(True)
        
        # v component
        plt.subplot(132)
        c = plt.pcolor(X_mesh, Y_mesh, v_pred_t, cmap='rainbow')
        plt.colorbar(c)
        plt.title(f'v - Error: {error_v:.3e}')
        plt.xlabel('x', fontsize=12, fontweight='bold')
        plt.ylabel('y', fontsize=12, fontweight='bold')
        plt.grid(True)
        
        # p component
        plt.subplot(133)
        c = plt.pcolor(X_mesh, Y_mesh, p_pred_t, cmap='rainbow')
        plt.colorbar(c)
        plt.title(f'p - Error: {error_p:.3e}')
        plt.xlabel('x', fontsize=12, fontweight='bold')
        plt.ylabel('y', fontsize=12, fontweight='bold')
        plt.grid(True)
        
        plt.suptitle(f'Model: {model_name} at t = {t_val:.2f}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{equation}_{model_name}_prediction_t{t_idx}_size_{size}_hidden_layer_{hidden_layer}_degree_{degree}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()

    # Plot ground truth
    fig = plt.figure(figsize=(20, 6))

    plt.subplot(131)
    c = plt.pcolor(X_mesh, Y_mesh, u_true, cmap='rainbow')
    plt.colorbar(c)
    plt.title('u')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.subplot(132)
    c = plt.pcolor(X_mesh, Y_mesh, v_true, cmap='rainbow')
    plt.colorbar(c)
    plt.title('v')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.subplot(133)
    c = plt.pcolor(X_mesh, Y_mesh, p_true, cmap='rainbow')
    plt.colorbar(c)
    plt.title('p')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.suptitle(f'Ground Truth at t = {t_val:.2f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{equation}_ground_truth_t{t_idx}_size_{size}_hidden_layer_{hidden_layer}_degree_{degree}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

    # Save error data to text file
    output_file_name = f"{equation}_model_comparison_results_t{t_idx}_size_{size}_hidden_layer_{hidden_layer}_degree_{degree}.txt"
    with open(output_file_name, 'w') as f:
        f.write(f"\nResults at t = {t_val:.2f}:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Model Name':<25} {'Training Time(s)':<18} {'Error u':<12} {'Error v':<12} {'Error p':<12} {'Avg Error':<12}\n")
        f.write("-" * 90 + "\n")
        for model_name in models_dict.keys():
            pred_tuple = results['predictions'][model_name]
            u_pred = pred_tuple[0].cpu().numpy().reshape(nx, ny, len(t))[:, :, t_idx]
            v_pred = pred_tuple[1].cpu().numpy().reshape(nx, ny, len(t))[:, :, t_idx]
            p_pred = pred_tuple[2].cpu().numpy().reshape(nx, ny, len(t))[:, :, t_idx]
            
            error_u = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
            error_v = np.linalg.norm(v_true - v_pred) / np.linalg.norm(v_true)
            error_p = np.linalg.norm(p_true - p_pred) / np.linalg.norm(p_true)
            total_error = (error_u + error_v + error_p) / 3
            
            f.write(f"{model_name:<25} {results['training_time'][model_name]:<18.2f} "
                    f"{error_u:<12.3e} {error_v:<12.3e} {error_p:<12.3e} {total_error:<12.3e}\n")

    # Set bold font
    from matplotlib import font_manager
    font_properties = font_manager.FontProperties(weight='bold')

    # Save training history
    file_name = f'{equation}_model_training_results_size_{size}_hidden_layer_{hidden_layer}.npz'
    np.savez(file_name,
             iter_history=results['iter_history'],
             loss_history=results['loss_history'],
             error_history=results['error_history'])

    # Plot training history
    plt.figure(figsize=(15, 6))
    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
    font_properties = font_manager.FontProperties(weight='bold')

    # Loss history
    plt.subplot(121)
    for i, model_name in enumerate(models_dict.keys()):
        plt.semilogy(results['iter_history'][model_name], 
                     results['loss_history'][model_name],
                     label=model_name,
                     linestyle=line_styles[i % len(line_styles)],
                     color=colors[i % len(colors)],
                     linewidth=2)
    plt.xlabel('Iterations', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss History', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, prop=font_properties)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)

    # Error history
    plt.subplot(122)
    for i, model_name in enumerate(models_dict.keys()):
        plt.semilogy(results['iter_history'][model_name],
                     results['error_history'][model_name],
                     label=model_name,
                     linestyle=line_styles[i % len(line_styles)],
                     color=colors[i % len(colors)],
                     linewidth=2)
    plt.xlabel('Iterations', fontsize=14, fontweight='bold')
    plt.ylabel('Relative L2 Error', fontsize=14, fontweight='bold')
    plt.title('Error History', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, prop=font_properties)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)

    plt.tight_layout()
    plt.savefig(f'{equation}_training_history_size_{size}_hidden_layer_{hidden_layer}.png', 
                dpi=300)
    plt.show()

    # Training time and error comparison
    file_name = f'{equation}_model_training_time_error_comparison_size_{size}_hidden_layer_{hidden_layer}.npz'
    np.savez(file_name,
             training_time=results['training_time'],
             l2_error=results['l2_error'])

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.bar(results['training_time'].keys(), 
            results['training_time'].values(), 
            color=colors[:len(results['training_time'])])
    plt.xticks(rotation=45)
    plt.title('Model Training Time Comparison')
    plt.ylabel('Training Time (s)')

    plt.subplot(122)
    plt.bar(results['l2_error'].keys(), 
            results['l2_error'].values(), 
            color=colors[:len(results['l2_error'])])
    plt.xticks(rotation=45)
    plt.title('Model L2 Error Comparison')
    plt.ylabel('Relative L2 Error')

    plt.tight_layout()
    plt.savefig(f'{equation}_time_error_comparison_size_{size}_hidden_layer_{hidden_layer}.png', 
                dpi=300)
    plt.show()

    # Save detailed results to text file
    output_file_name = f"{equation}_model_comparison_results_size_{size}_hidden_layer_{hidden_layer}.txt"
    with open(output_file_name, 'w') as f:
        f.write("\nDetailed Results Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model Name':<25} {'Training Time(s)':<18} {'L2 Error':<15}\n")
        f.write("-" * 60 + "\n")
        for model_name in models_dict.keys():
            f.write(f"{model_name:<25} {results['training_time'][model_name]:<18.2f} {results['l2_error'][model_name]:<15.3e}\n")

    # Print results
    print("\nDetailed Results Statistics:")
    print("-" * 60)
    print(f"{'Model Name':<25} {'Training Time(s)':<18} {'L2 Error':<15}")
    print("-" * 60)
    for model_name in models_dict.keys():
        print(f"{model_name:<25} {results['training_time'][model_name]:<18.2f} {results['l2_error'][model_name]:<15.3e}")


if __name__ == '__main__':
    main()
