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

# Import Jacobi models
from KAN_nn.jacobi_a2b2 import JacobiPINN2
from KAN_nn.jacobi_a1b1 import JacobiPINN1

# Check CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)


def kovasznay_solution(x, y, Re=40):
    """Compute analytical solution of Kovasznay flow"""
    nu = 1/Re
    lamb = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2)
    
    u = 1 - np.exp(lamb*x)*np.cos(2*np.pi*y)
    v = (lamb/(2*np.pi))*np.exp(lamb*x)*np.sin(2*np.pi*y)
    p = 0.5*(1 - np.exp(2*lamb*x))
    
    return u, v, p


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
        return u, v, p
    
    def net_f(self, x):
        """Compute PDE residuals - Steady-state NS equations"""
        u, v, p = self.net_uvp(x)
        
        # Compute partial derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0][:, 1:2]
        
        # Steady-state NS equations
        f_u = (u*u_x + v*u_y) + p_x - self.nu*(u_xx + u_yy)
        f_v = (u*v_x + v*v_y) + p_y - self.nu*(v_xx + v_yy)
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

    def train_lbfgs(self, X_test, UV_test, epoch_ADAM=0):
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
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.dnn.eval()
        with torch.no_grad():
            return self.net_uvp(X)

    def compute_error(self, X_test, UV_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        UV_test = torch.tensor(UV_test, dtype=torch.float32).to(device)
        
        u_pred, v_pred, _ = self.predict(X_test)
        UV_pred = torch.cat([u_pred, v_pred], dim=1)
        
        error = torch.norm(UV_test - UV_pred, 2) / torch.norm(UV_test, 2)
        return error.item()


def generate_training_data(n_b=101, n_f=2601):
    """Generate training data"""
    # Define computational domain
    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 1.5
    
    # Generate boundary points
    x_b = np.linspace(x_min, x_max, n_b)
    y_b = np.linspace(y_min, y_max, n_b)
    
    # Boundary point coordinates
    x_left = np.ones(n_b) * x_min
    x_right = np.ones(n_b) * x_max
    x_bottom = x_b
    x_top = x_b
    
    y_left = y_b
    y_right = y_b
    y_bottom = np.ones(n_b) * y_min
    y_top = np.ones(n_b) * y_max
    
    # Merge all boundary points
    X_b = np.vstack([
        np.stack([x_left, y_left], axis=1),
        np.stack([x_right, y_right], axis=1),
        np.stack([x_bottom, y_bottom], axis=1),
        np.stack([x_top, y_top], axis=1)
    ])
    
    # Compute analytical solution at boundary points
    u_b, v_b, _ = kovasznay_solution(X_b[:,0:1], X_b[:,1:2])
    UV_b = np.hstack([u_b, v_b])
    
    # Generate interior collocation points
    lb = np.array([x_min, y_min])
    ub = np.array([x_max, y_max])
    X_f = lb + (ub - lb) * lhs(2, n_f)
    
    return X_b, UV_b, X_f


def generate_test_data(nx=101, ny=101):
    """Generate test data"""
    x = np.linspace(-0.5, 1.0, nx)
    y = np.linspace(-0.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)
    
    X_test = np.stack([X.flatten(), Y.flatten()], axis=1)
    u_test, v_test, p_test = kovasznay_solution(X_test[:,0:1], X_test[:,1:2])
    UV_test = np.hstack([u_test, v_test])
    
    return X_test, UV_test, p_test, X, Y


def plot_results(X, Y, u_true, v_true, p_true, u_pred, v_pred, p_pred, title, equation, size, hidden_layer, degree):
    """Plot comparison results"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Velocity field u
    im1 = axes[0,0].contourf(X, Y, u_true.reshape(X.shape))
    axes[0,0].set_title('True u')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].contourf(X, Y, u_pred.reshape(X.shape))
    axes[0,1].set_title('Predicted u')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Velocity field v
    im3 = axes[1,0].contourf(X, Y, v_true.reshape(X.shape))
    axes[1,0].set_title('True v')
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].contourf(X, Y, v_pred.reshape(X.shape))
    axes[1,1].set_title('Predicted v')
    plt.colorbar(im4, ax=axes[1,1])
    
    # Pressure field p
    im5 = axes[2,0].contourf(X, Y, p_true.reshape(X.shape))
    axes[2,0].set_title('True p')
    plt.colorbar(im5, ax=axes[2,0])
    
    im6 = axes[2,1].contourf(X, Y, p_pred.reshape(X.shape))
    axes[2,1].set_title('Predicted p')
    plt.colorbar(im6, ax=axes[2,1])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{equation}_{title}_size{size}_hidden{hidden_layer}_deg{degree}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    # Parameter settings
    equation = 'Kovasznay_Jacobi'
    nu = 1/40  # Viscosity coefficient
    degree = 4  # Basis function degree
    size = 30   # Number of neurons per layer
    hidden_layer = 4  # Number of hidden layers
    layers = [2] + [size] * hidden_layer + [3]  # Network structure
    epoch_ADAM = 2000
    epoch_LBFGS = 20000

    # Generate training and test data
    X_b, UV_b, X_f = generate_training_data()
    X_test, UV_test, p_test, X_mesh, Y_mesh = generate_test_data()

    # Define Jacobi model dictionary
    models_dict = OrderedDict({
        'Jacobi_a2b2': JacobiPINN2,
        'Jacobi_a1b1': JacobiPINN1,
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

    print(f"Network architecture: {layers}")

    # Train all Jacobi models
    for model_name, model_class in models_dict.items():
        print(f"\nTraining {model_name} model...")
        
        start_time = time.time()
        model = PhysicsInformedNN(X_b, UV_b, X_f, layers, nu, model_class, degree, epoch_LBFGS)
        print(f"{model_name} total parameters: {model.total_params}")
        
        # Training
        # model.train_adam(epoch_ADAM, X_test, UV_test)
        model.train_lbfgs(X_test, UV_test)
        
        # Record results
        training_time = time.time() - start_time
        results['training_time'][model_name] = training_time
        
        # Prediction and error calculation
        u_pred, v_pred, p_pred = model.predict(X_test)
        error = model.compute_error(X_test, UV_test)
        results['l2_error'][model_name] = error
        
        # Store prediction results
        results['predictions'][model_name] = (u_pred, v_pred, p_pred)
        results['loss_history'][model_name] = model.loss_history
        results['error_history'][model_name] = model.error_history
        results['iter_history'][model_name] = model.iter_history
        
        # Plot results
        u_true, v_true, p_true = kovasznay_solution(X_test[:,0:1], X_test[:,1:2])
        plot_results(X_mesh, Y_mesh, 
                    u_true, v_true, p_true,
                    u_pred.cpu().numpy(), v_pred.cpu().numpy(), p_pred.cpu().numpy(),
                    model_name, equation, size, hidden_layer, degree)

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Loss history
    plt.subplot(121)
    for model_name in models_dict.keys():
        plt.semilogy(results['iter_history'][model_name], 
                    results['loss_history'][model_name],
                    label=model_name)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Error history
    plt.subplot(122)
    for model_name in models_dict.keys():
        plt.semilogy(results['iter_history'][model_name],
                    results['error_history'][model_name],
                    label=model_name)
    plt.xlabel('Iterations')
    plt.ylabel('Relative L2 Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{equation}_training_history_size{size}_hidden{hidden_layer}_deg{degree}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot comparison
    from matplotlib import font_manager
    font_properties = font_manager.FontProperties(weight='bold')
    
    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

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
    plt.savefig(f'{equation}_time_error_comparison_size{size}_hidden{hidden_layer}_deg{degree}.png', 
                dpi=300)
    plt.show()

    # Print final results
    print("\nFinal Results:")
    print("-" * 60)
    print(f"{'Model':<20} {'Training Time(s)':<20} {'L2 Error':<15}")
    print("-" * 60)
    for model_name in models_dict.keys():
        print(f"{model_name:<20} {results['training_time'][model_name]:>20.2f} {results['l2_error'][model_name]:>15.3e}")

    # Save results
    u_true, v_true, p_true = kovasznay_solution(X_test[:,0:1], X_test[:,1:2])
    
    scipy.io.savemat(f'{equation}_lbfgs_results_hidden{hidden_layer}_size{size}_deg{degree}.mat', {
        'X_mesh': X_mesh,
        'Y_mesh': Y_mesh,
        'u_true': u_true,
        'v_true': v_true,
        'p_true': p_true,
        'predictions': {name: {
            'u': results['predictions'][name][0].cpu().numpy(),
            'v': results['predictions'][name][1].cpu().numpy(),
            'p': results['predictions'][name][2].cpu().numpy()
        } for name in results['predictions'].keys()},
        'training_time': results['training_time'],
        'l2_error': results['l2_error'],
        'loss_history': results['loss_history'],
        'error_history': results['error_history'],
        'iter_history': results['iter_history']
    })
    
    print(f"\nResults saved to {equation}_lbfgs_results_hidden{hidden_layer}_size{size}_deg{degree}.mat")


if __name__ == '__main__':
    main()


