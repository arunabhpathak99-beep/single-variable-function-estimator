import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class FunctionApproximator(nn.Module):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
        
        self.hidden = nn.Linear(1, 100)
        
        self.relu = nn.ReLU()
        
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Function Approximator")
        
        self.x_entries = []
        self.y_entries = []
        
        
        tk.Label(root, text="Input (X)", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=10, pady=5)
        tk.Label(root, text="Output (Y)", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=10, pady=5)
        
        
        for i in range(10):
            x_entry = tk.Entry(root, width=15)
            x_entry.grid(row=i+1, column=0, padx=10, pady=2)
            self.x_entries.append(x_entry)
            
            y_entry = tk.Entry(root, width=15)
            y_entry.grid(row=i+1, column=1, padx=10, pady=2)
            self.y_entries.append(y_entry)
            
        
        self.train_btn = tk.Button(root, text="Train & Plot", command=self.process_data, bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'))
        self.train_btn.grid(row=11, column=0, columnspan=2, pady=15)

    def process_data(self):
        x_data = []
        y_data = []
        
        
        try:
            for i in range(10):
                x_val = float(self.x_entries[i].get())
                y_val = float(self.y_entries[i].get())
                x_data.append(x_val)
                y_data.append(y_val)
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all 10 rows are filled with valid numbers.")
            return

        
        self.train_btn.config(state=tk.DISABLED, text="Training...")
        self.root.update()

       
        self.train_and_plot(np.array(x_data), np.array(y_data))
        
        
        self.train_btn.config(state=tk.NORMAL, text="Train & Plot")

    def train_and_plot(self, x_np, y_np):
        
        x_mean, x_std = x_np.mean(), x_np.std()
        y_mean, y_std = y_np.mean(), y_np.std()
        
        
        x_std = x_std if x_std != 0 else 1.0
        y_std = y_std if y_std != 0 else 1.0

        x_norm = (x_np - x_mean) / x_std
        y_norm = (y_np - y_mean) / y_std

        
        X_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1)
        Y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)

        
        model = FunctionApproximator()
        criterion = nn.MSELoss()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        
        epochs = 5000
        for epoch in range(epochs):
           
            predictions = model(X_tensor)
            
           
            loss = criterion(predictions, Y_tensor)
            
            
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      

        
        model.eval() 
        
        
        x_plot = np.linspace(x_np.min() - 1, x_np.max() + 1, 200)
        
        
        x_plot_norm = (x_plot - x_mean) / x_std
        x_plot_tensor = torch.tensor(x_plot_norm, dtype=torch.float32).view(-1, 1)
        
        
        with torch.no_grad():
            y_plot_norm_pred = model(x_plot_tensor).numpy()
            
        
        y_plot_pred = (y_plot_norm_pred * y_std) + y_mean

        
        plt.figure(figsize=(8, 5))
        plt.scatter(x_np, y_np, color='red', label='Training Data (10 points)', zorder=5)
        plt.plot(x_plot, y_plot_pred, color='blue', label='NN Approximation Curve', zorder=4)
        plt.title('Neural Network Function Approximation')
        plt.xlabel('Input (X)')
        plt.ylabel('Output (Y)')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
