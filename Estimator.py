import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def train_and_plot():
    try:
    
        x_data = []
        y_data = []
        for i in range(10):
            x_val = float(x_entries[i].get())
            y_val = float(y_entries[i].get())
            x_data.append(x_val)
            y_data.append(y_val)
        
        
        X = np.array(x_data).reshape(-1, 1)
        y = np.array(y_data)
        
        
        nn = MLPRegressor(
            hidden_layer_sizes=(100,), 
            activation='relu', 
            solver='lbfgs', 
            max_iter=5000, 
            random_state=42
        )
        nn.fit(X, y)
        
        x_min, x_max = min(x_data), max(x_data)
        margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
        X_plot = np.linspace(x_min - margin, x_max + margin, 500).reshape(-1, 1)
        y_plot = nn.predict(X_plot)
        
        
        plt.figure("Neural Network Function Approximation")
        plt.scatter(x_data, y_data, color='red', zorder=5, label='Training Data (10 points)')
        plt.plot(X_plot, y_plot, color='blue', linewidth=2, label='NN Approximation Curve')
        plt.title('Neural Network Approximating a Single Variable Function')
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError:
       
        messagebox.showerror("Input Error", "Please ensure all 20 fields contain valid numbers.")


root = tk.Tk()
root.title("NN Function Approximator")
root.geometry("300x400")


tk.Label(root, text="X Values", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=10)
tk.Label(root, text="Y Values", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=10, pady=10)

x_entries = []
y_entries = []


for i in range(10):
    
    xe = tk.Entry(root, width=12)
    xe.grid(row=i+1, column=0, padx=10, pady=3)
    x_entries.append(xe)
    
   
    ye = tk.Entry(root, width=12)
    ye.grid(row=i+1, column=1, padx=10, pady=3)
    y_entries.append(ye)


train_btn = tk.Button(root, text="Train Network & Plot", command=train_and_plot, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
train_btn.grid(row=11, column=0, columnspan=2, pady=20)

    
root.mainloop()
