#!/usr/bin/python3
#===========================================
# MNIST Inference GUI
#-------------------------------------------
# Rev.0.1 2019.09.16 Munetomo Maruyama
#-------------------------------------------
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# based on https://github.com/pytorch/examples/tree/master/mnist

from __future__ import print_function
import tkinter as tk
from tkinter import font
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from threading import Thread, Event
import numpy as np
import pdb

#===========================
# Utility
#===========================
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

#==========================
# Neural Net
#==========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #  1x28x28 --> 20x24x24
        x = F.max_pool2d(x, 2, 2) # 20x24x24 --> 20x12x12 
        x = F.relu(self.conv2(x)) # 20x12x12 --> 50x 8x 8
        x = F.max_pool2d(x, 2, 2) # 50x 8x 8 --> 50x 4x 4
        x = x.view(-1, 4*4*50)    # 50x 4x 4 --> 1 x (4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
#============================
# Common Constants
#============================
args = dotdict({
	'batch_size'     : 64,    # Input batch size for training
	'test_batch_size': 1000,  # Input batch size for testing
	'epochs'         : 10,    # Number of epochs to train
	'lr'             : 0.01,  # Learning Rate
	'momentum'       : 0.5,   # SGD Momentum
	'no_cuda'        : False, # Disable CUDA Training
	'seed'           : 1,     # Random Seed
	'log_interval'   : 10,    # How many batches to wait before logging training status
	'save_model'     : True,  # For Saving the current Model
})

#=======================
# Inference GUI
#=======================
class InferGUI(object):
    # Constructor
    def __init__(self, root):
        # Initialize Internal State
        self.root = root
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print(self.device)
        torch.manual_seed(args.seed)
        self.x_prev = -1
        self.y_prev = -1
        self.dot_state = np.zeros((28, 28))

        # Load Model
        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load("mnist_cnn.pt"))
        self.model.eval()
        
        # Generate Canvas
        self.cvs = tk.Canvas(
            width  = 28 * 16,
            height = 28 * 16,
            highlightthickness = 0
        )
        self.cvs.bind("<Button-1>" , self.on_clicked)
        self.cvs.bind("<B1-Motion>", self.on_dragged)
        self.cvs.bind("<ButtonRelease-1>", self.on_released)
        self.cvs.grid(
            row = 0, column = 0,
            rowspan = 16,
            sticky = tk.W + tk.N + tk.S,
            padx = 0, pady = 0
        )
        self.draw_canvas()

        # Generate Message Label as Answer
        ans_fnt = font.Font(family='FreeSans', size=14, weight='bold')
        self.ans_str = tk.StringVar()
        self.ans_lbl = tk.Label(
            textvariable = self.ans_str,
            font = ans_fnt,
            width  = 16,
            borderwidth = 2,
            relief = "groove"
        )
        self.ans_lbl.grid(
            row = 0, column = 1,
            sticky = tk.W + tk.E + tk.N + tk.S,
            padx = 2, pady = 2
        )
        self.ans_str.set(u'MNIST')

        # Generate Message Label as Output
        out_fnt = font.Font(family='Monospace', size=12)
        self.out_str = tk.StringVar()
        self.out_lbl = tk.Label(
            textvariable = self.out_str,
            font = out_fnt,
            anchor = 'w',
            justify = 'left',
            borderwidth = 2,
            relief = "groove"
        )
        self.out_lbl.grid(
            row = 1, column = 1,
            rowspan = 9,
            sticky = tk.W + tk.E + tk.N + tk.S,
            padx = 2, pady = 2
        )
        self.out_str.set(u'')

        # Generate Clear Button
        self.btn_clear = tk.Button(
            text='Clear', state = tk.NORMAL,
            command = self.on_button_clear
        )
        self.btn_clear.grid(
            row = 10, column = 1,
            rowspan = 3,
            sticky = tk.W + tk.E + tk.N + tk.S,
            padx = 2, pady = 2
        )
         
        # Generate Infer Button
        self.btn_infer = tk.Button(
            text='Infer', state = tk.NORMAL,
            command = self.on_button_infer
        )
        self.btn_infer.grid(
            row = 13, column = 1,
            rowspan = 3,
            sticky = tk.W + tk.E + tk.N + tk.S,
            padx = 2, pady = 2
        )
                
    # Window Close
    def on_closing(self):
        self.root.destroy()
        sys.exit()
        
    # Draw Canvas
    def draw_canvas(self):
        for y in range (28):
            for x in range (28):
                dot = self.dot_state[y][x]
                self.draw_dot(x, y, dot)

    # Draw Dot
    def draw_dot(self, x, y, dot):
        color_fill = "#FFFFFF" if dot == 1 else "#000000" 
        self.cvs.create_rectangle(
            x*16, y*16, (x+1)*16, (y+1)*16,
            outline = "#C0C0C0",
            fill = color_fill
        )
        x = 27 if x > 27 else x
        y = 27 if y > 27 else y
        self.dot_state[y][x] = dot
        
    
    # Draw Line
    def draw_line(self, x0, y0, x1, y1, dot):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.draw_dot(x0, y0, dot)
            if (x0 == x1 and y0 == y1):
                break
            e2 = 2 * err
            if (e2 > -dy):
                err = err -dy
                x0 = x0 + sx
            if (e2 < dx):
                err = err + dx
                y0 = y0 + sy
                    
    # On Clicked on Canvas
    def on_clicked(self, event):
        x = int(event.x / 16)
        y = int(event.y / 16)
        self.dot = (self.dot_state[y][x] + 1) % 2
        self.draw_line(x, y, x, y, self.dot)
        self.x_prev = x
        self.y_prev = y

    # On Dragged on Canvas
    def on_dragged(self, event):
        x = int(event.x / 16)
        y = int(event.y / 16)
        self.draw_line(self.x_prev, self.y_prev, x, y, self.dot)
        self.x_prev = x
        self.y_prev = y

    # On Released on Canvas
    def on_released(self, event):
        x = int(event.x / 16)
        y = int(event.y / 16)
        self.draw_line(self.x_prev, self.y_prev, x, y, self.dot)
        self.x_prev = -1
        self.y_prev = -1

    # On Clicked Button Clear
    def on_button_clear(self):
        for y in range(28):
            for x in range(28):
                self.dot_state[y][x] = 0
        self.draw_canvas()
        self.ans_str.set(u'MNIST')
        self.out_str.set(u'')
        
    # On Clicked Button Infer
    def on_button_infer(self):
        # Prepare Input Image for CNN
        image = torch.from_numpy(self.dot_state)
        image = image * 255
        image = image[None, None] # add dimention of Batch and Channel
        image = image.type('torch.FloatTensor')
        image = image.to(self.device)
        
        # Input the Image to CNN to infer
        output = self.model(image)
        
        if (self.device == torch.device("cuda")):
            output = output.to(torch.device("cpu"))
            
        # Display Result
        output_list = output.detach().numpy()
        output_list = np.reshape(output_list, 10)
        output_max = np.amax(output_list)
        output_min = np.amin(output_list)
        output_range = output_max - output_min
        for i in range(10):
            output_list[i] = (output_list[i] - output_min) / output_range
            output_list[i] = int(output_list[i] * 100)
        answer = np.argmax(output_list)
        self.ans_str.set(f'Inferred as {answer}')
        result = ""
        for i in range(10):
            result = result + f'Out[{i}] = {output_list[i]:3.0f}\n'
        self.out_str.set(result)           

#=======================
# Main Entry
#=======================
if __name__ == '__main__':
    #main()
    
    # Execute GUI
    root = tk.Tk()
    root.title(u"MNIST")
    app = InferGUI(root) # do not use pack() which has grid inside
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

#===========================================
# End of Program
#===========================================
