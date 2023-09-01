# jetson_nano

## Sample Programs for Jetson Nano 
These are introduced in articles of "Transistor-Gijyutu, Sept/2019" by CQ publishing. <br>

 (1) GPIO : GPIO Controls written in CUDA C. <br>
 (2) GPUADD : Matrix Addition written in CUDA C. <br>
 (3) GPUDEV : Device Recognition written in CUDA C. <br>
 (4) GPUFP16 : FP16 Benchmark written in CUDA C. <br>
 (5) GPUFP32 : FP32 Benchmark written in CUDA C. <br>
 (6) GPUFP64 : FP64 Benchmark written in CUDA C. <br>
 (7) GPUMUL : Matrix Multiplication written in CUDA C. <br>
 (8) GPUPI : PI Caclulation written in CUDA C. <br>

## Sample Programs for AI <br>
Preparation : Install Python and PyTorch. <br>

 (1) MNIST : A Sample Program of Machine Learning <br>
    Handwritten Digit Recognition in Python using PyTorch and TkInter. <br><br>
    - mainInfer.py : GUI Application for MNIST Inference (please run this first as a demo.) <br>
    - mainTrain.py : MNIST Training and Test Program <br>
    
 (2) ReversiAZ : A Sample Program of Reinforcement Learning <br>
    Reversi Game based on Alpha-Zero algorithm. <br><br>
  - mainPlay : GUI Application for Reversi Game (please run this first as a demo.) <br>
          Edit following lines to change train data. Board size is 6x6 or 8x8. Please select corresponding train data. 
```
args = dotdict({ <br>
    'boardSize' : 8, <br>
    'model_dir' : './model_trained', <br>
    'model_file': '8x8_100checkpoints_best.pth.tar', <br>
}) <br><br>
```
  - mainTrain.py : Training of  Reinforcement Learning. <br>

Copyright 2019-2023 (C) Munetomo Maruyama <br>
