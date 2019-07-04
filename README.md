<p align="center">
  <img src="https://i.ibb.co/bvbW4YL/photo2.jpg" alt="logo">
</p>

# Learning-Lab-C-Library:

Creating the library:

```
make
```

# Run a file.c file using llab:

Link in your file.c the header "/directory-path/llab.h"
Link in your file.c the header "/directory-path/llab_dt.h"

and then

```
gcc -o file  -L /path-to-the-llab.a-library-created-with-the-makefile/ file.c -lllab -lm
```

# Current Roadmap:

- fully-connected-layers feed forward (20/11/2018)
- fully-connected-layers backpropagation (20/11/2018)
- nesterov momentum (20/11/2018)
- adam optimization algorithm (not implemented yet in model_update function) (20/11/2018)
- fully-connected layers dropout (20/11/2018)
- convolutional layers feed forward (20/11/2018)
- convolutional layers backpropagation (20/11/2018)
- convolutional 2d max-pooling (20/11/2018)
- convolutional 2d avarage pooling (20/11/2018)
- convolutional 2d local response normalization (20/11/2018)
- convolutional padding (20/11/2018)
- fully-connected sigmoid activation (20/11/2018)
- fully-connected relu activation (20/11/2018)
- fully-connected softmax activation (20/11/2018)
- fully-connected tanh activation (20/11/2018)
- mse loss (20/11/2018)
- cross-entropy loss (20/11/2018)
- reduced cross-entropy form with softmax (20/11/2018)
- convolutional sigmoid activation (20/11/2018)
- convolutional relu activation (20/11/2018)
- convolutional tanh activation (20/11/2018)
- residual layers filled with convolutional layers (20/11/2018)
- residual layers feed-forward (20/11/2018)
- residual layers backpropagation (20/11/2018)
- model structure with fully-connected,convolutional,residual layers (20/11/2018)
- fixed residual issues (22/11/2018)
- adam algorithm for model_update (22/11/2018)
- size_of_model(model* m) function (23/11/2018)
- L2 Regularization (27/11/2018)
- Manual Batch normalization feed forward and backpropagation (27/11/2018)
- fixed residual issues2 (28/11/2018)
- Manual Xavier Initialization (28/11/2018)
- Clipping gradient (29/1/2019)
- Convolutional Layers with only pooling function (30/1/2019)
- Leaky Relu Activation function (30/1/2019)
- Batch Normalization final mean and variance for feed forward output (1/2/2019)
- Decision Tree structure (3/2/2019)
- LSTM feed forward (13/5/2019)
- LSTM back propagation (13/5/2019)
- Recurrent Network (rmodel) with LSTM (13/5/2019)
- Recurrent Network update with nesterov and adam algorithm (13/5/2019)
- Group Normalization for convolutional layers (4/7/2019)

# Future implementations
- Residual LSTM coming soon...

# Examples:

- https://github.com/VivianoRiccardo/Image-Recognition-LLAB-Library
- https://github.com/VivianoRiccardo/Tetris-DQN-LlabLibrary
