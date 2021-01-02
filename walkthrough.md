# ml-simple-xnor-neural-network

### The mathematical explanation regarding neural network is in the Jupyter file in this repository.
This is a simple neural network that predicts an XNOR gate output.

XNOR gate logic:

```
A | B | Output
---------------
1 | 0 |   0
0 | 1 |   0
0 | 0 |   1
1 | 1 |   1
```
##Implementation

An XNOR gate has the following logic:

<img src="https://www.electronics-tutorial.net/wp-content/uploads/2015/08/XNOR1.png" align="center"/>

# Notation
[math]
    \Theta = \begin{bmatrix}
    \theta^{(1)} & \theta^{(2)}
    \end{bmatrix}\newline
    A=\begin{bmatrix}
    a^{(1)} & a^{(2)} & a^{(3)}
    \end{bmatrix}\newline
    \Delta=\begin{bmatrix}
    \delta^{(1)} & \delta^{(2)} & \delta^{(3)}
    \end{bmatrix}\newline
    a^{(L)} = \text{Output layer}\newline\newline
[/math]

#### First we feed the input into $$a^{(1)}$$
# Feedforward
#### To feedforward, we pass the weighted sum into the Sigmoid activation function:

[math]sigma(\sum_{i=0}^{S_l}\sum_{j=1}^{S_{l+1}}a_{i}^{(l)}\theta_{ji}^{(l)}) = \sigma(\Theta^{(l)}A^{(l)})[/math]

#### After feeding forward for training example $$x^{(t)}$$, we now perform backpropagation by calculating error for each layer:

#Backpropagation
The error for layer $$l$$ is calculated as:

[math]
    \frac{\partial J(\Theta)}{\partial\Theta^{(l)}} = 
    \frac{\partial J(\Theta)}{\partial a^{(l)}}
    \frac{\partial a^{(l)}}{\partial z^{(l)}}
    \frac{\partial z^{(l)}}{\partial \Theta^{(l)}}\newline\newline
[/math]


#### Without going too much into detail, the formula for the error in layer $l$ is:

[math]
    \text{for }0...L:\newline
    \delta^{(l)}=\begin{cases}
        y^{(t)} - a^{(L)}, \text{if }l = L\newline
        ((\Theta^{(l)})^T \delta^{(l + 1)}) \circ a^{(l)}
    \circ (1 - a^{(l)}), \text{if }l < L - 2
        \end{cases}
[/math]


#### After calculating the error, we can update the weights with the following:

[math]
        \Theta^{(l)}:=\Theta^{(l)} - \alpha \frac{\partial J(\Theta)}{\partial \Theta^{(l)}}\equiv \Theta^{(l)} - \frac{\alpha}{m} a^{(l)}\delta^{(l)}
[/math]


#### To update biases, the following update rule is performed:

[math]
    \Theta_b^{(l)}:= \Theta_b^{(l)} - \frac{\alpha}{m}1 \cdot \delta^{(l + 1)}
[/math]
