# Linear Regression
## Problem 2.1. 
We have loss function:

$$
L[\phi] = \sum_{i=1}^{I}(f[x_{i}, \phi] - y_{i})^2 = \sum_{i=1}^{I}(\phi_{0}+\phi_{1} \times x_{i} - y_{i})^2
$$

Thus:

$$
\begin{aligned}
\frac{\partial L}{\partial \phi_{0}} &= 2 \times \sum_{i=1}^{I}(f[x_{i}, \phi] - y_{i}) \times \frac{\partial (f[x_{i}, \phi] -y_{i})}{\partial \phi_{0}} \\
&= 2 \times \sum_{i=1}^{I}(f[x_{i}, \phi] - y_{i}) \times 1 \\ &= 2 \times \sum_{i=1}^{I}(f[x_{i}, \phi] - y_{i})
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial \phi_{1}} &= 2 \times \sum_{i=1}^{I}(f[x_{i}, \phi] - y_{i}) \times \frac{\partial (f[x_{i}, \phi] -y_{i})}{\partial \phi_{1}} \\
&= 2 \times \sum_{i=1}^{I}(f[x_{i}, \phi] - y_{i}) \times x_{i}
\end{aligned}
$$
From these, we can write a gradient descent for linear regression. The final result is:

$$
\phi_{0} = 0.8797722839092729, \phi_{1} = 0.4722428504004352
$$
And the final loss is $0.21136832171977532$.

## Problem 2.3.
We have:

$$
x = g[y, \phi] = \phi_{0} + \phi_{1} \times y
$$
Thus, we have new loss function:

$$
L[\phi] = \sum_{i=1}^{I} (f[y_{i}, \phi] - x_{i})^2
$$
The inference function is:

$$
y = g^{-1}[x, \phi] = \frac{x-\phi_{0}}{\phi_{1}}
$$
The gradient will be same, just change $x$ to $y$ and vice versa.

**Note:** 