# Loss Function

## Problem 1

Case 1:

$$
\begin{aligned}
\lim_{ z \to -\infty } \frac{1}{1+\exp(-z)}.
\end{aligned}
$$

The term $1+\exp(-z)$ with $z \to -\infty$ will tend to $\infty$. Thus:

$$
\lim_{ z \to -\infty } \frac{1}{1+\exp(-z)} = \frac{1}{\infty} = 0.
$$

Case 2:

$$
\lim_{ z \to 0 }  \frac{1}{1+\exp(-z)} = \frac{1}{1+1} = \frac{1}{2}.
$$

Case 3:

$$
\lim_{ z \to \infty } \frac{1}{1+\exp(-z)}.
$$

The term $1+\exp(-z)$ with $z \to \infty$ will tend to $1$. Thus:

$$
\lim_{ z \to \infty } \frac{1}{1+\exp(-z)} = \frac{1}{1} = 1.
$$

## Problem 2

Let $z = sig[f[x,\phi]]$. The loss function become:

$$
L = -(1-y)\log(1-z)-y\log(z)
$$

Case 1: The training label $y=0$. We have:

$$
\begin{aligned}
L &= -(1-0)\log(1-z)-0\log(z) \\
&= -\log(1-z).
\end{aligned}
$$

Case 2: The training label $y=1$. We have:

$$
\begin{aligned}
L &= -(1-1)\log(1-z)-1\log(z) \\
&= -\log(z).
\end{aligned}
$$
 Loss function:

![[img/image.png]]

As we can see in the above plotting, the loss of case 2 will high if probability z = 0, which means we make wrong decision, and it is lowing down when probability z tends to 1.0, which means we make correct choice. Case 1 has same intuition.