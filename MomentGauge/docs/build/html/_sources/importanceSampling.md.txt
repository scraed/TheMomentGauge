# Importance Sampling
```{eval-rst}
.. _my-reference-label:
Importance sampling helps construct a complex distribution sampler from primary samplers such as quadrature samplers :func:`Sampler.QuadratureSampler`.
```
Suppose we have a primary sampler for the distribution $g(\mathbf{u})$ generating $N$ samples $\mathbf{u}_{g,i}$, weights $w_{g,i}$, and likelihoods $g(\mathbf{u}_{g,i})$ such that 
\begin{equation}
\int \phi(\mathbf{u}) g(\mathbf{u}) d\mathbf{u} \approx \sum_{i=1}^N w_{g,i} \phi(\mathbf{u}_{g,i}),
\end{equation}\label{eq1}
in which $\phi$ is an arbitrary function of $\mathbf{u}$. Our target is to generate samples $\mathbf{u}_{f,i}$ and weights $w_{f,i}$ for a more complex distribution $f(\mathbf{u})$ such that
\begin{equation}
\int \phi(\mathbf{u}) f(\mathbf{u}) d\mathbf{u} \approx \sum_{i=1}^N w_{f,i} \phi(\mathbf{u}_{f,i}).
\end{equation}

Importance sampling helps us achieve this target utilizing the following observation
\begin{equation}\label{eq2}
\begin{split}
\int\phi(\mathbf{u}) f(\mathbf{u}) d\mathbf{u} &=\int \phi(\mathbf{u}) \frac{f(\mathbf{u})}{g(\mathbf{u})} g(\mathbf{u}) d\mathbf{u}\\
&\approx \sum_{i=1}^N w_{g,i} \frac{f(\mathbf{u}_{g,i})}{g(\mathbf{u}_{g,i})}  \phi(\mathbf{u}_{g,i}).
\end{split}
\end{equation}

Comparing the previous equation to equation (2), we conclude that 
\begin{equation}
\begin{split}
w_{f,i} = w_{g,i}\frac{f(\mathbf{u}_i)}{g(\mathbf{u}_i)}\quad \mathbf{u}_{f,i} = \mathbf{u}_{g,i} 
\end{split}
\end{equation}
The above equation gives the weights $w_{f,i}$ and samples $\mathbf{u}_{f,i}$ for the distribution $f(\mathbf{u})$ obtained by importance sampling.

## Self-Normalized Importance Sampling

Suppose we again have a primary sampler for the distribution $g(\mathbf{u})$ generating $N$ samples $\mathbf{u}_{g,i}$, weights $w_{g,i}$ such that

\begin{equation}
\int \phi(\mathbf{u}) g(\mathbf{u}) d\mathbf{u} \approx \sum_{i=1}^N w_{g,i} \phi(\mathbf{u}_{g,i}),
\end{equation}\label{eq1}

in which $\phi$ is an arbitrary function of $\mathbf{u}$. But we only know the likelihoods upto a constant multiplier $c_0$. Specifically, we partially known the likelihood $g(\mathbf{u}_{g,i}) = c_0 g_0(\mathbf{u}_{g,i})$ with $g_0$ known but $c_0$ not.



Our target is to generate samples $\mathbf{u}_{f,i}$ and weights $w_{f,i}$ for a more complex distribution $f(\mathbf{u}) = c_1 f_0(\mathbf{u})$ with $f_0$ known but $c_1$ not. Specifically, we aims to achieve
\begin{equation}
\int \phi(\mathbf{u}) f(\mathbf{u}) d\mathbf{u} \approx \sum_{i=1}^N w_{f,i} \phi(\mathbf{u}_{f,i}).
\end{equation}

Self-Normalized importance sampling helps us achieve this target utilizing the following observation
\begin{equation}
\begin{split}
\int\phi(\mathbf{u}) f(\mathbf{u}) d\mathbf{u} &=\frac{\int \phi(\mathbf{u}) \frac{f(\mathbf{u})}{g(\mathbf{u})} g(\mathbf{u}) d\mathbf{u}}{\int \frac{f(\mathbf{u})}{g(\mathbf{u})} g(\mathbf{u}) d\mathbf{u}}\\
&=\frac{\int \phi(\mathbf{u}) \frac{f_0(\mathbf{u})}{g_0(\mathbf{u})} g(\mathbf{u}) d\mathbf{u}}{\int \frac{f_0(\mathbf{u})}{g_0(\mathbf{u})} g(\mathbf{u}) d\mathbf{u}}\\
&\approx \frac{ \sum_{i=1}^N w_{g,i} \frac{f_0(\mathbf{u}_{g,i})}{g_0(\mathbf{u}_{g,i})}  \phi(\mathbf{u}_{g,i}) }{\sum_{i=1}^N w_{g,i} \frac{f_0(\mathbf{u}_{g,i})}{g_0(\mathbf{u}_{g,i})}   }.
\end{split}
\end{equation}

Comparing the previous equation to equation (6), we conclude that 
\begin{equation}
\begin{split}
w_{f,i} = \frac{ w_{g,i} \frac{f_0(\mathbf{u}_{g,i})}{g_0(\mathbf{u}_{g,i})} }{\sum_{i=1}^N w_{g,i} \frac{f_0(\mathbf{u}_{g,i})}{g_0(\mathbf{u}_{g,i})}   }  \quad \mathbf{u}_{f,i} = \mathbf{u}_{g,i} 
\end{split}
\end{equation}
The above equation gives the weights $w_{f,i}$ and samples $\mathbf{u}_{f,i}$ for the distribution $f(\mathbf{u})$ obtained by importance sampling.