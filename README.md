# Autograd

A tiny scalar-valued automatic gradient engine that implements backpropagation from scratch on a neural network / MLP. Potentially for learning purposes.

[Google Colab](https://colab.research.google.com/drive/1OIUyZ510KcllyK-8nI_jv_qtWUqixuDX?usp=sharing)

## Libraries Used

- Numpy
- Matplotlib
- Graphviz
- Pytorch

## Approach

Created a class named `Value` for scalars that keeps track of its children, operations, and gradients for backpropagation.

Gradients calculated by the engine are verified manually using the definition of derivative, and using `Pytorch`.

**tanh()** is used as an activation function that squeezes the final output between (-1, 1).

![tanh](tanh.png)

Graphs are generated using `graphviz` for visual understanding of every operation.

![graphviz](graphviz.svg)
