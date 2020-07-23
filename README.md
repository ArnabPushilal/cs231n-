# NOTES for this assigment

Some helpful links and things I found helpful I found along the way for the CS231n standord course

**Course link for contents of assignment and lecture modules**
https://cs231n.github.io/


**Gradient Computation for the Linear Classifier with a Soft Max cross entory loss**
https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html - This was not available or not explicty mentioned in the course lectures hence I had to dig this out , as I was just starting out. This gave me an intuition of computing Gradients for other loss functions as well.

EDIT ** Lesson Learned watch till lecture 4 before starting the assignment, things will be more clear about Gradients. Turns out it was a simple chain rule application


**Making Chain rule & gradient Easier for Neural Nets**
http://cs231n.stanford.edu/vecDerivs.pdf - I followed the advice mentioned and coded out the classifier with very small batches to see what was going on in each matrice computation. This helped me visualize the problem . You will see that I have made such practice cells around the notebooks


**Why Sigmoid functions are bad?**
The sigmoid neuron is that when the neuron’s activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero. Recall that during backpropagation, this (local) gradient will be multiplied to the gradient of this gate’s output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. Additionally, one must pay extra caution when initializing the weights of sigmoid neurons to prevent saturation. For example, if the initial weights are too large then most neurons would become saturated and the network will barely learn.

Sigmoid outputs are not zero-centered

**Why you should not use smaller networks to preven overfitting?**

The subtle reason behind this is that smaller networks are harder to train with local methods such as Gradient Descent: It’s clear that their loss functions have relatively few local minima, but it turns out that many of these minima are easier to converge to, and that they are bad (i.e. with high loss). Conversely, bigger neural networks contain significantly more local minima, but these minima turn out to be much better in terms of their actual loss. Since Neural Networks are non-convex, it is hard to study these properties mathematically, but some attempts to understand these objective functions have been made, e.g. in a recent paper The Loss Surfaces of Multilayer Networks. In practice, what you find is that if you train a small network the final loss can display a good amount of variance - in some cases you get lucky and converge to a good place but in some cases you get trapped in one of the bad minima. On the other hand, if you train a large network you’ll start to find many different solutions, but the variance in the final achieved loss will be much smaller. In other words, all solutions are about equally as good, and rely less on the luck of random initialization.
