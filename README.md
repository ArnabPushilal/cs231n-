# NOTES for this assigment

Some helpful links and things I found helpful I found along the way for the CS231n standord course

**Course link for contents of assignment and lecture modules**
https://cs231n.github.io/


**Gradient Computation for the Linear Classifier with a Soft Max cross entory loss**
https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html - This was not available or not explicty mentioned in the course lectures hence I had to dig this out , as I was just starting out. This gave me an intuition of computing Gradients for other loss functions as well.

EDIT ** Lesson Learned watch till lecture 4 before starting the assignment, things will be more clear about Gradients. Turns out it was a simple chain rule application


**Making Chain rule & gradient Easier for Neural Nets**
http://cs231n.stanford.edu/vecDerivs.pdf - I followed the advice mentioned and coded out the classifier with very small batches to see what was going on in each matrice computation. This helped me visualize the problem . You will see that I have made such practice cells around the notebooks

**Playing with the learning rate,data points,no of neurons to help with NN intuition & visulization of each layer**
https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html 
Also visualizing how the input space in context of the hidden layers really helped me devolop the intuition how the non linearaity helps makes the input space 'linearly separable' for the final layer


**Why Sigmoid functions are bad?**
The sigmoid neuron is that when the neuron’s activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero.

**Why you should not use smaller networks to preven overfitting?**

The subtle reason behind this is that smaller networks are harder to train with local methods such as Gradient Descent: It’s clear that their loss functions have relatively few local minima, but it turns out that many of these minima are easier to converge to, and that they are bad (i.e. with high loss). Conversely, bigger neural networks contain significantly more local minima, but these minima turn out to be much better in terms of their actual loss. 
