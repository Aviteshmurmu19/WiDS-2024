*Mirror Descent*

The gradient descent algorithm of the previous chapter is general and powerful: it allows us to (approximately) minimize convex functions over convex bodies. Moreover, it also works in the model of online convex optimization, where the convex function can vary over time, and we want to find a low-regret strategy—one which performs well against every fixed point $x^*$.

This power and broad applicability means the algorithm is not always the best for specific classes of functions and bodies: for instance, for minimizing linear functions over the probability simplex $\Delta_n$, we saw in §16.4.1 that the generic gradient descent algorithm does significantly worse than the specialized Hedge algorithm. This suggests asking: can we somehow *change gradient descent to adapt to the “geometry” of the problem*?

The *mirror descent* framework of this section allows us to do precisely this. There are many different (and essentially equivalent) ways to explain this framework, each with its positives. We present two of them here: the *proximal point view*, and the *mirror map view*, and only mention the others (the *preconditioned or quasi-Newton gradient flow view*, and the *follow the regularized leader view*) in passing.

17.1 *Mirror Descent: the Proximal Point View*

Here is a different way to arrive at the gradient descent algorithm from the last lecture: Indeed, we can get an expression for $x_{t+1}$ by

**Algorithm 15: Proximal Gradient Descent Algorithm**

15.1 $x_1 \leftarrow$ starting point

15.2 **for** $t \leftarrow 1$ **to** $T$ **do**

15.3 | $x_{t+1} \leftarrow \arg \min_x \{\eta\langle\nabla f_t(x_t), x\rangle + \frac{1}{2}||x - x_t||^2\}$

setting the gradient of the function to zero; this gives us the expression

\[
\eta \cdot \nabla f_t(x_t) + (x_{t+1} - x_t) = 0 \implies x_{t+1} = x_t - \eta \cdot \nabla f_t(x_t),
\]

which matches the normal gradient descent algorithm. Moreover, the intuition for this algorithm also makes sense: if we want to minimize the function <span class="math-inline">f\_t\(x\)</span>, we could try to minimize its linear approximation <span class="math-inline">f\_t\(x\_t\)</0\> \+ \\langle \\nabla f\_t\(x\_t\), x \- x\_t \\<0\>rangle</span> instead. But we should be careful not to "over-fit": this linear approximation is good only close to the point <span class="math-inline">x\_t</span>, so we could add in a penalty function (a "regularizer") to prevent us from straying too far from the point <span class="math-inline">x\_t</span>. This means we should minimize

$x_{t+1} \leftarrow \arg \min_x \{f_t(x_t) + \langle \nabla f_t(x_t), x - x_t \rangle + \frac{1}{2}||x - x_t||^2\}$

or dropping the terms that don't depend on *x*,

$$x_{t+1} \leftarrow \arg \min_x \{\langle \nabla f_t(x_t), x \rangle + \frac{1}{2}||x - x_t||^2\} \tag{17.1}$$

If we have a constrained problem, we can change the update step to:

$$x_{t+1} \leftarrow \arg \min_{x \in K} \{\eta\langle \nabla f_t(x_t), x \rangle + \frac{1}{2}||x - x_t||^2\} \tag{17.2}$$

The optimality conditions are a bit more complicated now, but they again can show this algorithm is equivalent to projected gradient descent from the previous chapter.

Given this perspective, we can now replace the squared Euclidean norm by other distances to get different algorithms. A particularly useful class of distance functions are Bregman divergences, which we now define and use.

17.1.1 *Bregman Divergences*

Given a *strictly convex* function $h$, we can define a distance based on how the function differs from its linear approximation:

**Definition 17.1.** The Bregman divergence from $x$ to $y$ with respect to function $h$ is

$$D_h(y||x) := h(y) - h(x) - \langle\nabla h(x), y - x\rangle.$$

The figure on the right illustrates this definition geometrically for a univariate function $h: \mathbb{R} \to \mathbb{R}$. Here are a few examples:

1.  For the function $h(x) = \frac{1}{2}||x||^2$ from $\mathbb{R}^n$ to $\mathbb{R}$, the associated Bregman divergence is

    $$D_h(y||x) = \frac{1}{2}||y - x||^2,$$

    the squared Euclidean distance.

2. For the (un-normalized) negative entropy function $h(x) = \sum_{i=1}^{n}(x_i \ln x_i - x_i)$,

$$D_h(y||x) = \sum_i (y_i \ln \frac{y_i}{x_i} - y_i + x_i).$$

Using that $\sum_i y_i = \sum_i x_i = 1$ for $y, x \in \Delta_n$ gives us $D_h(y||x) = \sum_i y_i \ln \frac{y_i}{x_i}$ for $x, y \in \Delta_n$: this is the Kullback-Leibler (KL) divergence between probability distributions.

Many other interesting Bregman divergences can be defined.

17.1.2 *Changing the Distance Function*

Since the distance function $\frac{1}{2}||x - y||^2$ in (17.1) is a Bregman divergence, what if we replace it by a generic Bregman divergence: what algorithm do we get in that case? Again, let us first consider the unconstrained problem, with the update:

$$x_{t+1} \leftarrow \arg \min_x \{\eta\langle\nabla f_t(x_t), x\rangle + D_h(x||x_t)\}.$$

Again, setting the gradient at $x_{t+1}$ to zero (i.e., the optimality condition for $x_{t+1}$) now gives:

$$\eta\nabla f_t(x_t) + \nabla h(x_{t+1}) - \nabla h(x_t) = 0,$$

or, rephrasing

$$\nabla h(x_{t+1}) = \nabla h(x_t) - \eta\nabla f_t(x_t) \tag{17.3}$$

$$\Rightarrow x_{t+1} = \nabla h^{-1}(\nabla h(x_t) - \eta\nabla f_t(x_t)) \tag{17.4}$$

Let's consider this for our two running examples:

1. When $h(x) = \frac{1}{2}||x||^2$, the gradient $\nabla h(x) = x$. So we get

$$x_{t+1} = x_t - \eta f_t(x_t),$$

the standard gradient descent update.

2. When $h(x) = \sum_i (x_i \ln x_i - x_i)$, then $\nabla h(x) = (\ln x_1, ..., \ln x_n)$, so

$$(x_{t+1})_i = e^{\ln(x_i) - \eta\nabla f_i(x_i)} = (x_t)_i \cdot e^{-\eta f_i(x_i)}.$$

Now if $f_t(x) = \langle l_t, x \rangle$, its gradient is just the vector $l_t$, and we get back precisely the *weights* maintained by the Hedge algorithm!

The same ideas also hold for constrained convex minimization: we now have to search for the minimizer within the set $K$. In this case the algorithm using negative entropy results in the same Hedge-like update, followed by scaling the point down to get a probability vector, thereby giving the probability values in Hedge.

To summarize: this algorithm that tries to minimize the linear ap- What would be the "right" choice of $h$ to minimize the function $f$? It would be $h = f$, because adding $D_f(x||x_t)$ to the linear approximation of $f$ at $x_t$ gives us back exactly $f$. Of course, the update now requires us to minimize $f(x)$, which is the original problem. So we should choose an $h$ that is "similar" to $f$, and yet such that the update step is tractable.

**Algorithm 16: Proximal Gradient Descent Algorithm**

16.1 $x_1 \leftarrow$ starting point

16.2 **for** $t \leftarrow 1$ **to** $T$ **do**

16.3 | $x_{t+1} \leftarrow \arg \min_{x \in K} \{\eta\langle \nabla f_t(x_t), x \rangle + D_h(x||x_t)\}$

proximation of the function, regularized by a Bregman distance $D_h$, gives us vanilla gradient descent for one choice of *h* (which is good for quadratic-like functions over Euclidean space), and Hedge for another choice of *h* (which is good for linear functions over the space of probability distributions). Indeed, depending on how we choose the function, we can get different properties from this algorithm—this is the mirror descent framework.

17.2 *Mirror Descent: The Mirror Map View*

A different view of the mirror descent framework is the one originally presented by Nemirovski and Yudin. They observe that in gradient descent, at each step we set $x_{t+1} = x_t - \eta f_t(x_t)$. However, the *gradient* was actually defined as a linear functional on $\mathbb{R}^n$ and hence naturally belongs to the *dual space* of $\mathbb{R}^n$. The fact that we represent this functional (i.e., this covector) as a vector is a matter of convenience, and we should exercise care.

In the vanilla gradient descent method, we were working in $\mathbb{R}^n$ endowed with $\ell_2$-norm, and this normed space is self-dual, so it is perhaps reasonable to combine points in the primal space (the iterates $x_t$ of our algorithm) with objects in the dual space (the gradients). But when working with other normed spaces, adding a covector $\nabla f_t(x_t)$ to a vector $x_t$ might not be the right thing to do. Instead, Nemirovski and Yudin propose the following:

...

---

17.2.1 *Norms and their Duals*

**Definition 17.2 (Norm).** A function $||\cdot|| : \mathbb{R}^n \to \mathbb{R}$ is a norm if

*   If $||x|| = 0$ for $x \in \mathbb{R}^n$, then $x = 0$;
*   for $\alpha \in \mathbb{R}$ and $x \in \mathbb{R}^n$ we have $||\alpha x|| = |\alpha|||x||$; and
*   for $x, y \in \mathbb{R}^n$ we have $||x + y|| \le ||x|| + ||y||$.

The well-known $\ell_p$-norms for $p \ge 1$ are defined by

$$||x||_p := (\sum_{i=1}^{n}|x_i|^p)^{1/p}$$

for $x \in \mathbb{R}^n$. The $\ell_\infty$-norm is given by

$$||x||_\infty := \max_{i=1}^{n} |x_i|$$

for $x \in \mathbb{R}^n$.

**Definition 17.3 (Dual Norm).** Let $||\cdot||$ be a norm. The dual norm of $||\cdot||$ is a function $||\cdot||_*$ defined as

$$||y||_* := \sup\{\langle x, y \rangle : ||x|| \le 1\}.$$

The dual norm of the $\ell_2$-norm is again the $\ell_2$-norm; the Euclidean norm is self-dual. The dual for the $\ell_p$-norm is the $\ell_q$-norm, where $1/p + 1/q = 1$.

**Corollary 17.4 (Cauchy-Schwarz for General Norms).** For $x, y \in \mathbb{R}^n$, we have $\langle x, y \rangle \le ||x|| ||y||_*$.

*Proof.* Assume $||x|| \ne 0$, otherwise both sides are 0. Since $||\frac{x}{||x||}|| = 1$, we have $\langle \frac{x}{||x||}, y \rangle \le ||y||_*$.

**Theorem 17.5.** For a finite-dimensional space with norm $||\cdot||$, we have $(||\cdot||_*)^* = ||\cdot||$.

---

Having fixed $|| \cdot ||$ and $h$, the *mirror map* is

$$\nabla h : \mathbb{R}^n \to \mathbb{R}^n.$$

Since $h$ is differentiable and strongly-convex, we can define the inverse map as well. This defines the mappings that we use in the Nemirovski-Yudin process: we set

$$\theta_t = \nabla h(x_t) \quad \text{and} \quad x_{t+1}' = (\nabla h)^{-1}(\theta_{t+1}).$$

For our first running example of $h(x) = \frac{1}{2}||x||^2$ the gradient (and hence its inverse) is the identity map. For the (un-normalized) negative entropy example, $(\nabla h(x))_i = \ln x_i,$ and hence $(\nabla h)^{-1}(\theta)_i = e^{\theta_i}.$

17.2.3 *The Algorithm (Again)*

Let us formally state the algorithm again, before we state and prove a theorem about it. Suppose we want to minimize a convex function $f$ over a convex body $K \subseteq \mathbb{R}^n$. We first fix a norm $|| \cdot ||$ on $\mathbb{R}^n$ and choose a distance-generating function $h : \mathbb{R}^n \to \mathbb{R}$, which gives the mirror map $\nabla h : \mathbb{R}^n \to \mathbb{R}^n$. In each iteration of the algorithm, we do the following:

(i) Map to the dual space $\theta_t \leftarrow \nabla h(x_t)$.

(ii) Take a gradient step in the dual space: $\theta_{t+1} \leftarrow \theta_t - \eta_t \cdot \nabla f_t(x_t).$

(iii) Map $\theta_{t+1}$ back to the primal space

$$x_{t+1}' \leftarrow (\nabla h)^{-1}(\theta_{t+1}).$$

(iv) Project $x_{t+1}'$ back into the feasible region $K$ by using the Bregman divergence: $x_{t+1} \leftarrow \min_{x \in K} D_h(x||x_{t+1}')$. In case $x_{t+1}' \in K$, e.g., in the unconstrained case, we get $x_{t+1} = x_{t+1}'.$

Note that the choice of $h$ affects almost every step of this algorithm.

---

Summing over all times,

$$\sum_{t=1}^{T} f_t(x_t) - \sum_{t=1}^{T} f_t(x^*) \le \Phi_1 - \Phi_{T+1} + \sum_{t=1}^{T} \text{blah}_t$$

$$\le \Phi_1 + \sum_{t=1}^{T} \text{blah}_t = \frac{D_h(x^*||x_1)}{\eta} + \sum_{t=1}^{T} \text{blah}_t.$$

The last inequality above uses that the Bregman divergence is always non-negative for convex functions.

To complete the proof, it remains to show that blah<sub>t</sub> in inequality (17.6) can be made $\frac{\alpha}{2\eta}||\nabla f_t(x_t)||^2$. Let us focus on the unconstrained case, where $x_{t+1} = x_{t+1}'$. The calculations below are fairly routine, and can be skipped at the first reading:

$$\Phi_{t+1} - \Phi_t = \frac{1}{\eta}(D_h(x^*||x_{t+1}) - D_h(x^*||x_t))$$

$$= \frac{1}{\eta}(h(x^*) - h(x_{t+1}) - \langle\nabla h(x_{t+1}), x^* - x_{t+1}\rangle - h(x^*) + h(x_t) + \langle\nabla h(x_t), x^* - x_t\rangle)$$

$$= \frac{1}{\eta}(h(x_t) - h(x_{t+1}) - \langle\theta_{t+1}, x^* - x_{t+1}\rangle + \langle\theta_t, x^* - x_t\rangle)$$

$$= \frac{1}{\eta}(h(x_t) - h(x_{t+1}) - \langle\theta_t - \eta\nabla f_t(x_t), x^* - x_{t+1}\rangle + \langle\theta_t, x^* - x_t\rangle)$$

$$\le \frac{1}{\eta}(-\frac{\alpha}{2}||x_{t+1} - x_t||^2 + \eta\langle\nabla f_t(x_t), x^* - x_{t+1}\rangle) \quad \text{(By $\alpha$-strong convexity of $h$ wrt to $||\cdot||$)}$$

Substituting this back into (17.6):

$$f_t(x_t) - f_t(x^*) + (\Phi_{t+1} - \Phi_t)$$

$$\le f_t(x_t) - f_t(x^*) - \frac{\alpha}{2\eta}||x_{t+1} - x_t||^2 + \langle\nabla f_t(x_t), x^* - x_{t+1}\rangle$$

$$\le f_t(x_t) + \langle\nabla f_t(x_t), x^* - x_t\rangle - \frac{\alpha}{2\eta}||x_{t+1} - x_t||^2 + \langle\nabla f_t(x_t), x_t - x_{t+1}\rangle$$

$$\le 0 \text{ by convexity of } f_t$$

---

To complete the proof, it remains to show that blah<sub>t</sub> in inequality (17.6) can be made $\frac{\alpha}{2\eta}||\nabla f_t(x_t)||^2$. Let us focus on the unconstrained case, where $x_{t+1} = x_{t+1}'$. The calculations below are fairly routine, and can be skipped at the first reading:

$$\Phi_{t+1} - \Phi_t = \frac{1}{\eta}(D_h(x^*||x_{t+1}) - D_h(x^*||x_t))$$

$$= \frac{1}{\eta}(h(x^*) - h(x_{t+1}) - \langle\nabla h(x_{t+1}), x^* - x_{t+1}\rangle - h(x^*) + h(x_t) + \langle\nabla h(x_t), x^* - x_t\rangle)$$


$$= \frac{1}{\eta}(h(x_t) - h(x_{t+1}) - \langle\theta_{t+1}, x^* - x_{t+1}\rangle + \langle\theta_t, x^* - x_t\rangle)$$

$$= \frac{1}{\eta}(h(x_t) - h(x_{t+1}) - \langle\theta_t - \eta\nabla f_t(x_t), x^* - x_{t+1}\rangle + \langle\theta_t, x^* - x_t\rangle)$$
$$\qquad \qquad  \qquad \nabla f_t$$
$$= \frac{1}{\eta}(h(x_t) - h(x_{t+1}) - \langle\theta_t, x^* - x_{t+1}\rangle + \eta\langle\nabla f_t(x_t), x^* - x_{t+1}\rangle + \langle\theta_t, x^* - x_t\rangle)$$

$$\le \frac{1}{\eta}(-\frac{\alpha}{2}||x_{t+1} - x_t||^2 + \eta\langle\nabla f_t(x_t), x^* - x_{t+1}\rangle) \quad \text{(By $\alpha$-strong convexity of $h$ wrt to $||\cdot||$)}$$

Substituting this back into (17.6):

$$f_t(x_t) - f_t(x^*) + (\Phi_{t+1} - \Phi_t)$$

$$\le f_t(x_t) - f_t(x^*) - \frac{\alpha}{2\eta}||x_{t+1} - x_t||^2 + \langle\nabla f_t(x_t), x^* - x_{t+1}\rangle$$

$$\le f_t(x_t) - f_t(x^*) + \langle\nabla f_t(x_t), x^* - x_t\rangle - \frac{\alpha}{2\eta}||x_{t+1} - x_t||^2 + \langle\nabla f_t(x_t), x_t - x_{t+1}\rangle$$

$$\le 0 \text{ by convexity of } f_t$$

$$\le -\frac{\alpha}{2\eta}||x_{t+1} - x_t||^2 + ||\nabla f_t(x_t)||_*||x^* - x_{t+1}|| \quad \text{(By Corollary 17.4)}$$

$$\le -\frac{\alpha}{2\eta}||x_{t+1} - x_t||^2 + \frac{1}{2}(\frac{\eta}{\alpha}||\nabla f_t(x_t)||_*^2 + \frac{\alpha}{\eta}||x^* - x_{t+1}||^2) \quad \text{(By AM-GM)}$$

$$= \frac{\eta}{2\alpha}||\nabla f_t(x_t)||_*^2$$

This completes the proof of Theorem 17.7. As you observe, it is syntactically similar to the original proof of gradient descent, just

---

**Theorem 4.1.** Suppose that assumption A is satisfied for the convex optimization problem (P). Let $\{x^k\}$ be the sequence generated by SANP with starting point $x^1 \in \text{int}(X)$. Then, for every $k \ge 1$ one has

(a) $$\min_{1 \le s \le k} f(x^s) - \min_{x \in X} f(x) \le \frac{B_\psi(x^*, x^1) + 2\sigma^{-1} \sum_{s=1}^{k} t_s^2 ||f'(x^s)||^2}{\sum_{s=1}^{k} t_s}. \tag{4.15}$$

(b) In particular, the method converges, i.e., $\min_{1 \le s \le k} f(x^s) - \min_{x \in X} f(x) \to 0$ provided that $\sum_s t_s = \infty$, $t_k \to 0$, $k \to \infty$.

---

**Theorem 4.2.** Suppose that assumption A is satisfied for the convex optimization problem (P). Let $\{x^k\}$ be the sequence generated by SANP with starting point $x^1 \in \text{int} X$. Then, with the stepsizes chosen as

$$t_k := \frac{\sqrt{2\sigma B_\psi(x^*, x^1)}}{L_f}\frac{1}{\sqrt{k}}, \tag{4.23}$$

one has the following efficiency estimate

$$\min_{1 \le s \le k} f(x^s) - \min_{x \in X} f(x) \le L_f\sqrt{\frac{2B_\psi(x^*, x^1)}{\sigma}}\frac{1}{\sqrt{k}}. \tag{4.24}$$

---

**Theorem 5.1.** Let $\{x^k\}$ be the sequence generated by EMDA with starting point $x^1 = n^{-1}e$. Then, for all $k \ge 1$ one has

$$\min_{1 \le s \le k} f(x^s) - \min_{x \in X} f(x) \le \frac{\sqrt{2 \ln n} ||f'(x)||_{\infty}}{\sqrt{k}}. \tag{5.28}$$

Thus, the EMDA appears as another useful candidate algorithm for solving large scale convex minimization problems over the unit simplex. Indeed, EMDA shares the same efficiency estimate than the $(MDA_1)$ obtained with $\psi_1$, but has the advantage of being completely explicit, as opposed to the $(MDA_1)$, which still requires the solution of one-dimensional nonlinear equation at each step of the algorithm to compute $\psi_i^*$.

**6. Concluding remarks and further applications**

We have presented a new derivation and analysis of mirror descent type algorithms. In its current state, the proposed approach has given rise to new insights on the properties of Mirror descent methods, bringing it in line of subgradient projection algorithms based on Bregman-based distance-like functions. This has led us to provide simple proofs for its convergence analysis and to introduce the new algorithm (EMDA) for solving convex problems over the unit simplex, with efficiency estimate mildly dependent on the problem's dimension. Many issues for potential extensions and further analysis include:

*   Extension to the cases where $f(x) = \sum_{i=1}^{m} f_i(x)$ which can be derived along the analysis of incremental subgradients techniques [9,1] and numerical implementations for the corresponding EMDA.
*   The choice of other functions $\psi$ can be considered in SANP, (see for example [14,8]) to produce other interior subgradient (gradient) methods.
*   Extension to semidefinite programs, in particular for problems with constraints of the type

---

**3 Parallelized SGD**

Parallelized SGD can be integrated into MapReduce where we can achieve even faster results. Earlier attempts to utilize distributed gradient calculations locally on each computer node that maintains portions of the data, followed by gradient aggregation to achieve a global update step. Each computer node must be synchronous and transfer information among them. In a simpler approach, parallelized SGD runs the SGD algorithm in each processor with a fixed learning rate on the local MapReduce"d" dataset over some iterations. Additionally, while running SGD on each processor where the MapReduce"d" data is located, we also run a master routine to record solutions coming from each computer node. On each computer node, we get updated weights, take their average, and return it at the end of the iterations. This process does not require constant communication between computer nodes, resulting in much faster convergence, preferably in the MapReduce framework. [18]

$$i \in 1,..., k, v_i = SGD(\Phi_{(z)}^{(i)}, T, \eta, \omega_0), v = \frac{1}{k}\sum_{i=1}^{k} v_i \tag{19}$$

**3.1 Optimize Parallelized and Distributed SGD**

There are several algorithms to optimize the parallelized SGD. Hogwild enables SGD updates on parallel processors. Processors are allowed to use a shared memory without forcing weights to be pre-determined. However, since processors share a memory, the bandwidth limitation might be a problem in overly large datasets. [14] [18] Downpour SGD is an asynchronous variant of SGD, each computer node or processor responsible storing and updating weights, however, processors do not communicate with each other, which might result in divergence. [5] [18] Delay-tolerant algorithms for SGD uses AdaGrad, which adapts to past gradients and update delays accordingly. [13] [18] Tensorflow Google's initiative that enables users to train large scale machine learning models with minor code change. For example, you can use distribute Strategy API to train your model across many CPUs, GPUs and TPUs, which might offer fast implementation if you aim to parallelize your training dataset. [12] Elastic Averaging ties weights with an elastic force, which lead to fluctuation and prospective better local minimums. [17] [18]

---

**1.5 Application: SGD Perceptron using Iris Dataset**

The perceptron is an algorithm for supervised learning of binary classifiers. It is a type of linear classifier [8]. The mathematical expression of gradient algorithms for perceptron is represented below. [2]

$$
\begin{aligned}
Q_\text{perceptron} &= \max\{0, -yw^T\Phi(x)\} \\
\Phi(x) &\in \mathbb{R}^d, y = \pm 1 \\
w &\leftarrow w + y_t\Phi(x_t) \quad \text{if } y_t w^T \Phi(x_t) \le 0 \\
0 &\qquad \qquad \qquad \qquad \text{otherwise}
\end{aligned} \tag{17}
$$

Here is the results of comparison for three different gradient descend method:

| Method        | accuracy | running time |
| ------------- | -------- | ------------ |
| Fully GD      | 92.67%   | 0.03s        |
| SGD           | 93.33%   | 0.02s        |
| Mini-batch SGD | 92.67%   | 0.06s        |

Table 1. Execution times and accuracies of different methods

From the result the accuracy of three method is quite close, but the running time is different, SGD spend the least time, then is Fully GD, Mini-batch SGD spent the most time. It should be noted that because of the computer architecture or system capabilities, the running time might differ from machine to machine.

**2 Application: Stochastic Gradient Descent Variants on Convolutional Neural Networks**

We used SGD based optimizers in the context of Convolutional neural network since it is well-known fact that the selecting learning rate is problematic. [9] Therefore, we introduced other variants. As mentioned before, the trainable dataset *Iris dataset* has 150 samples. It has 3 class labels and 4 features.

Then we train the model, after 3000 epochs the training and validation loss plot is showed in figure 16: From the plot, we can see that the model converge after about 1500 epochs. At this time, training loss is 0.1388 and validation loss is 0.1391. Additionally, training accuracy is 94% and validation accuracy is 94%. After that, the model will be overfitting. One drawback of SGD method is that its update direction is completely dependent on the gradient calculated by the current batch, so it is very unstable. However, momentum algorithm will observe the historical

---

**Nesterov Accelerated Gradient Descent**

Nesterov accelerated gradient descent (NAG) [59] finds a more brilliant ball that knows where it slows down before speeding up again. The NAG updates the gradient of the future position. The gradient of the current situation is replaced by the next position, which is approximated to $\theta_{t-1} - \alpha \nabla f(\theta_{t-1})$. NAG iterates the following equations for each step:

$$
\begin{align}
\theta_{t} &= \theta_{t-1} + \alpha v_{t-1} \tag{4} \\
v_{t} &= \beta v_{t-1} + \alpha (1-\beta) \nabla f(\theta_{t-1} - \alpha v_{t-1})
\end{align}
$$

where $\theta, \alpha > 0$.

Figure 3 shows the HB(1-2-3) and NAG(1-4-5). HB first computes the current gradient (small blue arrow near point 1), then makes a big jump with momentum $v_{k+1}$ (big blue arrow) to point 2 (the real green arrow is the new momentum, $v_{k+1}$). The next step begins at point 2. HB computes the current gradient (small blue arrow near point 2), then makes a big jump with momentum $v_{k+1}$ (green broken arrow) to point 3, and the new momentum, $v_{k+2}$, is the purple arrow. NAG first makes a big jump with momentum $v_{k}$ (red broken arrow), measures the gradient, and then makes a correction (red line arrow near point 4). The next step begins at point 4. NAG makes a big jump with new momentum $v_{k+1}$ (yellow broken arrow) to measure the gradient and then makes a correction (red line arrow near point 5).

NAG+SGD (SNAG) converges sublinearly without a convex assumption [61]. The ASG method has been proved under the smooth, strongly convex assumption and has achieved a sublinear convergence rate [61]. The authors of [70] analyzed the convergence of variants of the NAG and SNAG, respectively. The SNAG weighs the speed of convergence and stability [67]. The authors of [71] demonstrated some limitations of NAG, such as the possibility of not converging or achieving speedup under finite sum conditions. These theoretical results were confirmed in models such as SVM and DNN. Considering only performance, SNAG performs well on many tasks, such as RNNs [72]. A new type of momentum named Katyusha momentum was proposed in [73]. However, if the step size is chosen as small, it leads to a slower convergence rate, which cancels out the benefit of the momentum term [69].

---

By Equation 5.3, we can write
$$
f(x^{(k)}) \geq f(z^{(k)}) \geq f(z^{(k+1)}), \forall k. \tag{5.7}
$$
Let $\bar{x} = (\bar{x}_1, ..., \bar{x}_n)$ be a limit point of the sequence $\{x^{(k)}\}$. Let $x \in X$, where $X$ is a closed set. Hence, $\bar{x} \in X$. Equation 5.7 indicates that the sequence $\{f(x^{(k)})\}$ converges to $f(\bar{x})$. Now, it is to be shown that $\bar{x}$ minimizes $f$ over $X$.

Let $\{x^{(k_j)}\}_{j=0, 1, ...}$ be a subsequence of $\{x^{(k)}\}$ that converges to $\bar{x}$. We first show that $\{x^{(k_j+1)} - x^{(k_j)}\}$ does not converge to zero as $j \rightarrow \infty$. Assume the contrary, or equivalently, that $\{||x^{(k_j+1)} - x^{(k_j)}||\}$ converges to zero as $j \rightarrow \infty$. Let $y^{(k_j)} = \frac{x^{(k_j+1)} - x^{(k_j)}}{||x^{(k_j+1)} - x^{(k_j)}||}$. By possibly restricting to a subsequence of $\{k_j\}$, we may assume that there exists some $\gamma > 0$ such that $||y^{(k_j)}|| \geq \gamma$ for all $j$. Let $s_1^{(k_j)} = \frac{1}{||y^{(k_j)}||}$. Thus $z^{(k_j)} = x^{(k_j)} + \gamma s_1^{(k_j)} \in X$, and $s_1^{(k_j)}$ differs from zero only along the first coordinate direction. $s_1^{(k_j)}$ belongs to a compact set and therefore has a limit point $\bar{s}_1$. By restricting to a further subsequence of $\{k_j\}$, we can assume that $s_1^{(k_j)}$ converges to $\bar{s}_1$.

Let us fix some $\epsilon \in (0, 1]$. Now, $0 \leq \epsilon \gamma \leq \gamma^{(k_j)}$. Therefore, $x^{(k_j)} + \epsilon \gamma s_1^{(k_j)}$ lies on the segment of the line joining $x^{(k_j)}$ and $x^{(k_j)} + \gamma s_1^{(k_j)} = x^{(k_j+1)}$, and belongs to $X$, because $X$ is convex. Using the fact that $x^{(k_j+1)}$ minimizes $f$ over all $x$ that differ from $x^{(k_j)}$ along the first coordinate direction, we obtain,
$$
f(x^{(k_j+1)}) = f(x^{(k_j)} + \gamma s_1^{(k_j)}) \leq f(x^{(k_j)} + \epsilon \gamma s_1^{(k_j)}) \leq f(x^{(k_j)}). \tag{5.8}
$$
Since $\{f(x^{(k)})\}$ converges to $f(\bar{x})$, Equation 5.7 shows that $\{f(z^{(k_j)})\}$ also converges to $f(\bar{x})$. Now we can take the limit as $j \rightarrow \infty$, to obtain $f(\bar{x}) \leq f(\bar{x} + \epsilon \gamma \bar{s}_1) \leq f(\bar{x})$. We conclude that
