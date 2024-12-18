Instructions: (Please read carefully and follow them!)

Try to solve all problems on your own. If you have difficulties, ask the instructor or TAs.

In this session, we will start with implementation of algorithms for solving nonlinear optimization problems. Today, we will discuss gradient descent to solve problems of the form min<sub>x∈R<sup>n</sup></sub> f(x).

Gradient descent is one of the oldest algorithms (dating back to Cauchy), yet simple and popular in many scientific communities even today. Gradient descent works iteratively where in each iteration a suitable descent direction and an appropriate step length are chosen to update the current iterate.

The implementation of the optimization algorithms in this lab will involve extensive use of the numpy Python package. It would be useful for you to get to know some of the functionalities of numpy package. For details on numpy Python package, please consult [https://numpy.org/doc/stable/index.html](https://numpy.org/doc/stable/index.html)

For plotting purposes, please use matplotlib.pyplot package. You can find examples in the site [https://matplotlib.org/examples/](https://matplotlib.org/examples/).

Please follow the instructions given below to prepare your solution notebooks:

- Please use different notebooks for solving different Exercise problems.
- The notebook name for Exercise 1 should be `YOURROLLNUMBER_IE684_Lab1_Ex1.ipynb`.
- Similarly, the notebook name for Exercise 2 should be `YOURROLLNUMBER_IE684_Lab1_Ex2.ipynb`, etc.
- Please post your doubts in MS Teams Discussion Forum channel so that TAs can clarify.

There are only 2 exercises in this lab. Try to solve all problems on your own. If you have difficulties, ask the Instructors or TAs.

Only the questions marked [R] need to be answered in the notebook. You can either print the answers using `print` command in your code or you can write the text in a separate text tab. To add text in your notebook, click +Text. Some questions require you to provide proper explanations; for such questions, write proper explanations in a text tab. Some questions require the answers to be written in LaTeX notation. Please see the demo video to know how to write LaTeX in Google notebooks. Some questions require plotting certain graphs. Please make sure that the plots are present in the submitted notebooks.

After completing this lab's exercises, click File → Download .ipynb and save your files to your local laptop/desktop. Create a folder with name `YOURROLLNUMBER_IE684_Lab1` and copy your `.ipynb` files to the folder. Then zip the folder to create `YOURROLLNUMBER_IE684_Lab1.zip`. Then upload only the `.zip` file to Moodle. There will be extra marks for students who follow the proper naming conventions in their submissions.

Please check the submission deadline announced in moodle.

The first Laboratory exercise aims to help you learn the plotting of the level sets of the function and implementation of optimization algorithms such as zero order, first order (Gradient Descent) and second order (Newton's method).

**Exercise 1 (15 marks)** Plot the good-looking visuals of _level sets_ for each of the following functions in (1) to (4):-

1.  f(x) = f(x<sub>1</sub>,x<sub>2</sub>) = 100x<sub>1</sub><sup>4</sup> + 100x<sub>2</sub><sup>2</sup> - 200x<sub>1</sub><sup>2</sup>x<sub>2</sub> + x<sub>1</sub><sup>2</sup> - 2x<sub>1</sub> + 1
2.  f(x) = f(x<sub>1</sub>,x<sub>2</sub>) = x<sub>1</sub><sup>4</sup> + x<sub>2</sub><sup>4</sup> - 20x<sub>1</sub><sup>3</sup> - 20x<sub>2</sub><sup>3</sup> + 100x<sub>1</sub><sup>2</sup> + 100x<sub>2</sub><sup>2</sup>
3.  f(x) = f(x<sub>1</sub>, x<sub>2</sub>) = -cos(x<sub>1</sub>)cos(x<sub>2</sub>)exp(-((x<sub>1</sub> - π)<sup>2</sup> + (x<sub>2</sub> – π)<sup>2</sup>))
4.  f(x) = f(x<sub>1</sub>,x<sub>2</sub>) = 2x<sub>1</sub><sup>4</sup> - 1.05x<sub>1</sub><sup>6</sup> + x<sub>1</sub>x<sub>2</sub> + x<sub>2</sub><sup>2</sup>

5.  What do you observe from the level sets so obtained in (1) to (4)? Are they useful in characterizing the optimal solution visually, give reasoning via finding minima of each of the functions in (1) to (4) using usual first and second order optimality conditions.

**Note:-** Please use the theory references wisely related to the level sets, you need to explore the equivalent Python commands over the internet. Please use the axes range at least -4 to 4 and also choose levels in log or linear space wisely depending on particular function.

**Exercise 2 (20 marks)** In zero-order oracle optimization, the focus is solely on utilizing function values without access to derivatives. The goal is to find the optimal solution and optimal function value within a specified rectangle or box by iteratively evaluating the function within this region. The Algorithm 1 is corresponding to the zero-order oracle optimization:

**Algorithm 1 Zero Order Oracle Optimization**

Require: $\epsilon$, $L$ > 0 and search space $S$

1: Choose grid resolution = $\frac{2\epsilon}{L}$ and let $X$ := set of all grid points on chosen search space $S$

2: Evaluate $f(x)$ for all $x$ ∈ $X$

3: Output: $\bar{x}$ = arg min<sub>x∈X</sub> $f(x)$

The example for search space $S$ being, $S$ = $\{(x,y): -4 ≤ x ≤ 4, -4 ≤ y ≤ 4\}$ for any two variable function, one can choose search space of their choice. The "grid resolution" refers to the distance between adjacent grid points on the search space. It is essentially a measure of how finely the optimization domain is discretized. The grid resolution is determined by the choice of $\epsilon$, $L$.

1.  Implement the Algorithm 1 for $f(x) = \sqrt{x^2 + 5}$, in $S = \{x: -\frac{A}{10} - 2 ≤ x ≤ \frac{A}{10} + 2\}$ where $A$ is the last digit of your roll number. Take $L$ = 1 and $\epsilon$ ∈ {$10^{-1}$, $10^{-2}$, $10^{-3}$, $10^{-4}$, $10^{-5}$, $10^{-6}$, $10^{-7}$, $10^{-8}$, $10^{-9}$, $10^{-10}$}. Report the values of $\bar{x}$ corresponding to each of the $\epsilon$. Comment on the observations. Comment about the objective function values obtained for different choices of the $\epsilon$ at $\bar{x}$.

2.  Implement the Algorithm 1 for $f(x) = f(x_1, x_2) = \sqrt{x_1^2 + x_2^2}$ in $S = \{(x_1, x_2): -\frac{A}{10} - 2 ≤ x_1 ≤ \frac{A}{10} + 2, -\frac{A}{5} - 2 ≤ x_2 ≤ \frac{A}{5} + 2\}$ where $A$ is your roll number. Take $L$ = 1 and $\epsilon$ ∈ {$10^{-1}$, $10^{-2}$, $10^{-3}$, $10^{-4}$, $10^{-5}$, $10^{-6}$, $10^{-7}$, $10^{-8}$, $10^{-9}$, $10^{-10}$}. Report the values of $\bar{x}$ corresponding to each of the $\epsilon$. Comment on the observations. Comment about the objective function values obtained for different choices of the $\epsilon$ at $\bar{x}$. Plot the level set of this function showing the obtained minima.

3.  What do you observe from (1) and (2), write a general observation related to (1) and (2) and any other general function. Also find the cardinality of the set of all grid points on chosen search space $S$ in terms of $\epsilon$, $L$ assuming any general $d$ variable function, what does this represent?

4.  What may be the shortcomings of this algorithm and suggest a possible solution to deal with it ?

**Exercise 3 (20 marks)** _Gradient descent_ is one of the oldest algorithms (dating back to Cauchy), yet simple and popular in many scientific communities even today. Gradient descent works iteratively where in each iteration a suitable descent direction and an appropriate step length are chosen to update the current iterate.

We will start with a procedure which helps to find a minimizer of the function f(x), x ∈ R<sup>n</sup>.

We will use the following gradient descent type algorithm:

**Algorithm 2 Gradient Descent Procedure with Constant Step Length**

Require: Starting point $x_0$, Tolerance level $\tau$, Step length $\eta$

1: Initialize $k = 0$

2: while $||\nabla f(x_k)||_2 > \tau$ do

3: $x_{k+1} \leftarrow x_k - \eta\nabla f(x_k)$

4: $k \leftarrow k + 1$

5: Output: $x_k$

1. What is the minimizer and minimum function value of $f(x) = f(x_1, x_2) = (a + 1 - x_1)^2 + b \cdot (x_2 - x_1^2)^2$?, Where _a_ is the last digit of your Roll Number and _b_ is 100 if _a_ is an even number, else _b_ is 10. Use these values _a_, _b_ for other problems that follows.

2. With the starting point $x_0 = (-1.5, 1.5)$ and $\eta = 0.001$, we aim to analyze the behavior of the algorithm 2 for different tolerance values. We set $\tau = 10^{-p}$ where $p = 1, 2, ..., 13$. For each $\tau$, record the final minimizer, objective function value at termination, and the number of iterations required for convergence in a tabular form. Generate a plot, illustrating the relationship between the number of iterations and $\tau$ values. Comment on the observations. Comment about the minimizers and objective function values obtained for different choices of the tolerance values.

3. Plot the level sets of the function in (1) and also plot the trajectory of the optimization on the same plot and report your observations. "In optimization, a trajectory refers to the path or sequence of points that a numerical optimization algorithm traverses while iteratively updating the solution in search of an optimal point".

4. What may be the shortcomings of this algorithm and suggests a possible solution to deal with it ?

**Exercise 4 (25 marks)** _Newton's method_, inspired by Isaac Newton's work, is a classic optimization approach. It iteratively updates the current solution by calculating a direction to descend and an optimal step size. This process relies on second-order information, specifically the Hessian matrix. Despite its complexity, Newton's method is widely used and valued in various scientific communities.

We will start with a procedure which helps to find a minimizer of the function $f(x)$, $x \in R^n$.

We will use the following Newton's Method type of algorithm:

**Algorithm 3 Newton's Method**

Require: Starting point $x_0$, Stopping tolerance $\tau$

1: Initialize $k = 0$

2: while $||\nabla f(x_k)||_2 > \tau$ do

3: $x_{k+1} \leftarrow x_k - (\nabla^2 f(x_k))^{-1}\nabla f(x_k)$

4: $k \leftarrow k + 1$

5: Output: $x_k$

1. What is the minimizer and minimum function value of $f(x) = f(x_1, x_2) = (a + 1 - x_1)^2 + b \cdot (x_2 - x_1^2)^2$?, Where $\frac{6-10}{10}$ is the last digit of your Roll Number and _a_ these values _a_, _b_ for other problems that follows.

2. Is the minimizer in (1) unique?, Is it local or global minima?, Is the function _f(x)_ convex?, explain each of them.

3. With the starting point $x_0 = (-1.5, 1.5)$ we aim to analyze the behavior of the algorithm 3 for different tolerance values. We set $\tau = 10^{-p}$ where $p = 1, 2, ..., 20,$ For each $\tau$, record the final minimizer, objective function value at termination, and the number of iterations required for convergence in a tabular form. Generate a plot, illustrating the relationship between the number of iterations and $\tau$ values. Comment on the observations. Comment about the minimizers and objective function values obtained for different choices of the tolerance values. Plot the level sets of the function in (1) and also plot the trajectory of the optimization on the same plot and report your observations. "In optimization, a trajectory refers to the path or sequence of points that a numerical optimization algorithm traverses while iteratively updating the solution in search of an optimal point".

4. Redo (3) by implementing the algorithm 2 (With $\eta = 0.001$) instead of algorithm 3 on the same function _f(x)_ of (1). What do you observe?, Compare algorithm 2 and algorithm 3 based on the results you got on this function _f(x)_ of (1).

5. What may be the shortcomings of this algorithm and suggests a possible solution to deal with it?
