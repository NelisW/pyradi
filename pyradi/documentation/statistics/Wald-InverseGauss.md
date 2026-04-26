### Overview of Kurtosis

**Kurtosis** is a statistical measure that describes the shape of a probability distribution's tails relative to its center. While often associated with the "peakedness" of a curve, it more accurately measures the presence of outliers.

* **Leptokurtic (High Kurtosis):** Heavy tails and a sharp center. This indicates that the variance is driven by infrequent, extreme deviations.
* **Platykurtic (Low Kurtosis):** Thin tails and a flatter top. This indicates that extreme outliers are rare.
* **Mesokurtic:** A distribution with kurtosis identical to a normal distribution (excess kurtosis of 0).



---

### The Wald Distribution (Inverse Gaussian)

The **Wald distribution** is a continuous probability distribution with support on $(0, \infty)$. It is a special case of the **Inverse Gaussian distribution**. 

Physically, it represents the **First Passage Time** of a Brownian motion (a random walk) with a positive drift to reach a fixed positive threshold. Essentially, if you are tracking a particle moving randomly but trending in one direction, the Wald distribution tells you when it will likely hit a certain "finish line."

#### Real-World Applications
* **Finance:** Modeling the time until a stock price hits a certain "stop-loss" or "take-profit" limit.
* **Psychology:** Modeling "Response Time" in cognitive tasks (how long it takes a brain to accumulate enough evidence to make a decision).
* **Reliability Engineering:** Modeling the lifetime of components that degrade over time until they reach a failure threshold.

---

### Mathematical Definition

The Probability Density Function (PDF) of the Wald distribution is defined by two parameters: the **mean ($\mu$)** and the **shape parameter ($\lambda$)**.

$$f(x; \mu, \lambda) = \sqrt{\frac{\lambda}{2\pi x^3}} \exp \left( -\frac{\lambda(x-\mu)^2}{2\mu^2 x} \right)$$

for $x > 0$. As $\lambda$ increases, the distribution becomes more symmetric and begins to resemble a normal distribution.



---

### Moments and Their Relationships

In the Wald distribution, the higher-order moments (skewness and kurtosis) are directly linked to the relationship between the mean and the shape parameter.

| Metric | Equation |
| :--- | :--- |
| **Mean** | $E[X] = \mu$ |
| **Variance** | $Var(X) = \frac{\mu^3}{\lambda}$ |
| **Skewness** | $\gamma_1 = 3 \sqrt{\frac{\mu}{\lambda}}$ |
| **Excess Kurtosis** | $\gamma_2 = \frac{15\mu}{\lambda}$ |

#### How They Are Related
There is a beautiful mathematical nesting in these parameters. You can derive the "tailedness" and "asymmetry" of the Wald distribution simply by looking at the ratio of the mean to the shape:

1.  **The Base Ratio:** Let $R = \frac{\mu}{\lambda}$.
2.  **Variance** is the mean multiplied by the mean-shape ratio: $\mu \cdot (\mu R)$.
3.  **Skewness** is exactly $3\sqrt{R}$.
4.  **Excess Kurtosis** is exactly $15R$.

#### Determining Kurtosis from Parameters
If you have the basic parameters $\mu$ and $\lambda$, the excess kurtosis is found by a simple linear scaling:
$$\gamma_2 = 15 \left( \frac{\mu}{\lambda} \right)$$

This reveals a fixed relationship between skewness and kurtosis for any Wald distribution: **the excess kurtosis is always exactly $\frac{5}{3}$ times the square of the skewness.**

$$\gamma_2 = \frac{5}{3} \gamma_1^2$$