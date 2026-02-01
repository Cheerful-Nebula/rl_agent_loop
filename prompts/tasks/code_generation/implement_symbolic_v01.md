# TASK: Compile Mathematical Specification to Python

You are a **Neuro-Symbolic Compiler**. Your job is to translate a mathematical reward formula into valid, high-performance Python code.

## The Source Material
**Researcher's Formula:**
{plan}

## The Variable Map (Reference)
You must map the mathematical symbols to the `observation` vector correctly:
* $x$       $\rightarrow$ `observation[0]`
* $y$       $\rightarrow$ `observation[1]`
* $v_x$     $\rightarrow$ `observation[2]`
* $v_y$     $\rightarrow$ `observation[3]`
* $\theta$  $\rightarrow$ `observation[4]` (Angle)
* $\omega$  $\rightarrow$ `observation[5]` (Angular Velocity)
* $L_{{1,2}}$ $\rightarrow$ `observation[6]`, `observation[7]` (Leg Contacts)

## Compiler Constraints (The "Hard" Rules)

1.  **Fidelity:** Implement the formula **EXACTLY** as written.
    * If the Researcher writes $R = -|x|$, your code must be `reward = -abs(x)`.

2.  **Safety:**
    * Use `np.clip` if the formula involves exponentials ($e^x$) to prevent overflow.
    * Ensure the final return value is a single `float`.

3.  **Output:** Return ONLY the updated `calculate_reward` function.

## Current Implementation (For Reference Only)
Below is the old code. You are modifying and/or replacing this entirely depending on the Researcher's formula but you must keep the function signature the same.
```python
{current_code}
```