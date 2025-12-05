"""
Simple Z3 pointwise function example from Gemini.
PROMPT:
I am trying to use z3 from python to infer a pointwise function from inputs to outputs. I have some examples as follows:

f(1, 2, 3) = 3

f(4, 5, 6) = 9

f(2, 3, 4) = 5

How can I get f?
"""

from z3 import *

# 1. Declare the uninterpreted function f
# f takes three Integer arguments and returns an Integer
f = Function("f", IntSort(), IntSort(), IntSort())

# 2. Create a Solver instance
s = Solver()

# 3. Add the examples as constraints (assertions)
s.add(f(1, 2) == 3)
s.add(f(4, 5) == 9)
s.add(f(2, 3) == 5)

# 4. Check for satisfiability
if s.check() == sat:
    # 5. Get the model (the inferred function f)
    m = s.model()

    # 6. Print the model's interpretation of f
    print("Solver found a model (interpretation for f):")
    print(m)

    # 7. Evaluate f for the known examples and a new input

    # Known examples:
    # print(f"\nf(1, 2, 3) = {m.evaluate(f(1, 2, 3))}")
    # print(f"f(4, 5, 6) = {m.evaluate(f(4, 5, 6))}")
    # print(f"f(2, 3, 4) = {m.evaluate(f(2, 3, 4))}")

    # New input (Z3 will assign a default value, often 0)
    # print(f"f(0, 0, 0) = {m.evaluate(f(0, 0, 0))}")

else:
    print("The assertions are unsatisfiable (this should not happen with only equality constraints).")
