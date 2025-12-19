"""
This module contains Luca's repair function and auxilliary functions.
"""

import ast
import re

import patsy
import statsmodels
import statsmodels.formula.api as smf
from deap import gp
from patsy import EvalEnvironment

op_mapp = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
}

cmpop_map = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne_",
}

allowed_primitives = {
                "add", "sub", "mul", "div", "pow", 
                "lt", "le", "gt", "ge", "eq", "ne",
                "and", "or", "not"
            }


def recurse(node):
    if isinstance(node, ast.BinOp):
        op = op_mapp[type(node.op)]
        return f"{op}({recurse(node.left)}, {recurse(node.right)})"
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"-{recurse(node.operand)}"
        return recurse(node.operand)
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            op = "and_"
        elif isinstance(node.op, ast.Or):
            op = "or_"
        else:
            raise NotImplementedError(node.op)

        args = ", ".join(recurse(v) for v in node.values)
        return f"{op}({args})"
    elif isinstance(node, ast.Compare):
        left = recurse(node.left)
        result = []

        for op, right in zip(node.ops, node.comparators):
            cmp = cmpop_map[type(op)]
            result.append(f"{cmp}({left}, {recurse(right)})")
            left = recurse(right)
        if len(result) == 1:
            return result[0]
        return f"and({', '.join(result)})"
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        # Handle inner I(...) wrappers
        if isinstance(node.func, ast.Name) and node.func.id == "I":
            return recurse(node.args[0])
        elif isinstance(node.func, ast.Name) and node.func.id in allowed_primitives:
            args = ", ".join(recurse(a) for a in node.args)
            return f"{node.func.id}({args})"
        raise NotImplementedError(node)
    else:
        raise NotImplementedError(node)


def infix_to_prefix2(expr):
    """
    Convert an infix arithmetic expression like '(r0 + r2 * 10)'
    into DEAP prefix notation: add(r0, mul(r2, 10))
    """
    tree = ast.parse(expr, mode="eval")
    return recurse(tree.body)


def split(individual, creator):
    if len(individual) > 1:
        terms = []
        # Recurse over children if add/sub
        if individual[0].name in ["add", "sub"]:
            terms.extend(
                split(
                    creator.Individual(
                        gp.PrimitiveTree(
                            individual[individual.searchSubtree(1).start : individual.searchSubtree(1).stop]
                        )
                    ),
                    creator,
                )
            )
            terms.extend(
                split(creator.Individual(gp.PrimitiveTree(individual[individual.searchSubtree(1).stop :])), creator)
            )
        else:
            terms.append(individual)
        return terms
    return [individual]


def repair(individual, data_points, pset, creator):
    if pset.ret != bool:
        return individual
    if data_points.iloc[:, -1].dtype == "int64":
        eq = f"y ~ {' + '.join(str(x) for x in split(individual, creator))}"
        data_points.rename(columns={data_points.columns[-1]: "y"}, inplace=True)
        data_points = data_points.astype(float)

        pattern_mul = r"mul\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
        while re.search(pattern_mul, eq):
            eq = re.sub(pattern_mul, r"(\1 * \2)", eq)

        pattern_add = r"add\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
        while re.search(pattern_add, eq):
            eq = re.sub(pattern_add, r"(\1 + \2)", eq)

        pattern_sub = r"sub\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
        while re.search(pattern_sub, eq):
            eq = re.sub(pattern_sub, r"(\1 - \2)", eq)

        # If both sides constant e.g 1000 + 10 then evaluate and replace with constant e.g 1010
        match = re.search(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\*/\+\-])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", eq)
        if match:
            left, op, right = match.groups()
            result = eval(f"{left} {op} {right}")
            # print(result)
            eq = re.sub(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\*/\+\-])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", str(result), eq)
            if re.match(r"^\s*y\s*~\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*$", eq):
                return individual

        # Add I() to right/left side constant and left/right side variable e.g 1000 + r2
        eq = re.sub(r"(\b[A-Za-z_]+\d*\b)\s*([\*/\+\-])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", r"I(\1 \2 \3)", eq)
        eq = re.sub(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\*/\+\-])\s*(\b[A-Za-z_]+\d*\b)", r"I(\1 \2 \3)", eq)
        # constant in parentheses
        eq = re.sub(r"(\b[A-Za-z_]+\d*\b)\s*([\*/\+\-])\s*\((?<![A-Za-z_])(-?\d+\.?\d*)\)", r"I(\1 \2 \3)", eq)
        eq = re.sub(r"\((?<![A-Za-z_])(-?\d+\.?\d*)\)\s*([\*/\+\-])\s*(\b[A-Za-z_]+\d*\b)", r"I(\1 \2 \3)", eq)

        # Add I() to right/left side constant and left/right side expression I() e.g 1000 * I(10 + r2)
        eq = re.sub(r"\(?\s*I\(([^()]*)\)\s*\)?\s*([\+\-\*/])\s*(?<![A-Za-z_])(\d+\.?\d*)", r"I(\1 \2 \3)", eq)
        eq = re.sub(r"(?<![A-Za-z_])(\d+\.?\d*)\s*([\+\-\*/])\s*\(?\s*I\(([^()]*)\)\s*\)?", r"I(\1 \2 \3)", eq)

        # Add I() to right/left side constant and left/right side expression () e.g 1000 * (r0 + r2)
        eq = re.sub(r"(\([^()]+\))\s*([\+\-\*/])\s*(?<![A-Za-z_])(-?\d+\.?\d*)", r"I((\1) \2 \3)", eq)
        eq = re.sub(r"(?<![A-Za-z_])(-?\d+\.?\d*)\s*([\+\-\*/])\s*(\([^()]+\))", r"I(\1 \2 (\3))", eq)

        env = EvalEnvironment.capture()

        try:
            # Create model, fit (run) it, give estimates from it]
            model = smf.ols(eq, data_points, eval_env=env)
            res = model.fit()

            if "Intercept" in res.params:
                eqn = f"{int(round(res.params['Intercept']))}"
            else:
                eqn = "0"
            for term, coefficient in res.params.items():
                if term != "Intercept":
                    if ":" in term:
                        parts = term.split(":")
                        term = "mul(" + ", ".join(parts) + ")"
                    term = re.sub(r"I\((.*?)\)", r"(\1)", term)

                    term = infix_to_prefix2(term)

                    eqn = f"add({eqn}, mul({int(round(coefficient))}, {term}))"
            repaired = type(individual)(gp.PrimitiveTree.from_string(eqn, pset))
            return repaired
        except (
            UnboundLocalError,
            SyntaxError,
            TypeError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
            statsmodels.tools.sm_exceptions.MissingDataError,
            patsy.PatsyError,
            np.core._exceptions._UFuncOutputCastingError,
            np.linalg.LinAlgError,
        ) as e:
            return individual
    else:
        return individual
    
if __name__ == "__main__":
    print(infix_to_prefix2("(le(mul(sub(mul(r2, i0), sub(r0, r0)), mul(r2, r2)), add(add(add(r0, i0), add(r2, r1)), add(sub(r0, r2), sub(r2, r0)))) and ne(i0, r0))"))
