import operator
import random
import pandas as pd
import numpy as np
from sklearn import tree
from deap import gp
from gp.deap_gp import setup_pset
from gp.gp_generalise import infer_output
import matplotlib.pyplot as plt
from gp import deap_gp

def create_input_samples(numInputParameters, parameterRange=(0, 100), distinctInputSet=None, tableLength=10, random_seed=42):
    np.random.seed(random_seed)
    if distinctInputSet:
        df = pd.DataFrame()
        for i in range(numInputParameters):
            df[f'i{i}']=np.random.choice(distinctInputSet, size=tableLength)
            reg = np.zeros_like(df[f'i{i}'])
            reg[0] = np.random.choice(distinctInputSet)
            reg[1:] = df[f'i{i}'][:-1].values
            df[f'r{i}'] = reg 
    else:
        df = pd.DataFrame()
        for i in range(numInputParameters):
            df[f'i{i}']=np.random.randint(*parameterRange, size=tableLength)
            reg = np.zeros_like(df[f'i{i}'])
            reg[0] = np.random.randint(*parameterRange)
            reg[1:] = df[f'i{i}'][:-1].values
            df[f'r{i}'] = reg
    df = df[[c for c in df.columns if c.startswith('i')] + [c for c in df.columns if c.startswith('r')]]
    return df

def get_true_function_output(trueFunction, df):
    # build a pset with a dummy output column so arg names are renamed to df columns
    # add an output register to the dataframe 
    regs = df.columns.tolist()
    regs = [r for r in regs if r.startswith('r')]
    regsnums = [int(r[1:]) for r in regs]
    max_reg_num = max(regsnums) if regsnums else -1
    df[f'r{max_reg_num + 1}'] = np.zeros((len(df),), dtype=int)
    #add randomvalue to the first row of the output register to ensure it's not all zeros if the true function uses it
    df.at[0, f'r{max_reg_num + 1}'] = np.random.randint(1,100)

    # compile the true function and apply it to the dataframe to get the output column

    tmp = df.copy()
    tmp['o'] = np.zeros_like(df[f'r{max_reg_num + 1}'])
    pset = deap_gp.setup_pset(tmp)
    tree = gp.PrimitiveTree.from_string(trueFunction, pset)
    trueFunctionCompiled = gp.compile(tree, pset)

    df['o'] = df.apply(lambda row: trueFunctionCompiled(**row.to_dict()), axis=1)

    df.loc[1:, f'r{max_reg_num + 1}'] = df['o'][:-1].values

    # reorder columns to have inputs, then registers, then output
    df = df[[c for c in df.columns if c.startswith('i')] + [c for c in df.columns if c.startswith('r')] + ['o']]
    return df

def setFunction(parameters, complexity, constantProb=0.3, maxConstant=10):
    regs = parameters.copy()
    regs = [r for r in regs if r.startswith('r')]
    regsnums = [int(r[1:]) for r in regs]
    max_reg_num = max(regsnums) if regsnums else -1
    parameters.append(f'r{max_reg_num + 1}') # add output register as a parameter for the function to use
    ops = ['add', 'sub', 'mul']
    u = np.random.random_sample()
    if u < constantProb:
        param1 = str(np.random.randint(1, maxConstant + 1))
    else:
        param1 = np.random.choice(parameters)

    if complexity <= 1:
        if complexity == 0:
            ops = ['add', 'sub']
        else:
            parameters = [p for p in parameters if p != param1]
        param2 = np.random.choice(parameters)
        op = np.random.choice(ops)
        return op + "(" + param1 + "," + param2 + ")"
    else:
        func = setFunction(parameters, complexity-1, constantProb, maxConstant)
        op = np.random.choice(ops)
        return op + "(" + param1 + "," + func + ")"

def setFunctionLinear(parameters, depth, maxCoeff=10):
    ops = ['add', 'sub', 'mul']
    expr = ""
    while depth > 0:
        op = np.random.choice(ops)
        param = np.random.choice(parameters)
        if op == 'mul':
            coeff = np.random.randint(1, maxCoeff + 1)
            plusMinus = np.random.choice(['', '-'])
            newexpr = plusMinus + str(coeff) + "*" + param
        else:
            param1 = np.random.choice(parameters)
            param2 = np.random.choice(parameters)
            newexpr = param1 + op + param2
            depth -= 1
        if expr == "":
            expr = newexpr
        else:
            expr = expr + "+" + newexpr
    return expr

def compile(expr, df):
    code = str(expr)
    args = ", ".join(df.columns)
    code = "lambda {args}: {code}".format(args=args, code=code)

    return eval(code, {operator.add.__name__: operator.add, operator.sub.__name__: operator.sub, operator.mul.__name__: operator.mul, operator.truediv.__name__: operator.truediv}, {})

def infer_w_gp(df, ngen=100, random_seed=42):
    tmp = df.drop(columns=['o']).copy()
    print("cols: ", df.columns.tolist())
    pset = deap_gp.setup_pset(df)
    inferredFuncStr = infer_output(df, ngen=ngen, random_seed=random_seed)
    tree = gp.PrimitiveTree.from_string(inferredFuncStr, pset)
    inferredFunctionCompiled = gp.compile(tree, pset)
    df['inferred'] = tmp.apply(lambda row: inferredFunctionCompiled(**row.to_dict()), axis=1)
    return inferredFuncStr, df


sampledFuncs = 1
max_table_length = 50
tableLength = np.arange(30, max_table_length+1, 10)
#tableLength = [50]
numGenerations = [100, 200]
#numGenerations = np.arange(50, 251, 50)
seed = 0
linear = False
complexity = 3
NRMSE = np.zeros((len(tableLength), len(numGenerations)))
outputs = np.zeros((max_table_length, len(numGenerations)))

if sampledFuncs == 1: 
    inferredFuncs = []
for s in range(sampledFuncs):
    random_seed = seed
    #Create input samples
    complete_df = create_input_samples(numInputParameters=2, parameterRange=(0, 100), tableLength=max_table_length, random_seed=seed)

    #Define true function and output samples
    if linear:
        trueFunction = setFunctionLinear([d for d in complete_df.columns], complexity)
    else:
        trueFunction = setFunction([d for d in complete_df.columns], complexity)
    print(f"True function: {trueFunction}")
    complete_df = get_true_function_output(trueFunction, complete_df)
    complete_df['inferred'] = np.zeros(len(complete_df))


    # plotting the results for different table lengths and number of generations  
    
    for i, t in enumerate(tableLength):
        df = complete_df.head(t).copy()
        if sampledFuncs == 1:
            inferredFuncs.append([])
        for j, g in enumerate(numGenerations):
            func, df = infer_w_gp(df.drop(columns=['inferred']), ngen=g, random_seed=seed)
            NRMSE[i, j] += np.sqrt(np.mean((df['inferred'] - df['o'])**2))/np.mean(np.abs(df['o']))
            if sampledFuncs == 1:
                inferredFuncs[-1].append(func)
            if t == max_table_length:
                print(inferredFuncs[-1][-1])
                outputs[:, j] = df['inferred'].values
     

NRMSE = NRMSE / sampledFuncs

if sampledFuncs == 1:
    plt.figure(figsize=(12, 6))
    for i, g in enumerate(numGenerations):
        for j, t in enumerate(tableLength):
            plt.text(t, NRMSE[j, i], f'{inferredFuncs[j][i]}', fontsize=8, ha='center', va='bottom')
        plt.plot(tableLength, NRMSE[:, i], marker='o', label=f'Generations={g}')
        plt.title(f'NRMSE vs # Samples for Complexity {complexity}, true function: {trueFunction}')
        plt.xlabel('# Samples')
        plt.ylabel('NRMSE')
else:
    plt.figure(figsize=(12, 6))
    for i, g in enumerate(numGenerations):
        plt.plot(tableLength, NRMSE[:, i], marker='o', label=f'Generations={g}')
        #plt.title(f'NRMSE vs # Samples for different number of generations and function {trueFunction}')
        plt.title(f'NRMSE vs # Samples for different number of generations and function complexity {complexity}, averaged over {sampledFuncs} functions')
        plt.xlabel('# Samples')
        plt.ylabel('NRMSE')

plt.legend()    
plt.show()

errors = outputs
print(errors.shape)
plt.figure(figsize=(12, 6))
for i, g in enumerate(numGenerations):
    errors[:, i] = errors[:, i] - complete_df['o'].values
    plt.plot(np.arange(max_table_length), errors[:, i], label=f'Generations={g}, inferred func: {inferredFuncs[-1][i] if sampledFuncs == 1 else ""}')
plt.title(f'Error over time for Complexity {complexity}, true function: {trueFunction}, table length: {max_table_length}')
plt.xlabel('time')
plt.ylabel('error')
plt.legend()    
plt.show()

variance_over_time = np.var(errors, axis=0)
print("Variance of error over time: ", variance_over_time)

deTrended_errors = errors - np.mean(errors, axis=0)
variance_deTrended_over_time = np.var(deTrended_errors, axis=0)
print("Variance of de-trended error over time: ", variance_deTrended_over_time)

#Histogram of errors
# plt.figure(figsize=(12, 6))
# for i, g in enumerate(numGenerations):
#     plt.hist(errors[:, i], bins=20, alpha=0.5, label=f'Generations={g}, inferred func: {inferredFuncs[-1][i] if sampledFuncs == 1 else ""}')
# plt.title(f'Histogram of errors for Complexity {complexity}, true function: {trueFunction}, table length: {max_table_length}')
# plt.xlabel('error')
# plt.ylabel('frequency')
# plt.legend()
# plt.show()