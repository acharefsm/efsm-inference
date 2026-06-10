import pandas as pd
import numpy as np
from deap import gp
from gp.deap_gp import setup_pset
from gp.gp_generalise import infer_output
import matplotlib.pyplot as plt
from gp import deap_gp

'''Creates a sample table with the specified number of input parameters and fills it with random values.
It also creates registers that hold the previous value of each input parameter, with the first value being random.
The input parameters are named i0, i1, ... and the registers are named r0, r1, ... respectively.
The function returns a pandas dataframe with the input parameters and registers as columns.

It expects numInputParameters to be a positive integer, 
parameterRange to be a tuple of two integers specifying the range of random values for the input parameters, 
distinctInputSet to be a list of distinct values to sample from instead of random integers, 
tableLength to be a positive integer specifying the number of rows in the table, and 
random_seed to be an integer.'''

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

''' Returns the true output of a given function for each row in the dataframe.
    It expects the trueFunction to be a string representing a valid function and
    and df to be a pandas dataframe with the input parameters and registers as columns.
    The function returns the dataframe with an additional column 'o' containing the outputs of the function 
    and a register column that holds the previous value of the output for each row.
    The output register is named r{max_reg_num + 1} where max_reg_num is the highest numbered register in the input dataframe.
    The returned dataframe has the columns ordered with input parameters first, then registers, and the output column last.'''

def get_true_function_output(trueFunction, df):
    # build a pset with a dummy output column so arg names are renamed to df columns
    # add an output register to the dataframe 
    regs = df.columns.tolist()
    regs = [r for r in regs if r.startswith('r')]
    regsnums = [int(r[1:]) for r in regs]
    max_reg_num = max(regsnums) if regsnums else -1
    df[f'r{max_reg_num + 1}'] = np.zeros((len(df),), dtype=int)
    #add randomvalue to the first row of the output register to ensure it's not all zeros if the true function uses it
    # df.at[0, f'r{max_reg_num + 1}'] = np.random.randint(1,100)

    # compile the true function and apply it to the dataframe to get the output column

    df['o'] = np.zeros_like(df[f'r{max_reg_num + 1}'])
    pset = deap_gp.setup_pset(df)
    tree = gp.PrimitiveTree.from_string(trueFunction, pset)
    trueFunctionCompiled = gp.compile(tree, pset)
    if f'r{max_reg_num + 1}' not in trueFunction:
        df['o'] = df.drop(columns='o').apply(lambda row: trueFunctionCompiled(**row.to_dict()), axis=1)
        df.loc[1:, f'r{max_reg_num + 1}'] = df['o'][:-1].values
    else:
        for idx in df.index:
            if idx > 0:
                df.at[idx, f'r{max_reg_num + 1}'] = df.at[idx-1, 'o']
            row = df.loc[idx, df.drop(columns='o').columns]   
            df.at[idx, 'o'] = trueFunctionCompiled(**row.to_dict())
    
    # reorder columns to have inputs, then registers, then output
    df = df[[c for c in df.columns if c.startswith('i')] + [c for c in df.columns if c.startswith('r')] + ['o']]
    return df

'''Generates a random function with the specified complexity using the given parameters.
The function is generated recursively by combining randomly chosen parameters and operations. 
If complexity is 0, the function will be a simple operation (add or sub) between two parameters.
If the output register is used as a parameter, only add and sub operations are allowed to avoid infinite growth of the output value.
The function returns a string representing the generated function.

Parameters:
- parameters: a list of strings representing the input parameters and registers that can be used in the function.
- complexity: a non-negative integer representing the complexity of the function. Higher values will result in more complex functions.
- constantProb: a float between 0 and 1 representing the probability of using a constant value instead of a parameter in the function. Default is 0.3.
- maxConstant: an integer representing the maximum value of the constant that can be used in the function. Default is 10.
- random_seed: an integer used to seed the random number generator.
- original_output_register: a string representing the name of the output register, used to avoid using it as a parameter in a way that allows infinite growth.
'''
def setFunction(parameters, complexity, constantProb=0.3, maxConstant=10, random_seed=42, original_output_register=None):
    np.random.seed(random_seed)
    #Added in main code instead of here to avoid duplication 
    # regs = parameters.copy()
    # regs = [r for r in regs if r.startswith('r')]
    # regsnums = [int(r[1:]) for r in regs]
    # max_reg_num = max(regsnums) if regsnums else -1
    # parameters.append(f'r{max_reg_num + 1}') # add output register as a parameter for the function to use
    ops = ['add', 'sub', 'mul']
    constantProb = 1/(len(parameters)+1)
    u = np.random.random_sample()
    if u < constantProb:
        param1 = str(np.random.randint(1, maxConstant + 1))
    else:
        param1 = np.random.choice(parameters)
    # if the output register is used as a parameter, only allow add and sub to avoid infinite growth
    # remove this if you want to allow mul with the output register, but be aware it can lead to very large values and overflow
    if param1 == original_output_register:
        ops = ['add', 'sub']   
    if complexity <= 1:
        if complexity == 0:
            ops = ['add', 'sub']
        else:
            parameters = [p for p in parameters if p != param1]
        param2 = np.random.choice(parameters)
        op = np.random.choice(ops)
        # if the output register is used as a parameter, only allow add and sub to avoid infinite growth
        # remove this if you want to allow mul with the output register, but be aware it can lead to very large values and overflow
        if param2 == original_output_register:
            op = np.random.choice(['add', 'sub']) 
        return op + "(" + param1 + "," + param2 + ")"
    else:
        func = setFunction(parameters, complexity-1, constantProb, maxConstant, original_output_register=original_output_register)
        op = np.random.choice(ops)
        return op + "(" + param1 + "," + func + ")"


'''Generates a random linear function with the specified depth (number of terms).
The function is generated by creating a linear combination of randomly chosen parameters with random coefficients.
The function returns a string representing the generated linear function.
Parameters:
- parameters: a list of strings representing the input parameters and registers that can be used in the function.
- depth: a positive integer representing the number of terms in the linear function.
- maxCoeff: an integer representing the maximum value of the coefficients used in the linear combination. Default is 10.
- random_seed: an integer used to seed the random number generator. Default is 42.'''

def setFunctionLinear(parameters, depth, maxCoeff=10, random_seed=42):
    expr = ""
    np.random.seed(random_seed)
    while depth > 0:
        param = np.random.choice(parameters)
        coeff = np.random.randint(1, maxCoeff + 1)
        plusMinus = np.random.choice(['', '-'])
        newexpr = 'mul(' + plusMinus + str(coeff) + ',' + param + ')'
        if expr == "":
            expr = newexpr
        else:
            expr = 'add(' + expr + ',' + newexpr + ')'
        if (depth == 1):
            u = np.random.random_sample()
            if u < 0.5:
                const = str(np.random.randint(1, maxCoeff + 1))
                expr = 'add(' + expr + ',' + const + ')'
        depth -= 1
    return expr


