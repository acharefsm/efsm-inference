import operator
import random
import pandas as pd
import numpy as np
from gp.gp_generalise import infer_output
from deap import gp
import matplotlib.pyplot as plt



def create_input_samples(numInputParameters, numRegisters, parameterRange=(0, 100), distinctInputSet=None, tableLength=10, random_seed=42):
    np.random.seed(random_seed)
    if distinctInputSet:
        df = pd.DataFrame()
        for r in range(numRegisters):
            df[f'r{r}'] = np.random.choice(distinctInputSet, size=tableLength)
        for i in range(numInputParameters):
            df[f'i{i}']=np.random.choice(distinctInputSet, size=tableLength)
    else:
        df = pd.DataFrame()
        for r in range(numRegisters):
            df[f'r{r}'] = np.random.randint(*parameterRange, size=tableLength)
        for i in range(numInputParameters):
            df[f'i{i}']=np.random.randint(*parameterRange, size=tableLength)
    return df

def get_true_function_output(trueFunction, df):
    trueFunctionCompiled = compile(trueFunction)
    print("true function ", trueFunction)
    print("true function code ", trueFunctionCompiled)
    print("input parameters ", df.columns)
    df['o'] = df.apply(lambda row: trueFunctionCompiled(**row), axis=1)
    return df


def infered_output(trueFunc, inputDf, tableLength = None, numGenerations=100, random_seed=42):
    df = inputDf.copy()
    if tableLength:
        df = df.head(tableLength)
    inferredFuncStr = infer_output(df, ngen=numGenerations, random_seed=42)
    inferredFunc = compile(inferredFuncStr)
    df['inferred'] =df.drop('o', axis=1).apply(lambda args: inferredFunc(**(args.to_dict())), axis=1)
    return inferredFuncStr, df

def create_verified_table(trueFunction, numInputParameters, numRegisters, parameterRange=(0, 100), distinctInputSet=None):
    df = pd.DataFrame()
    for r in range(numRegisters):
        df[f'r{r}'] = np.random.randint(*parameterRange, size=1)

    for i in range(numInputParameters):
        df[f'i{i}']=np.random.randint(*parameterRange, size=1)

    df['o'] = df.apply(lambda row: trueFunction(*row), axis=1)
    inferredFunc = compile(infer_output(df))
    df['inferred'] =df.drop('o', axis=1).apply(lambda args: inferredFunc(**(args.to_dict())), axis=1)
    iteration = 1
    while (iteration < 1000):
        row = {}
        for r in range(numRegisters):
            row[f'r{r}'] = np.random.randint(*parameterRange)

        for i in range(numInputParameters):
            row[f'i{i}'] = np.random.randint(*parameterRange)

        row['o'] = trueFunction(**row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        inferredFunc = compile(infer_output(df.drop(columns=['inferred'])))
        df['inferred'] = df.drop(columns=['o', 'inferred']).apply(lambda row: inferredFunc(**(row.to_dict())), axis=1)
        iteration  += 1
        print(df[['inferred', 'o']])
        if (df['inferred'].values == df['o'].values).all():
            break
    return df

# def compile(expr):
#     code = str(expr)
#     # print("inferred code ", code)
#     code = "lambda {args}: {code}".format(args="r0, r1,r2,r3,r4,i0,i1", code=code)
    
#     return eval(code, {operator.add.__name__: operator.add, operator.sub.__name__: operator.sub, operator.mul.__name__: operator.mul, operator.truediv.__name__: operator.truediv}, {})



# def trueFunc(r0, r1,r2,r3,r4,i0,i1):
#     return r1*3+i1#r0+i1**2


def setFunction(parameters, complexity):
    ops = ['add', 'sub', 'mul']
    if complexity == 1:
        u = np.random.random_sample()
        if u < 0.5:
            param1 = str(np.random.randint(11))
        else:
            param1 = np.random.choice(parameters)
        param2 = np.random.choice(parameters)
        op = np.random.choice(ops)
        return op + "(" + param1 + "," + param2 + ")"
    else:
        func1 = setFunction(parameters, complexity-1)
        func2 = setFunction(parameters, complexity-1)
        op = np.random.choice(ops)
        return op + "(" + func1 + "," + func2 + ")"
    
    

#Create input samples
max_table_length = 50
df = create_input_samples(2, 5, parameterRange=(0, 100), tableLength=max_table_length, random_seed=42)

#Define true function and output samples
complexity = 2
trueFunction = setFunction([d for d in df.columns], complexity)

df = get_true_function_output(trueFunction, df)
    
tableLength = np.arange(10, max_table_length, 10)
numGenerations = np.arange(50, 250, 50)

seed = 42
RMSE = np.zeros((len(tableLength), len(numGenerations)))
i = 0
inferredFuncs= []
for t in tableLength:
    j = 0
    inferredFuncs.append([])
    for g in numGenerations:
        inferredFuncStr, df_t_g = infered_output(trueFunction, df, tableLength=t, numGenerations=g, random_seed=seed)
        inferredFuncs[-1].append(inferredFuncStr)
        RMSE[i, j] = np.sqrt(np.mean((df_t_g['inferred'] - df_t_g['o'])**2))
        # accuracies[t-1] += np.mean((df_t_g['inferred'] - df_t_g['o'])**2)
        print(f"Table Length: {t}, Generations: {g}, RMSE: {RMSE[i, j]}")
        j += 1
    i += 1

plt.figure(figsize=(12, 6))
for i, g in enumerate(numGenerations):
    for j, t in enumerate(tableLength):
        plt.text(t, RMSE[j, i], f'{inferredFuncs[j][i]}', fontsize=8, ha='center', va='bottom')
    plt.plot(tableLength, RMSE[:, i], marker='o', label=f'Generations={g}')
    plt.title(f'RMSE vs # Samples for Complexity {complexity}, function: {trueFunction}')
    plt.xlabel('# Samples')
    plt.ylabel('RMSE')

plt.legend()    
plt.show()








# i = 0
# accuracies = np.zeros(len(numGenerations))
# for g in numGenerations:
#     df = infered_output(trueFunc, 2, 5, 6, parameterRange=(0, 10), numGenerations=g)
#     accuracies[i] = (df['inferred'] == df['o']).mean()
#     i += 1


# plt.plot(tableLength, accuracies)
# plt.title(r'Accuracy vs Table Length $r_0+i_1^2$')
# plt.xlabel('Table Length')  
# plt.xlim(0, (max(tableLength)+1))

# plt.plot(numGenerations, accuracies)
# plt.title(r'Accuracy vs Number of Generations $r_0+i_1^2$')
# plt.ylabel('Accuracy of Inferred Output')
# plt.xlabel('Number of Generations')  
# plt.xlim(min(numGenerations), (max(numGenerations)+50))

# plt.ylim(0, 1)

# plt.show()




# print(create_verified_table(trueFunc, 2, 5, parameterRange=(0, 10)))
# print infered_output(trueFunc, 2, 5, 6))
# print infered_output(trueFunc, 2, 5, 6, distinctInputSet=[0,100, 200]))