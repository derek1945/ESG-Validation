#%%
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Set the seed for reproducibility
np.random.seed(123)

# Generate an array of 10,000 standard normal random variables
data = np.random.uniform(size=10000)

# Convert the array to a dataframe
df = pd.DataFrame(data, columns=['u'])

# %%
displacement = [0, 0.01, 0.1, 0.45]
param = [(-3.6, 0.7),]

# %%
# Create a function to calculate mean and variance of lognormal
def lognormal(param, displacement):
    mu, sigma = param
    mean = np.exp(mu + (sigma ** 2) / 2) - displacement
    variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + (sigma ** 2))
    return mean, variance

# %% calculate target mean and variance
stat = []
target_stat = lognormal(param[0], displacement[0])

# %% create objective function
def create_objective_func(target_stat, displ):
    def objective(x):
        return np.sum((np.array(lognormal(x, displ)) - np.array(target_stat)) ** 2)
    return objective


# %%
for i in range(1, len(displacement)):
    objective = create_objective_func(target_stat, displacement[i])
    result = minimize(objective, x0=param[0], tol=1e-16).x
    param.append(tuple(result))


# %%
for i in range(len(displacement)):
    df[displacement[i]] = lognorm(s=param[i][1], scale=np.exp(param[i][0])).ppf(df['u']) - displacement[i]

df_melted = pd.melt(df, id_vars=['u'], var_name='Displacement', value_name='shifted lognormal')


#%%
sns.displot(
    df_melted, col="Displacement", x="shifted lognormal", kind="hist", fill=True, kde=True, rug=False,
)
plt.gca().set(xlim=(-0.1, 0.2))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# %%
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 1)], x="shifted lognormal", hue="Set", kind="hist", fill=True, kde=True)
plt.gca().set(xlim=(-0.2, 0.2))

# %%
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 2)], x="shifted lognormal", hue="Set", kind="hist", fill=True, kde=True)
plt.gca().set(xlim=(-0.2, 0.2))

# %%
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 3)], x="shifted lognormal", hue="Set", kind="hist", fill=True, kde=True)
plt.gca().set(xlim=(-0.2, 0.2))

# %%
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 4)], x="shifted lognormal", hue="Set", kind="hist", fill=True, kde=True)
plt.gca().set(xlim=(-0.2, 0.2))

# %%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 1)], x="shifted lognormal", hue="Set", kind="kde", fill=True, ax=axs[0, 0])
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 1)], x="shifted lognormal", hue="Set", kind="kde", fill=True, ax=axs[0, 1])
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 1)], x="shifted lognormal", hue="Set", kind="kde", fill=True, ax=axs[1, 0])
sns.displot(df_melted[(df_melted['Set'] == 0) | (df_melted['Set'] == 1)], x="shifted lognormal", hue="Set", kind="kde", fill=True, ax=axs[1, 1])



# %%
