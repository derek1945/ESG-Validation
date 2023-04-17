# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% read excel file
df = pd.read_excel('factor_volatility.xlsx', sheet_name='Summary')
df.drop(['lt_vol_usd_truncated', 'lt_vol_hkd_truncated'], axis=1, inplace=True)

# %% melt all columns except tenor
df_melted = pd.melt(df, id_vars=['Tenor'], var_name='variable', value_name='Long Term Volatility')
df_melted['fx'] = df_melted['variable'].apply(lambda x: 'USD' if 'usd' in x else 'HKD')


# %% plot
ax = sns.lineplot(data=df_melted, x="Tenor", y="Long Term Volatility", hue="fx", linestyle=(0, (1, 10)))
ax = sns.lineplot(data=df_melted[(df_melted['fx'] == 'USD') & (df_melted['Tenor'] <= 30)], x="Tenor", y="Long Term Volatility", color='blue')
ax = sns.lineplot(data=df_melted[(df_melted['fx'] == 'HKD') & (df_melted['Tenor'] <= 20)], x="Tenor", y="Long Term Volatility", color='orange')
ax.hlines(y=0, xmin=20, xmax=120, linewidth=2, color='orange')
ax.hlines(y=0.0001, xmin=30, xmax=120, linewidth=2, color='blue')
x1 = ax.lines[0].get_xydata()[:30,0]
y1 = ax.lines[0].get_xydata()[:30,1]
x2 = ax.lines[1].get_xydata()[:20,0]
y2 = ax.lines[1].get_xydata()[:20,1]
ax.fill_between(x1, y1, color="blue", alpha=0.2)
ax.fill_between(x2, y2, color="orange", alpha=0.2)


# %%
