import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['BMI'] = df['weight'] / (df['height'] / 100)**2
df['overweight'] = (df['BMI'] > 25).astype(int)
# 3
df['Cholesterol'] = df['Cholesterol'].apply(lambda x: 0 if x==1 else 1)
df['Gluc'] = df['Gluc'].apply(lambda x: 0 if x==1 else 1)



# 4
def draw_cat_plot():
     sns.catplot(x = "cholesterol", y = "cardio", data = df, kind = "bar")
  
    # 5
df_cat = df.melt(id_vars=['id'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'], var_name='variable', value_name='value')



    # 6
df_cat = df.groupby('cardio').agg(count=('cardio', 'count'))
df_cat = df_cat.rename(columns={'count': 'total'})
df_cat = df_cat.reset_index()
df_cat
    

    # 7
df_long = df.melt(id_vars=['id'], value_vars=['sex', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'], var_name='feature', value_name='value')
sns.catplot(data=df_long, x='feature', hue='value', kind='count')



    # 8
fig = plt.figure()


    # 9
fig.savefig('catplot.png')
return fig


# 10
def draw_heat_map():
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax, cmap="coolwarm", square=True)
    plt.show()

draw_heat_map(df)
    # 11
df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    # 12
corr = df.corr()

    # 13
mask = np.triu(np.ones_like(df.corr(), dtype=bool))



    # 14
fig, ax = plt.figure(figsize=(10, 6))

    # 15
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)


    # 16
fig.savefig('heatmap.png')
return fig
