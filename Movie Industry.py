#Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import and read data
df = pd.read_csv(r"C:\Users\nuzha\Downloads\movies.csv")  # Adjust the path as needed
df.head(5)


# Drop NA values
df.dropna(subset=['rating', 'budget', 'gross'], inplace=True)


# Change data types of votes, budget, and gross columns
df = df.astype({"votes": "int64", "budget": "int64", "gross": "int64"})

# Drop company duplicates
df['company'].drop_duplicates(inplace=True)

# Order by gross
df.sort_values(by=['gross'], inplace=True, ascending=False)

# Scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Revenue')
plt.ylim([0, 2*10**9])
plt.xlabel('Gross Revenue')
plt.ylabel('Budget for Films')
plt.show()


# Regplot budget vs gross
sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color':'red'}, line_kws={'color':'blue'})
plt.ylim([0, 2*10**9])
plt.show()


# Heatmap of correlation matrix
corr_mat = df.corr(method='pearson')
sns.heatmap(corr_mat, annot=True)
plt.title('Correlation Metric for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# Numerization of Object Data Types
df_numerized = df.copy()
for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

df_numerized.head()


## Heatmap of numerized correlation matrix
correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Metric for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()
