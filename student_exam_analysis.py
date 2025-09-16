import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "student_exam_scores.csv"
df = pd.read_csv(file_path)

# Print the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Print the data types and check for missing values
print("\nData types and missing values:")
print(df.info())
print(df.isnull().sum())

# Print basic statistics (mean, median, std) for numerical columns
print("\nBasic statistics (mean, median, std) for numerical columns:")
print(df.describe())

# 1. Average exam score for each attendance percentage group
attendance_group_avg_score = df.groupby('attendance_percent')['exam_score'].mean()
print("\nAverage exam score for each attendance percentage group:")
print(attendance_group_avg_score)

# 2. Optional: Correlation matrix (to see relationships between numeric columns)
numeric_cols_for_corr = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols_for_corr].corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix", fontsize=15)
plt.show()

# Optional: If you want to see the correlation matrix values printed:
print("\nCorrelation matrix between numeric columns:")
print(corr_matrix)

# 3. Optional: Scatter plot of hours studied vs. exam score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='hours_studied', y='exam_score', color='blue')
plt.title('Scatter plot of Hours Studied vs. Exam Score', fontsize=15)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

# 4. Optional: Regression plot of hours studied vs. exam score
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='hours_studied', y='exam_score', scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.title('Regression plot of Hours Studied vs. Exam Score', fontsize=15)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

# 5. Optional: Box plot of exam scores for different attendance percentage ranges
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='attendance_percent', y='exam_score')
plt.title('Box plot of Exam Scores vs. Attendance Percentage', fontsize=15)
plt.xlabel('Attendance Percentage')
plt.ylabel('Exam Score')
plt.xticks(rotation=90)
plt.show()
