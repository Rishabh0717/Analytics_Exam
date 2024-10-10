import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('sample_customer_data_for_exam.csv')

# 7a. Comprehensive Exploratory Data Analysis (EDA)

# i. Display the first few rows and summary statistics
print(df.head())
print(df.describe())

# ii. Create a heatmap for correlation between numerical variables
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# iii. Create histograms for 'age' and 'income'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.hist(df['age'], bins=20)
ax1.set_title('Distribution of Age')
ax1.set_xlabel('Age')
ax2.hist(df['income'], bins=20)
ax2.set_title('Distribution of Income')
ax2.set_xlabel('Income')
plt.show()

# iv. Box plot of 'purchase amount' across 'product category'
plt.figure(figsize=(12, 6))
sns.boxplot(x='product_category', y='purchase_amount', data=df)
plt.title('Distribution of Purchase Amount by Product Category')
plt.xticks(rotation=45)
plt.show()

# v. Pie chart of customers by gender
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Proportion of Customers by Gender')
plt.show()

# 7b. Analyze relationship between customer characteristics and purchase behavior

# i. Average purchase amount by education level
avg_purchase_by_education = df.groupby('education')['purchase_amount'].mean().sort_values(ascending=False)
print("Average Purchase Amount by Education Level:")
print(avg_purchase_by_education)

# ii. Average satisfaction score by loyalty status
avg_satisfaction_by_loyalty = df.groupby('loyalty_status')['satisfaction_score'].mean().sort_values(ascending=False)
print("\nAverage Satisfaction Score by Loyalty Status:")
print(avg_satisfaction_by_loyalty)

# iii. Bar plot of purchase frequency across regions
plt.figure(figsize=(10, 6))
sns.countplot(x='region', hue='purchase_frequency', data=df)
plt.title('Purchase Frequency Across Regions')
plt.show()

# iv. Percentage of customers who used promotional offers
promo_usage_percentage = (df['promotion_usage'].sum() / len(df)) * 100
print(f"\nPercentage of customers who used promotional offers: {promo_usage_percentage:.2f}%")

# v. Correlation between income and purchase amount
income_purchase_corr = df['income'].corr(df['purchase_amount'])
print(f"\nCorrelation between income and purchase amount: {income_purchase_corr:.2f}")

# 7c. Explore impact of loyalty status and promotion usage on customer behavior

# i. Scatter plot of purchase frequency vs purchase amount, color-coded by loyalty status
plt.figure(figsize=(12, 8))
sns.scatterplot(x='purchase_frequency', y='purchase_amount', hue='loyalty_status', data=df)
plt.title('Purchase Frequency vs Purchase Amount by Loyalty Status')
plt.show()

# ii. Average purchase amount for customers who used promotions vs those who didn't
avg_purchase_promo = df.groupby('promotion_usage')['purchase_amount'].mean()
print("\nAverage Purchase Amount by Promotion Usage:")
print(avg_purchase_promo)

# iii. Violin plot of satisfaction score for different loyalty status groups
plt.figure(figsize=(12, 6))
sns.violinplot(x='loyalty_status', y='satisfaction_score', data=df)
plt.title('Distribution of Satisfaction Score by Loyalty Status')
plt.show()

# iv. Stacked bar chart of promotion usage across product categories
promotion_by_category = df.groupby(['product_category', 'promotion_usage']).size().unstack()
promotion_by_category_percentage = promotion_by_category.div(promotion_by_category.sum(axis=1), axis=0)
promotion_by_category_percentage.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Proportion of Promotion Usage Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Percentage')
plt.legend(title='Promotion Usage', labels=['No', 'Yes'])
plt.show()

# v. Correlation between satisfaction score and purchase frequency
satisfaction_frequency_corr = df['satisfaction_score'].corr(df['purchase_frequency'])
print(f"\nCorrelation between satisfaction score and purchase frequency: {satisfaction_frequency_corr:.2f}")

