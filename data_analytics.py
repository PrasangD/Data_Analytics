# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, ttest_ind

# Reading the dataset
data = pd.read_csv('nike_survey.csv')

# Exploratory Data Analysis
# Plotting age distribution
sns.histplot(data=data, x='age', bins=5, kde=True)
plt.title('Age Distribution')
plt.show()

# Plotting gender distribution
sns.countplot(data=data, x='gender')
plt.title('Gender Distribution')
plt.show()

# Plotting education distribution
sns.countplot(data=data, x='education')
plt.xticks(rotation=30, ha='right')
plt.title('Education Distribution')
plt.show()

# Plotting income distribution
sns.histplot(data=data, x='income', bins=4, kde=True)
plt.title('Income Distribution')
plt.show()

# Plotting advertising campaign awareness and purchase behavior
adv_data = data[['advertising_awareness', 'advertising_purchase']]
adv_data['advertising_awareness'] = np.where(adv_data['advertising_awareness']==1, 'Aware', 'Not aware')
sns.countplot(data=adv_data, x='advertising_awareness', hue='advertising_purchase')
plt.title('Advertising Campaign Awareness and Purchase Behavior')
plt.show()

# Plotting sales promotion usage and purchase behavior
promo_data = data[['sales_promotion_usage', 'sales_promotion_purchase']]
promo_data['sales_promotion_usage'] = np.where(promo_data['sales_promotion_usage']==1, 'Used', 'Not used')
sns.countplot(data=promo_data, x='sales_promotion_usage', hue='sales_promotion_purchase')
plt.title('Sales Promotion Usage and Purchase Behavior')
plt.show()

# Plotting direct marketing email receipt and purchase behavior
dm_data = data[['direct_marketing_email_receipt', 'direct_marketing_purchase']]
dm_data['direct_marketing_email_receipt'] = np.where(dm_data['direct_marketing_email_receipt']==1, 'Received', 'Not received')
sns.countplot(data=dm_data, x='direct_marketing_email_receipt', hue='direct_marketing_purchase')
plt.title('Direct Marketing Email Receipt and Purchase Behavior')
plt.show()

# Plotting event attendance and purchase behavior
event_data = data[['event_attendance', 'event_purchase']]
event_data['event_attendance'] = np.where(event_data['event_attendance']==1, 'Attended', 'Not attended')
sns.countplot(data=event_data, x='event_attendance', hue='event_purchase')
plt.title('Event Attendance and Purchase Behavior')
plt.show()

# Plotting product quality rating and recommendation behavior
quality_data = data[['product_quality_rating', 'product_quality_recommend']]
quality_data['product_quality_rating'] = np.where(quality_data['product_quality_rating']==1, 'Poor', np.where(quality_data['product_quality_rating']==2, 'Fair', np.where(quality_data['product_quality_rating']==3, 'Good', 'Excellent')))
sns.countplot(data=quality_data, x='product_quality_rating', hue='product_quality_recommend')
plt.title('Product Quality Rating and Recommendation Behavior')
plt.show()

# Plotting customer service satisfaction and repeat purchase behavior
service_data = data[['customer_service_satisfaction', 'customer_service_repeat_purchase']]
service_data['customer_service_satisfaction'] = np.where(service_data['customer_service_satisfaction']==1, 'Unsatisfied', 'Satisfied')
sns.countplot(data=service_data, x='customer_service_satisfaction', hue='customer_service_repeat_purchase')
plt.title('Customer Service Satisfaction and Repeat Purchase Behavior')
plt.show()

#Hypothesis Testing
#Chi-square test for advertising campaign awareness and purchase behavior
adv_ct = pd.crosstab(data['advertising_awareness'], data['advertising_purchase'])
_, p_adv, _, _ = chi2_contingency(adv_ct)
print('Advertising Campaign Awareness and Purchase Behavior:')
print('p-value:', p_adv)

#Chi-square test for sales promotion usage and purchase behavior
promo_ct = pd.crosstab(data['sales_promotion_usage'], data['sales_promotion_purchase'])
_, p_promo, _, _ = chi2_contingency(promo_ct)
print('Sales Promotion Usage and Purchase Behavior:')
print('p-value:', p_promo)

#Chi-square test for direct marketing email receipt and purchase behavior
dm_ct = pd.crosstab(data['direct_marketing_email_receipt'], data['direct_marketing_purchase'])
_, p_dm, _, _ = chi2_contingency(dm_ct)
print('Direct Marketing Email Receipt and Purchase Behavior:')
print('p-value:', p_dm)

#Chi-square test for event attendance and purchase behavior
event_ct = pd.crosstab(data['event_attendance'], data['event_purchase'])
_, p_event, _, _ = chi2_contingency(event_ct)
print('Event Attendance and Purchase Behavior:')
print('p-value:', p_event)

#Independent sample t-test for product quality rating and recommendation behavior
good_quality = data[data['product_quality_rating']==3]['product_quality_recommend']
excellent_quality = data[data['product_quality_rating']==4]['product_quality_recommend']
_, p_quality, _ = ttest_ind(good_quality, excellent_quality)
print('Product Quality Rating and Recommendation Behavior:')
print('p-value:', p_quality)

#One-way ANOVA for customer service satisfaction and repeat purchase behavior
unsatisfied_service = data[data['customer_service_satisfaction']==1]['customer_service_repeat_purchase']
satisfied_service = data[data['customer_service_satisfaction']==2]['customer_service_repeat_purchase']
_, p_service, _ = f_oneway(unsatisfied_service, satisfied_service)
print('Customer Service Satisfaction and Repeat Purchase Behavior:')
print('p-value:', p_service)