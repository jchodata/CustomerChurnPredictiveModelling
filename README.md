<h1>Predicting Customer Churn</h1> <h2>Description</h2> This project aims to develop a predictive model to forecast customer churn. In the subscription-based industry, customer churn refers to the phenomenon where customers stop using a service. Understanding and predicting churn can help businesses develop targeted strategies to retain customers.<br /><br /> <h2>Dataset Overview</h2> The dataset used for this project contains several key features related to customer behavior, subscription details, and monthly activity. These features are used to predict whether a customer is likely to churn.<br /><br />
<b>Customer Information</b>: Account age, gender, parental control enabled, and subtitle enabled.<br />
<b>Viewing Behavior</b>: Viewing hours per week, average viewing duration, and watchlist size.<br />
<b>Subscription Details</b>: Subscription type, payment method, and total charges.<br />
<b>Monthly Activity</b>: Content downloads per month, support tickets per month, and monthly charges.<br /><br />
<h2>Key Data Insights</h2> - 
<b>Account Age</b>: Churned customers tend to have shorter account ages.
<img src="https://i.imgur.com/3WQiP2m.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> - <b>Monthly Charges</b>: Churned customers typically have higher monthly charges.
<img src="https://i.imgur.com/UNnz5Jg.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> - <b>Subscription Type</b>: Customers on basic plans are more likely to churn.
<img src="https://i.imgur.com/5nqq2Qv.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> - <b>Viewing Behavior</b>: Higher viewing hours are associated with lower churn rates.
<img src="https://i.imgur.com/zHrH5Xn.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> - <b>Payment Method</b>: Churned customers are more likely to use checks for payment.
<img src="https://i.imgur.com/w2MFQNf.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> - <b>Content Downloads</b>: Churned customers download less content per month.
<img src="https://i.imgur.com/JJEqQtK.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> - <b>Support Tickets</b>: Customers with higher support ticket submissions are more likely to churn.<br />
<img src="https://i.imgur.com/VGVWw73.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<br /> <h2>Model Evaluation Summary</h2> The following models were evaluated based on their accuracy, precision, recall, and F1 score. The model using the top 19 variables with MinMax scaling performed the best.<br /><br />
| Model | Variables | Accuracy | Precision | Recall | F1 Score |<br />  | Top 19 MinMaxScaler | 19 | 0.6842 | 0.3233 | 0.6783 | 0.4379 |<br /> | Top 19 StandardScaler | 19 | 0.6847 | 0.3235 | 0.6766 | 0.4377 |<br /> | Top 19 RobustScaler | 19 | 0.6836 | 0.3227 | 0.6781 | 0.4373 |<br /> | Top 18 MinMaxScaler | 18 | 0.6836 | 0.3228 | 0.6782 | 0.4374 |<br /> | Top 20 MinMaxScaler | 20 | 0.6840 | 0.3232 | 0.6788 | 0.4379 |<br /><br />

<h2>Languages and Tools Used</h2> - <b>Python</b>: For data analysis and model training.<br /> - <b>Scikit-learn</b>: For model building and evaluation.<br /><br /> <h2>Conclusion</h2> The model successfully identified key factors contributing to customer churn, including account age, monthly charges, and subscription type. Businesses can leverage these insights to improve customer retention by focusing on these areas.<br /><br />