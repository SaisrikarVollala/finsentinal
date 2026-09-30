# Credit Risk Prediction - Dataset Data Dictionary

Dataset: **UCI Default of Credit Card Clients** (Yeh & Lien, 2009)
Task: Predict next-month default (`default_payment_next_month`) for existing credit-card cardholders in Taiwan.

| Column Name | Type | Description | Values / Scale |
|-------------|------|-------------|----------------|
| `LIMIT_BAL` | numerical | Amount of given credit in New Taiwan dollars (NT$), including individual and family/supplementary credit. | [10000, 1000000] |
| `SEX` | categorical | Gender of the cardholder. | {1: 'Male', 2: 'Female'} |
| `EDUCATION` | categorical | Education level of the cardholder. | {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others', 5: 'Unknown', 6: 'Unknown', 0: 'Unknown'} |
| `MARRIAGE` | categorical | Marital status of the cardholder. | {1: 'Married', 2: 'Single', 3: 'Others', 0: 'Unknown'} |
| `AGE` | numerical | Age of the cardholder in years. | [18, 100] |
| `PAY_0` | categorical_ordinal | Repayment status in September 2005 (most recent month). | -2=No consumption/balance, -1=Paid in full, 0=Revolving credit used (min payment), 1=Delay 1 month, 2=Delay 2 months, ..., 8=Delay 8 months, 9=Delay 9+ months |
| `PAY_2` | categorical_ordinal | Repayment status in August 2005. | -2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed |
| `PAY_3` | categorical_ordinal | Repayment status in July 2005. | -2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed |
| `PAY_4` | categorical_ordinal | Repayment status in June 2005. | -2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed |
| `PAY_5` | categorical_ordinal | Repayment status in May 2005. | -2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed |
| `PAY_6` | categorical_ordinal | Repayment status in April 2005 (oldest month). | -2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed |
| `BILL_AMT1` | numerical | Amount of bill statement in September 2005 (NT$). | NT$ |
| `BILL_AMT2` | numerical | Amount of bill statement in August 2005 (NT$). | NT$ |
| `BILL_AMT3` | numerical | Amount of bill statement in July 2005 (NT$). | NT$ |
| `BILL_AMT4` | numerical | Amount of bill statement in June 2005 (NT$). | NT$ |
| `BILL_AMT5` | numerical | Amount of bill statement in May 2005 (NT$). | NT$ |
| `BILL_AMT6` | numerical | Amount of bill statement in April 2005 (NT$). | NT$ |
| `PAY_AMT1` | numerical | Amount paid in September 2005 (towards August statement) in NT$. | NT$ |
| `PAY_AMT2` | numerical | Amount paid in August 2005 (towards July statement) in NT$. | NT$ |
| `PAY_AMT3` | numerical | Amount paid in July 2005 (towards June statement) in NT$. | NT$ |
| `PAY_AMT4` | numerical | Amount paid in June 2005 (towards May statement) in NT$. | NT$ |
| `PAY_AMT5` | numerical | Amount paid in May 2005 (towards April statement) in NT$. | NT$ |
| `PAY_AMT6` | numerical | Amount paid in April 2005 (towards March statement) in NT$. | NT$ |
| `default_payment_next_month` | binary_target | Binary target: Did the customer default on the next month's payment (October 2005)? | {0: 'Non-default', 1: 'Default'} |
