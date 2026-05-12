IFRS 9 Expected Credit Loss

Dataset
Real US loan records from LendingClub (Kaggle), cleaned to 1,179,687 loans issued from 2007–2018, worth 17 billion $ in total. Each record includes loan amount, interest rate, credit grade, borrower income, debt levels, and whether the loan was ultimately repaid or defaulted. Given that the US loan dataset was used, the four macroeconomic variables were pulled from the US Federal Reserve's public database and joined to each loan: unemployment rate, GDP growth, interest rates, and house prices.

What and why
Banks lend money and some borrowers do not pay it back. The accounting standard IFRS 9 requires banks to estimate how much money they expect to lose on their loan portfolios, not after the losses actually happen, but in advance, as a provision set aside on the balance sheet. This estimate is called Expected Credit Loss (ECL).
To calculate ECL for a single loan, three questions need to be answered. First, how likely is the borrower to stop paying (probability of default, or PD)? Second, if they do stop paying, how much of the money will actually be lost? For unsecured consumer loans (UCL), this is the loss given default. And third, how much is still outstanding at that point? The ECL figure is calculated by multiplying these three variables together to estimate the loss on one loan. After repeating this for each loan and summing the results, the total is the ECL figure a bank would report.
The complication is that IFRS 9 does not allow banks to use only historical data. The estimate has to reflect what the bank expects to happen in the future, meaning it must be adjusted for economic conditions going forward. This is where the main finding of my project comes from.

Methodology and steps
1.	Data cleaning. Loans without a clear outcome (still active or too recent to judge) were removed, loans that had not had at least two years to mature were filtered out, and data quality issues such as extreme income values were corrected. 
2.	Probability of default model. I trained a logistic regression model on loans issued before 2016 and tested it strictly on loans from 2016 onwards. I used 19 features, including credit grade, interest rate, debt-to-income ratio, and the macroeconomic variables. A gradient boosting model was trained as a benchmark. The logistic regression model was chosen for its transparency and auditability at a negligible cost in accuracy. Raw predicted probabilities were then calibrated to better match observed default rates. 
3.	Loss given default. I calculated the actual fraction of money lost on the 234,000 historical defaults and estimated average loss rates by loan grade and loan purpose. 
4.	Exposure at default. I used the standard loan repayment formula to project the outstanding balance of each loan forward in time from the evaluation date. 
5.	IFRS 9 staging. Each loan was classified into one of three buckets: healthy loans (12-month loss estimate applied), loans showing signs of deterioration (lifetime estimate applied), and already-defaulted loans (lifetime estimate applied). 
6.	Forward-looking macro overlay. Three economic scenarios were built: baseline, adverse, and severe, and adjusted default probability estimates accordingly. The data-driven overlay decreased ECL under stress, which is the opposite of what economic theory predicts. The cause was Simpson's paradox in the data, LendingClub progressively loosened its lending standards as the economy recovered between 2009 and 2016, meaning years with high unemployment actually had lower default rates simply because the lending bar was higher at the time. The model learned this pattern from the data and reproduced it in the stress scenarios. A second overlay was built using standard regulatory stress-test coefficients (from US and EU frameworks), which correctly increase ECL under stress and are the recommended reporting figure. 
7.	Validation. I tested the model for accuracy, calibration, and stability across different loan vintages and scenario assumptions. The framework is fully reproducible. Re-running all scripts regenerates every number to within 0.01$.


Results:
| Scenario | ECL | % of portfolio |
|---|---:|---:|
| Baseline (no macro adjustment) | 403.5M $ | 2.37% |
| Data-driven overlay | 390.8M $ | 2.30% |
| Regulatory overlay (recommended) | 484.5M $ | 2.85% |
		
The recommended IFRS 9 reporting figure is 484.5m $, a 20% uplift over the baseline reflecting stressed economic conditions. ECL coverage climbs monotonically from 0.24% for the safest borrowers (Grade A) to 8.59% for the riskiest (Grade G), confirming the model ranks risk in the correct direction.

