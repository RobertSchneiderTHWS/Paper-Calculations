import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms

# Data for negative and positive framing participants
negative_points = np.array([3,5,5,6,6,6,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,10,10])
positive_points = np.array([2,4,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,9,9,9,10])

# Mean (Average)
mean_negative = np.mean(negative_points)
mean_positive = np.mean(positive_points)

# Standard Deviation
std_dev_negative = np.std(negative_points, ddof=1)
std_dev_positive = np.std(positive_points, ddof=1)

# Shapiro-Wilk Test for normality
shapiro_test_negative = stats.shapiro(negative_points)
shapiro_test_positive = stats.shapiro(positive_points)

# T-test
t_test_results = stats.ttest_ind(negative_points, positive_points, equal_var=False)

# Effect Size (Cohen's d)
cohen_d = (mean_negative - mean_positive) / (np.sqrt((std_dev_negative**2 + std_dev_positive**2) / 2))

# Confidence Interval for Mean Difference
cm = sms.CompareMeans(sms.DescrStatsW(negative_points), sms.DescrStatsW(positive_points))
ci = cm.tconfint_diff(usevar='unequal')

# Output results
print(f'Mean (Negative): {mean_negative:.3f}')
print(f'Mean (Positive): {mean_positive:.3f}')
print(f'Standard Deviation (Negative): {std_dev_negative:.3f}')
print(f'Standard Deviation (Positive): {std_dev_positive:.3f}')
print(f'Shapiro-Wilk Test (Negative): Statistic={shapiro_test_negative[0]:.3f}, P-value={shapiro_test_negative[1]:.4f}')
print(f'Shapiro-Wilk Test (Positive): Statistic={shapiro_test_positive[0]:.3f}, P-value={shapiro_test_positive[1]:.4f}')
print(f'T-test: Statistic={t_test_results.statistic:.3f}, P-value={t_test_results.pvalue:.4f}')
print(f'Cohen\'s d: {cohen_d:.3f}')
print(f'95% confidence interval for the difference in means: {ci}')
