'''
The following script is a template to conduct an A/B test on the file "AB_trial_data.csv"
It is expected that "AB_trial_data.csv" is saved in the same directory as this script.
Before running, make sure that the following modules in the import section have been installed to the
project's interpreter using "conda install [package]" or "pip install [package]", i.e. "pip install pandas"
Much of the code has been provided, but several lines or functions require completion for the assignment. These lines
and associated comments/hints can be found by searching for the string "#####".
'''

import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime


pd.options.display.max_columns = 10     # sets viewing configuration for easier readability in the console


# ========== FUNCTION DEFINITIONS ==========
def calc_proportion(array_TF):
    return sum(array_TF)/len(array_TF)


def calc_zscore(phat, p, n_f):
#    z_score_f = None    ##### Replace None with formula
    z_score_f = (phat-p) / (p*(1-p)/n_f)**0.5
    return z_score_f


def get_z_crit_value(alpha_f, num_sides_f):
#    z_crit_value_f = None   ##### Replace None with formula; hint: use norm.ppf package
    z_crit_value_f = norm.ppf(alpha_f/num_sides_f)
    return z_crit_value_f


def get_p_value(zscore_f, num_sides_f):
#    p_value_f = None   ##### Replace None with formula; hint: use norm.cdf package
    p_value_f = (1-norm.cdf(zscore_f))*num_sides_f
    return p_value_f


def reject_null(variantA_outcomes_f, variantB_outcomes_f, alpha_f, num_sides_f):
    p_hat_f = calc_proportion(variantB_outcomes_f)
    p_f = calc_proportion(variantA_outcomes_f)
    n_f = len(variantB_outcomes_f)
    z_score = calc_zscore(p_hat_f, p_f, n_f)
    p_value = get_p_value(z_score, num_sides_f)
    z_crit = get_z_crit_value(alpha_f, num_sides_f)
#    reject_null_TF_f = None     ##### Replace None with formula. This should result in a boolean variable (True or False). You can check the variable type in the console with the command: "type(reject_null_TF_f)"

    if p_value <= alpha_f and z_score >= z_crit:
        reject_null_TF_f = True
    else:
        reject_null_TF_f = False

    return reject_null_TF_f, z_score, p_value


def calc_optimal_sample_size(p0_f, mde_f, alpha_f, power_f):
    t_alpha2 = abs(norm.ppf(alpha_f/2))
    t_beta = abs(norm.ppf(1-power_f))
    p1_f = p0_f + mde_f
    p_avg = (p0_f + (p0_f + mde_f))/2
#    sample_size = None  ##### Replace None with formula
    sample_size = (t_alpha2*(2*p_avg*(1-p_avg))**0.5 + t_beta*(p0_f*(1-p0_f) + p1_f*(1-p1_f))**0.5)**2 / mde**2
    return sample_size


# ========== DECLARE PARAMETERS ==========
trial_start_date = datetime.date(year=2020, month=8, day=1)

# ========== LOAD DATA ==========
df = pd.read_csv('AB_trial_data.csv')
df.date = pd.to_datetime(df.date, format='%Y-%m-%d')    # parse string format
df.date = df.date.apply(lambda x: datetime.date(year=x.year, month=x.month, day=x.day)) # convert to standard (non-pandas) format for comparison against other dates

# ========== ANALYSES ==========
# ----- Get summary stats -----
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df_summary = df[['year', 'month', 'Variant', 'id', 'purchase_TF']].groupby(['year', 'month', 'Variant']).agg({'id': 'count', 'purchase_TF': 'sum'}).rename(columns={'id': 'num_exposures', 'purchase_TF': 'num_bookings'})
df_summary['conv_rate'] = df_summary['num_bookings']/df_summary['num_exposures']
perc_vA = df_summary.loc[(trial_start_date.year, trial_start_date.month, 'A'), 'num_bookings'] / df_summary.loc[(trial_start_date.year, trial_start_date.month, 'A'), 'num_exposures']
perc_vB = df_summary.loc[(trial_start_date.year, trial_start_date.month, 'B'), 'num_bookings'] / df_summary.loc[(trial_start_date.year, trial_start_date.month, 'B'), 'num_exposures']
print('For month beginning %s, Variant A had %d exposures (%3.1f%%) and Variant B had %d exposures (%3.1f%%)' % (trial_start_date, int(df_summary.loc[(trial_start_date.year, trial_start_date.month, 'A'), 'num_exposures']), perc_vA*100, int(df_summary.loc[(trial_start_date.year, trial_start_date.month, 'B'), 'num_exposures']), perc_vB*100))
print('For month beginning %s, Variant A had %d exposures (%3.1f%%) and Variant B had %d exposures (%3.1f%%)'
      % (trial_start_date,
         int(df_summary.loc[(trial_start_date.year, trial_start_date.month, 'A'), 'num_exposures']),
         perc_vA*100,
         int(df_summary.loc[(trial_start_date.year, trial_start_date.month, 'B'), 'num_exposures']),
         perc_vB*100))


# ------ Question 1 ------
# set parameters
alpha = 0.05    # significance level
num_sides = 2   # one-sided=1 or two-sided=2 test

# --- choose to use all data or trial period data only by commenting and uncommenting the "ALL DATA" and "TRIAL DATA ONLY" sections
# ALL DATA
# variantA_outcomes = df.loc[df['Variant'] == 'A', 'purchase_TF']
# variantB_outcomes = df.loc[df['Variant'] == 'B', 'purchase_TF']

# TRIAL DATA ONLY
variantA_outcomes = df.loc[(df['Variant'] == 'A') & (df.date >= trial_start_date), 'purchase_TF']
variantB_outcomes = df.loc[(df['Variant'] == 'B') & (df.date >= trial_start_date), 'purchase_TF']

# --- conduct tests
reject_null_test, z_score, p_value = reject_null(variantA_outcomes, variantB_outcomes, alpha, num_sides)
print('Conversion rate for Variant A: %3.1f%%' % (calc_proportion(variantA_outcomes)*100))
print('Conversion rate for Variant B: %3.1f%%' % (calc_proportion(variantB_outcomes)*100))
print('Using all Variant B, reject null T/F?: %s' % reject_null_test)
print('z-score = %3.2f and p-value = %3.1f%%' % (z_score, p_value*100))

# ----- Question 2 -----
# set parameters
alpha = 0.05    # significance level
power = 0.80    # power of test
mde = 0.03      # desired minimum detectable effect
num_sides = 2   # one-sided or two-sided test

# --- Configure data
variantA_outcomes_control = df.loc[(df['Variant'] == 'A') & (df.date < trial_start_date), 'purchase_TF']
variantA_outcomes = df.loc[(df['Variant'] == 'A') & (df.date >= trial_start_date), 'purchase_TF']
variantB_outcomes = df.loc[(df['Variant'] == 'B') & (df.date >= trial_start_date), 'purchase_TF']

# --- Calculate and display the optimal sample size
p0 = calc_proportion(variantA_outcomes_control)     # this would be the baseline prior to starting a test since we would only have historical information
n_star = int(np.ceil(calc_optimal_sample_size(p0, mde, alpha, power)))    # optimal sample size; rounding up using np.ceil because sample must be a whole number that is *at least* as large as the optimal size
num_samples = 10

print('The optimal sample size is %d ' % n_star)
if n_star > variantB_outcomes.shape[0]:
    print('Warning: optimal sample size is larger than number of observations.')

# --- conduct test for n samples of the optimal size
# initialize list to store variables from each loop iteration
variantB_outcomes_samples = list()  # this will store the data samples
reject_null_list = list()   # this will store the results of the significance tests
z_score_list = list()   # this still store the z-scores from each test
p_value_list = list()   # this will store the p-values from each test
for i in range(0, num_samples):
#    t_perc_of_trial_data_to_use = None  ##### Replace None with the formula for what percent of the trial data should be used
    numA = int(df_summary.loc[(trial_start_date.year, trial_start_date.month, 'A'), 'num_exposures'])
    numB = int(df_summary.loc[(trial_start_date.year, trial_start_date.month, 'B'), 'num_exposures'])
    t_perc_of_trial_data_to_use = n_star/(numA+numB)

    t_sample = variantB_outcomes.sample(frac=min(t_perc_of_trial_data_to_use, 1))   ##### No changes needed, but think about why we're using a min() function here
    t_reject, t_z_score, t_p_value = reject_null(variantA_outcomes, t_sample, alpha, num_sides)

    variantB_outcomes_samples.append(list(t_sample))    ##### No changes needed, but investigate what the append function is doing here
    ##### Add lines here to update the remainder of the lists initilaized before this loop
    reject_null_list.append(t_reject)
    z_score_list.append(t_z_score)
    p_value_list.append(t_p_value)

print("For %d samples of optimal sample size %d, %3.2f%% rejected the null" % (num_samples, n_star, calc_proportion(reject_null_list)*100))

df_sample_summary = pd.DataFrame(data={'sample number': list(range(0, num_samples)), 'z_score': z_score_list, 'p_value': p_value_list})
print(df_sample_summary[['sample number', 'z_score', 'p_value']])
df_sample_summary.to_csv('sample_summary.csv', index=False)
