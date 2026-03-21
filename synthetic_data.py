'''Load Saturation Detection.
Standalone Python script
'''
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from pyloadsat import (is_constant, is_monotonic_increasing, has_transition,
                       trend_lowess, window_acf, moments_right, response_lr, plateau_earliest,
                       VRSDetector)
from pyloadsat.data import make_demand_exp, load_carried


if __name__ != '__main__':
    raise RuntimeError('This module must be run as a standalone script.')

# Setup NumPy print format
np.set_printoptions(threshold=10, precision=4)

print('''\nA Framework for Detecting The Onset of Load Saturation in Capacity-Constrained Systems
Experiment with Synthetic Data
(C) 2026, Andrei Batyrov, andrey_batyrov@hotmail.com
''')
print('''\nI. Synthetic Data
-----------------
We generate synthetic incoming load A_t and then apply a service model to convert it to carried load Y_t.
As discussed in the Setting and Assumptions subsection, the load is assumed to be:
(1) observed in a system with a constant capacity constraint;
(2) monotonically growing to reflect the underlying demand dynamics;
(3) & (4) transition observable with two operational regimes: elastic followed by saturated.
No assumptions are made in the time domain about load's
- arrival distribution,
- stationarity,
- linearity,
- autocorrelation structure,
- missing values and/or outliers.

Load Model
1. Number of observations: T, observation time labels t in [1, T].
2. Configured constant capacity constraint: C.
3. Let incoming load A_hat_t reflect monotonic demand growth over time with rate r: A_hat_t = exp[rt].
4. Since we do not make any assumptions about arrival distribution, we consider two service models for carried load:
    * Erlang B: Y_hat_t = A_hat_t(1 - P_b(A_hat_t, C)).
    * Hard capacity: Y_hat_t = min(A_hat_t, C) (not shown in the article).
5. Since we do not make any assumptions about autocorrelation, 
      we add seasonal AR(1)_s noise e_t = phi e_{t-s} + eta_t to the carried load: 
      Y_t = Y_hat_t + e_t.
7. Missing values and/or outliers are possible.
''')

print('We first execute the framework algorithm step-by-step and later run the complete framework end-to-end.')

T = 1000 # Generate a sample for 1000 observations
t = np.arange(T) # Discrete time labels
rate = 2 / T # Growth rate
demand = make_demand_exp(rate=rate, size=T) # Exponential demand growth model

carried, capacity = load_carried(model='erlangb')
# carried, capacity = load_carried(model='hardcap') # Not shown in the article

print("""\nLet's take a look at the carried load profile. 
Note the noisy nature of data with possible missing values and outliers.""")

print('''We observe that the hypothetical incoming load exceeds the system capacity, 
while the observed served load exhibits saturation. 
According to the theorem, this necessarily induces an unknown critical point t* separating the elastic and saturated regimes, 
which forms the basis for the subsequent detection. 
Before t* the carried load Y_t is a good proxy of incoming demand A_t. 
However, after that transition point the carried load becomes saturated and does not reflect genuine demand dynamics any longer.''')

plt.figure(figsize=(8, 5))
plt.plot(demand, ls='--', lw=3, color='C5', label='$\\hat{A}_t$ (unobserved)')
plt.plot(carried, lw=0, marker='o', markersize=5, color='C0', alpha=0.25, label='$Y_t$')
plt.plot(capacity, lw=1.5, ls='--', color='C3', label='$C$')
plt.scatter(t.min(), carried.min(), marker='none')
plt.xlabel('Time')
plt.ylabel('LU')
plt.title('Demand $A_t$ (unobserved), carried load $Y_t$, and system capacity $C$')
plt.margins(x=0.012, y=0.02)
plt.gca().set_axisbelow(True)
plt.grid(lw=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.show()

print('\nCheck Assumption (1): Constant capacity constraint C_t = C, for all t.')

constant_status = is_constant(capacity)
if constant_status:
    print('Constant capacity constraint is met, continue.')
else:
    raise RuntimeError('Constant capacity constraint is not met, stop.')

print('''\nII. Trend Estimation
--------------------
We use `statsmodels` implementation of LOWESS where the span is configured as a fraction of the number of all observations.
As discussed in the Sensitivity Analysis section, we use the default smoothing span f = 2/3.
''')

print("Let's take a look at the estimated trend.")

frac = 2. / 3
trend = trend_lowess(carried, frac=frac)
print(f'{trend = }')

print('The estimated trend is robust to noise and outliers, whereas still preserving the regime shift information.')

plt.figure(figsize=(8, 5))
plt.plot(carried, lw=0, marker='o', markersize=5, color='C0', alpha=0.25, label='$Y_t$')
plt.plot(trend, ls='--', lw=3, color='C4', label='$\\hat{Y}_t$')
# plt.plot(capacity, ls='--', color='C3', label='$C$')
plt.xlabel('Time')
plt.ylabel('LU')
plt.title('Carried load with estimated trend (not shown in the article)')
plt.legend(loc='upper left', fontsize=11)
plt.gca().set_axisbelow(True)
plt.grid(lw=0.5)
plt.show()

print('\nCheck Assumption (2): Monotonic demand growth. As discussed in the Sensitivity Analysis section, p_0 = 0.8.')

p_0 = 0.8
monotonic_status, _ = is_monotonic_increasing(trend, p_0)
if monotonic_status:
    print('Trend is monotonically increasing, continue.')
else:
    raise RuntimeError('Trend is not monotonically increasing, stop.')

print('''\nIII. Local Moment Estimation
----------------------------
As discussed in the Sensitivity Analysis section, 
we estimate the local moment window size from the detrended load ACF and set it equal to the longest lag, 
or 2 if no significant autocorrelation is found.
''')

acf, acf_peaks, w = window_acf(carried - trend)
print(f'{w = }')

_, ax_acf = plt.subplots(figsize=(8, 5))
plot_acf(carried - trend, zero=False, auto_ylims=True, use_vlines=True, missing='drop', alpha=0.05, 
         title='ACF (not shown in the article)', ax=ax_acf)
ax_acf.plot(acf[0] * acf_peaks, lw=0, marker='s', color='C1', label='ACF peaks')
ax_acf.set_xlabel('Lag')
plt.gca().set_axisbelow(True)
ax_acf.grid(lw=0.25)
ax_acf.legend(loc='upper right')
plt.show()

print('We use the estimated trend and window size to estimate the sequence of pairs of local means and local variances')

print("Let's take a look at the variance-mean relationship.")

means, variances, mean_max = moments_right(trend, window=w)
print(f'{means = }\n{variances = }')

plt.figure(figsize=(7, 5))
plt.plot(means, variances, ls='--', lw=3, color='C0', 
         label='$\\operatorname{Var[\\hat{Y}]}\\left(\\operatorname{\\mathbb{E}}[\\hat{Y}]\\right)$')
plt.xlabel('LU')
plt.ylabel('LU²')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title('Variance-mean relationship (not shown in the article)')
plt.legend(loc='upper right', fontsize=11)
plt.gca().set_axisbelow(True)
plt.grid(lw=0.5)
plt.show()

print('''\nCheck Assumptions (3) & (4): Transition observability. 
As discussed in the Sensitivity Analysis section, the system is transition-observable 
if the mean maximizing the variances is in [q_0.35, q_0.65].''')

print(f'{mean_max = :.2f}')

q_lower, q_upper = 0.35, 0.65 # core around median
transition_status, transition_regime, means_lower, means_upper = has_transition(means, mean_max, q_lower, q_upper)
print(f'{transition_status = }\n{means_lower = :.2f}, {means_upper = :.2f}')

means_hist_y, means_hist_x = np.histogram(means[~np.isnan(means)], bins=30, density=True)
means_hist_x = means_hist_x[:-1]

plt.figure(figsize=(5, 5))
plt.bar(means_hist_x, means_hist_y, width=0.1, color='C0', alpha=0.35)
plt.bar(means_hist_x[(means_hist_x >= means_lower) & (means_hist_x <= means_upper)], 
        means_hist_y[(means_hist_x >= means_lower) & (means_hist_x <= means_upper)], 
        width=0.1, color='C2', alpha=0.35, label=f'$[q_{{{q_lower}}}, q_{{{q_upper}}}]$')
plt.bar(mean_max, means_hist_y.max(), width=0.05, color='C1', alpha=0.35, 
        label='$\\operatorname{\\mathbb{E}}_{\\max}[\\hat{Y}]$')
plt.xlabel('Erl')
plt.title('Distribution of $\\operatorname{\\mathbb{E}}[\\hat{Y}]$ (not shown in the article)')
plt.legend()
plt.gca().set_axisbelow(True)
plt.grid(lw=0.5)
plt.show()

if transition_status:
    print('Regime transition is identifiable, continue.')
else:
    raise RuntimeError(f'Regime transition is not identifiable: {transition_regime}, stop.')

print('''\nIV. Local Response Function Estimation 
--------------------------------------
We use `statsmodels` implementation of rolling linear regression to compute R ~ beta_LR. 
As discussed in the Sensitivity Analysis section, 
we use ROT to estimate the bandwidth h = floor(k T^{-1/5}).
''')

k_min, k_max = 2 * T ** (1/5) + 1e-9, T ** (6/5)
print(f'{k_min = :.2f}, {k_max = :.2f}')

print('''We can adjust the robustness of the linear regression estimator to noise by changing the bandwidth h 
which may incur a delay in reporting its sign change. 
This therefore may shift t* to the right, i.e. incur a delay in reporting the onset of load saturation.
We choose k = T / 3 as a reasonable trade-off between ensuring robustness while minimizing delay.
''')

k = T / 3
h = int(np.floor(k * T ** (-1/5)))
print(f'{k = :.2f}, {h = }')

print("Let's look at the estimated local response function for h.")

response = response_lr(y=variances, X=means, bandwidth=h)
if response is None:
    raise RuntimeError('Response function cannot be estimated, stop.')
print(f'{response = }')

print('We observe the prominent extremum (maximum) of the variance-mean relationship and the sign change of the response function.')

plt.figure(figsize=(7, 5))
plt.plot(means, variances, ls='--', lw=3, color='C0', 
            label='$\\operatorname{Var[\\hat{Y}]}\\left(\\operatorname{\\mathbb{E}}[\\hat{Y}]\\right)$')
plt.xlabel('LU')
plt.ylabel('LU²')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title('Variance-mean relationship $\\operatorname{Var[\\hat{Y}]}\\left(\\operatorname{\\mathbb{E}}[\\hat{Y}]\\right)$ and response function $\\mathcal{R}$ estimated via local response parameter $\\beta^{(LR)}$')
plt.legend(loc='upper left', fontsize=11)
plt.grid(axis='x', lw=0.5)
ax2 = plt.gca().twinx()
ax2.plot(means, response, lw=0, marker='.', markersize=5, color='C5', alpha=0.75, 
         label='$\\beta^{(LR)}$')
ax2.set_ylabel('LU')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.legend(loc='upper right', fontsize=10)
plt.gca().set_axisbelow(True)
ax2.grid(axis='y', lw=0.5)
plt.show()

print('''\nV. Detection of Saturation
--------------------------
We classify the regimes as follows:
1. beta_LR > 0, elastic regime, variance responds proportionally to increases in load level.
2. beta_LR <= 0, saturated regime, variance decreases or stagnates as load grows.
The transition time t* is the earliest time when a persistent plateau beta_LR <= 0 in the sequence {beta_LR} is established. 
As discussed in the Sensitivity Analysis section, the length of the sought plateau l in [w, T]. 
We start from the smallest l and increase it, until the longest contiguous plateau is found.
''')

t_crit = -1
for l in range(w, T + 1, w):
    t_crit_l = plateau_earliest((response <= 0.), l)
    t_crit = max(t_crit, t_crit_l)
    # print(l, t_crit_l)
if t_crit == -1:
    raise RuntimeError(f'No persistent plateau of length {l} found, stop.')

carried_crit = means[t_crit]
print(f'{t_crit = }\n{carried_crit = :.2f}')

print('Label the carried load, as per the identified regimes.')

carried_elastic = carried[:t_crit]
carried_saturated = carried[t_crit:]

print('Finally, we can report the identified t* and Y* and show the labeled carried load.')

plt.figure(figsize=(8, 5))
plt.plot(demand, ls='--', lw=3, color='C5', label='$\\hat{A}_t$ (unobserved)')
plt.scatter(t_crit, carried.min(), color='black', marker='v', label='$t^*$')
plt.scatter(t.min(), carried_crit, color='black', marker='<', label='$Y^*$')
plt.plot(t[:t_crit], carried_elastic, lw=0, marker='o', markersize=5, color='C2', alpha=0.25, label='$Y_{el}$')
plt.plot(t[t_crit:], carried_saturated, lw=0, marker='x', markersize=5, color='C1', alpha=0.375, label='$Y_{sat}$')
plt.plot(capacity, lw=1.5, ls='--', color='C3', label='$C$')
plt.xlabel('Time')
plt.ylabel('LU')
plt.title('Carried load $Y_t$ with identified operational regimes: elastic $Y_{el}$ followed by saturated $Y_{sat}$')
plt.margins(x=0.012, y=0.02)
plt.gca().set_axisbelow(True)
plt.grid(lw=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.show()

print('''\nVI. Running Framework End-to-End
--------------------------------
Here we run the complete framework to produce identical results of the load saturation detection as in the step-by-step implementation above.
''')

print('1. Create a detector.')
detector = VRSDetector('synth')
print('Detector is initialized:')
pprint(detector.summary(), sort_dicts=False)

print('2. Fit the detector on our data: estimate trend, local moments, and local response function.')
detector.fit(carried)
print('Detector is fitted:')
pprint(detector.summary(), sort_dicts=False)

print('3. Identify the critical point.')
detect_status, t_crit, load_crit = detector.detect()
print(f'{detect_status = }\n{t_crit = }\n{load_crit = }')
print('Final state of the detector:')
pprint(detector.summary(), sort_dicts=False)

print('''\nVII. Results
------------
We have identified two operational regimes of the system:
    * Elastic in which the carried load is a reliable proxy of the incoming load; 
      demand forecasting is admissible.
    * Saturated in which the carried load does not convey the genuine demand dynamics; 
      capacity expansion is recommended.

An important empirical observation is that Y* < C. 
This indicates that the onset of load saturation is not tied to the configured capacity limit itself, 
but rather reflects an earlier loss of load elasticity. 
Such behavior highlights the diagnostic value of the proposed framework in revealing latent saturation effects 
that are not captured by static capacity parameters.
''')
