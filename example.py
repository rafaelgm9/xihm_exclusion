import numpy as np
import xihm_exclusion
from matplotlib import pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['text.usetex'] = True


# Set cosmology
from colossus.cosmology import cosmology
from classy import Class


# First set up cosmology to get power spectrum
Omega_b = 0.049
Omega_m = 0.318
Omega_cdm = Omega_m - Omega_b
h = 0.6704 #H0/100
sigma8 = 0.835
n_s = 0.962

# Fox cosmology for CLASS
foxclass = {'output': 'mPk',
            'non linear': 'halofit',
            'Omega_b': Omega_b,
            'Omega_cdm': Omega_cdm,
            'h': h,
            'sigma8': sigma8,
            'n_s': n_s,
            'P_k_max_1/Mpc': 1000,
            'z_max_pk': 10.  # Default value is 10
                }

classcosmo = Class()
classcosmo.set(foxclass)
classcosmo.compute()

# Fox cosmology for Colossus
foxcolossus = {'flat': True,
               'H0': h * 100 ,
               'Om0': Omega_m,
               'Ob0': Omega_b,
               'sigma8': sigma8,
               'ns': n_s}

cosmology.addCosmology('fox', foxcolossus)
cosmology.setCosmology('fox')
cosmo = cosmology.setCosmology('fox')

Om = Omega_m


# Power spectrum
k = np.logspace(-4, 2, num=1000)  # 1/Mpc
z = 0.
Plin = np.array([classcosmo.pk_lin(ki, z) for ki in k])
Pnl = np.array([classcosmo.pk(ki, z) for ki in k])
# NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
# to use in the toolkit. To do this you would do:
k /= foxclass['h']
Plin *= foxclass['h'] ** 3
Pnl *= foxclass['h'] ** 3
# End of set cosmology


# Read xihm data
################# Mbin 2e14-5e14 Msun/h #################
xihm = np.loadtxt('data/xihm1_2e14_5e14_z0.0_down10')
covxihm = np.loadtxt('data/covxihm1_2e14_5e14_z0.0_down10')
r = np.loadtxt('data/r')

# Parameters
m = 10**14.48
rt = 1.832
c = 6.474
alpha = 0.18
Delta = 0.25
bias = 2.860
ra = 1.8
rb = 0.6
Delta2 = 0.28
reff = 1.6
Deltaeff = 0.70


# Get ximm
ximm = np.loadtxt('data/ximm_z0.0_down10')

# Get model
model = xihm_exclusion.xihm_model(r, m, rt, c, alpha, Delta, bias, ra, rb, Delta2, ximm, reff, Deltaeff, Om)

# Plot
plt.errorbar(r, r * r * xihm, yerr=r * r * np.sqrt(covxihm.diagonal()), marker='o', linewidth=0, markersize=3, elinewidth=1, capsize=2, label=r'data', zorder=1)
plt.loglog(r, r * r * model, linewidth=2, label=r'best fit', zorder=2)
plt.xlabel(r'$r$', fontsize=24)
plt.ylabel(r'$r^2 \xi_{hm}$', fontsize=24)
plt.legend()
plt.savefig('figs/xihm.png')
plt.clf()

plt.axhline(0., color='black')
plt.semilogx(r, xihm/model - 1.)
plt.fill_between(r, -np.sqrt(covxihm.diagonal())/xihm, np.sqrt(covxihm.diagonal())/xihm, color='grey',alpha=0.2, label=r'data', zorder=1)
plt.xlabel(r'$r$', fontsize=24)
plt.ylabel(r'fractional difference', fontsize=24)
plt.ylim(-0.10, 0.10)
plt.savefig('figs/fracdif_xihm.png')