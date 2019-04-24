import numpy as np
from colossus.halo import profile_einasto
from scipy.special import erfc


rhocrit = 2.77536627e+11
SQRT2 = np.sqrt(2.)


# Functions
def xihm_model(r, m, rt, c, alpha, Delta, bias, ra, rb, Delta2, ximm, reff, Deltaeff, Om):
    """Our model for the halo-mass correlation function.

    Args:
        r (float or array like): 3d distances from halo center in Mpc/h comoving.
        m (float): Mass in Msun/h.
        rt (float): Truncation radius. The boundary of the halo.
        c (float): Concentration.
        alpha (float): Einasto model parameter.
        Delta (float): Width parameter of the truncation term.
        bias (float): Large scale halo bias.
        ra (float): Radius of the first correction term in Msun/h.
        rb (float): Radius of the second correction term in Msun/h.
        Delta2 (float): Width parameter of the exclusion terms.
        ximm (float or array like): Matter-matter correlation function.
        reff (float): Effective radius to substract 1h term from ximm.
        Deltaeff (float): Effective width to substract 1h term from ximm.
        Om (float): Omega_matter.

    Returns:
        float or arraylike: Halo-mass correlation function.

    """

    rhom = Om * rhocrit

    # Get xi1h
    p_einasto = profile_einasto.EinastoProfile(M=m, c=c, alpha=alpha, z=0.0, mdef='200m')
    xi1h = p_einasto.density(1000 * r) / rhom * 1000 ** 3
    xi1h *= thetat(r, rt, Delta)

    # Substract 1halo term
    xi2h = (1 - thetat(r, reff, Deltaeff)) * ximm

    # Get xi2h
    xi2h = bias * xi2h

    # Get correction
    C = - thetat(r, ra, Delta2) * xi2h - thetat(r, rb, Delta2)

    # Full xi
    xi = xi1h + xi2h + C

    return xi


def thetat(r, rt, Delta):
    """Exclusion function. Error function.

        Args:
            r (float or array like): 3d distances from halo center in Mpc/h comoving.
            rt (float): Truncation radius. The boundary of the halo.
            Delta (float): Width parameter of the exclusion function.

        Returns:
            float or arraylike: Exclusion function.

    """

    return 0.5 * erfc((r-rt)/Delta/rt/SQRT2)


def re(r1, r2, scheme=1):
    """Exclusion radius given a percolation scheme. Default Rockstar is 1.

            Args:
                r1 (float): 3d distance from halo center in Mpc/h comoving.
                r2 (float): 3d distance from halo center in Mpc/h comoving.
                scheme (integer): Percolation scheme.

            Returns:
                float: Exclusion radius in Mpc/h comoving.

    """

    if scheme == 1:
        re = max(r1, r2)
    elif scheme == 2:
        re = (r1**3 + r2**3)**(1/3)
    elif scheme == 3:
        re = r1 + r2

    return re