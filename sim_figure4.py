import numpy as np

from scipy.constants import speed_of_light
import scipy.integrate as integrate

from tqdm import trange

########################################
# Private functions
########################################


def integrand(x, mm, coeff_idx, fundamental_freq, comp):
    """
    Define integrand to compute the coefficients of the Fourier series.

    Parameters
    ----------

        x : float
            Corresponds to the angle of reflection in radians.

        mm : int
            Element index.

        coeff_idx : int
            Coefficient index.

        fundamental_freq : float
            Fundamental frequency in radians.

        comp : str
            Choose to return 'real' or 'imag' components of the result.

    Returns
    -------

        integrand : float
            Return value of the integrand.

    """
    result = np.exp(-1j * 2 * np.pi * fundamental_freq * (mm * np.sin(x) + coeff_idx * x))

    if comp == 'real':
        return result.real
    else:
        return result.imag


def fourier_complex_coeff(ue_angle, ris_num_els_hor, fundamental_freq, coeff_idx):
    """
    Compute the complex coefficient of the Fourier series with respect to the
    given coefficient index.

    Parameters
    ----------

        ue_angle : float
            Angular position of the UE in radians.

        ris_num_els_hor : int
            Number of elements on the horizontal dimension of the RIS.

        fundamental_freq : float
            Fundamental frequency in radians.

        coeff_idx : int
            Coefficient index.

    Returns
    -------

        coeff : complex
            Complex coefficient of the Fourier series according to coeff_idx.
    """

    # Enumerate elements
    enumeration_els_hor = np.arange(1, ris_num_els_hor + 1)

    # Prepare to save complex coefficient components
    components = np.zeros(ris_num_els_hor, dtype=np.complex_)

    # Go through all horizontal elements of the RIS
    for mi, mm in enumerate(enumeration_els_hor):

        # Compute constant
        cons = np.exp(1j * 2 * np.pi * fundamental_freq * mm * np.sin(ue_angle))

        # Evaluate integral
        result_real = integrate.quad(integrand, 0, np.pi/2, args=(mm, coeff_idx, fundamental_freq, 'real'), limit=500, epsabs=1e-4)
        result_imag = integrate.quad(integrand, 0, np.pi/2, args=(mm, coeff_idx, fundamental_freq, 'imag'), limit=500, epsabs=1e-4)

        # Get component
        components[mi] = cons * (result_real[0] + 1j * result_imag[0])

    # Calculate coefficient
    coeff = fundamental_freq * components.sum()

    return coeff


def integrand_power(x, ue_angles, ris_num_els_hor, fundamental_freq):
    """
    Integrand to compute average power.

    Parameters
    ----------

        x : float
            Corresponds to the angle of reflection in radians.

        ue_angles : array with floats
            Angular positions of the UEs in radians.

        ris_num_els_hor : int
            Number of elements on the horizontal dimension of the RIS.

    Returns
    -------

        result : float
            Result of the integral.
    """

    # Enumerate elements
    enumeration_els_hor = np.arange(1, ris_num_els_hor + 1)

    # Compute result
    result = np.abs((np.exp(1j * 2 * np.pi * fundamental_freq * enumeration_els_hor * (np.sin(ue_angles) - np.sin(x)))).sum())**2

    return result


def average_power(ue_angles, ris_num_els_hor, fundamental_freq):
    """
    Compute average power of the sigal.

    Parameters
    ----------

        ue_angles : array with floats
            Vector with angular positions of the UEs in radians.

        ris_num_els_hor : int
            Number of elements on the horizontal dimension of the RIS.

        fundamental_freq : float
            Fundamental frequency in radians.

    Returns
    -------

        average_power : array with floats shape of ue_angles
            Vector with average power of the signal according to the positions
            of the UEs.
    """

    # Prepare to save average power
    avg_power = np.zeros_like(ue_angles)

    # Go through all angles
    for ue, angle in enumerate(ue_angles):

        # Solve integral
        result = integrate.quad(integrand_power, 0, np.pi/2, args=(angle, ris_num_els_hor, fundamental_freq), epsabs=1e-4)

        # Compute average power
        avg_power[ue] = result[0]

    avg_power *= fundamental_freq

    return avg_power

########################################
# Parameters
########################################

# Electromagnetics
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency
wavenumber = 2 * np.pi / wavelength

# Number of horizontal elements
ris_num_els_hor = 10

# Size of each element
ris_size_el = wavelength/2

########################################
# Evaluating maximum frequency
########################################

# Define fundamental frequency
fundamental_freq = 0.5

# Define range of UE's angle
ue_angles = np.linspace(0, np.pi/2)

# Compute average power
avg_power = average_power(ue_angles, ris_num_els_hor, fundamental_freq)

# Define range of epsilon
epsilon_range = 10**(-np.array([1, 2, 3], dtype=float))

# Prepare to save smallest symmetric interval
interval = np.zeros((epsilon_range.size, ue_angles.size))

# Go through all UE's angles
for ue in trange(ue_angles.size, desc='Simulating', unit='positions'):

    # Extract current angle
    angle = ue_angles[ue]

    # Store complex coefficients
    complex_coeffs = {}

    # Compute basic complex coefficient
    complex_coeffs[0] = fourier_complex_coeff(angle, ris_num_els_hor, fundamental_freq, 0)
    complex_coeffs[-1] = fourier_complex_coeff(angle, ris_num_els_hor, fundamental_freq, -1)
    complex_coeffs[+1] = fourier_complex_coeff(angle, ris_num_els_hor, fundamental_freq, +1)

    # Compute temp index
    coeff_idx = 2

    while True:

        # Evaluate conservation efficiency
        true_ratio = (np.abs(list(complex_coeffs.values())) ** 2).sum() / avg_power[ue]

        if true_ratio < 1 - epsilon_range[-1]:

            # Compute complex coefficients
            complex_coeffs[+coeff_idx] = fourier_complex_coeff(angle, ris_num_els_hor, fundamental_freq, +coeff_idx)
            complex_coeffs[-coeff_idx] = fourier_complex_coeff(angle, ris_num_els_hor, fundamental_freq, -coeff_idx)

            # Update temp index
            coeff_idx += 1

        else:
            break

    # Create a copy
    epsilon_list = list(epsilon_range)

    # Create temp variable
    coeff_idx = 1

    # Store temporally the results
    temp = []

    while True:

        # Indexes
        rang = np.arange(-coeff_idx, coeff_idx+1)

        # Get values
        values = [complex_coeffs[ii] for ii in rang]

        # Evaluate sum
        approx_avg_power = (np.abs(values)**2).sum()

        # Ratio
        ratio = approx_avg_power/avg_power[ue]

        if len(epsilon_list) == 1:

            # Store coeff_idx
            temp.append(-list(complex_coeffs.keys())[-1])

            # Update list
            del epsilon_list[0]

        else:

            if ratio > (1 - epsilon_list[0]):

                # Store coeff_idx
                temp.append(coeff_idx)

                # Update list
                del epsilon_list[0]

        if len(epsilon_list) == 0:
            break

        # Update temp var
        coeff_idx += 1

    # Store simulation result
    interval[:, ue] = temp

# Compute approximated maximum frequency according to Method 1
appr_max_freq_1 = fundamental_freq * ris_num_els_hor

# Compute approximated maximum frequency according to Method 2
appr_max_freq_2 = fundamental_freq * interval

# Save results
np.savez('data/figure4.npz',
    ue_angles=ue_angles,
    epsilon_range=epsilon_range,
    appr_max_freq_1=appr_max_freq_1,
    appr_max_freq_2=appr_max_freq_2
    )
