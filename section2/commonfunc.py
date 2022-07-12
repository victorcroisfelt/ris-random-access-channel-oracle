########################################
#   commonfunc.py
#
#   Description:
#
#   Author: @victorcroisfelt
#
#   Date: December 31, 2021
#
#   This code is part of the code package used to generate the numeric results
#   of the paper:
#
#
#   Available on:
#
#
#
########################################
import numpy as np

def dl_channel_gains(
        wavelength,
        bs_gain, bs_pos,
        ue_gain, ue_pos,
        ris_size_el, ris_num_els_hor, ris_num_els_ver, ris_configs
        ):
    """
    Calculate downlink channel gains.

    Parameters
    ----------

    wavelength : float
        Transmission wavelength in meters.

    bs_gain : float
        BS antenna gain.

    bs_pos : ndarray of shape (3, )
        3D-position of the BS.

    ue_gain : float
        UE antenna gain.

    ue_pos : ndarray of shape (3, num_ues)
        3D position of K UEs.

    ris_size_el : float
        Size of each element of the RIS.

    ris_num_els_hor : int
        Number of RIS elements in the horizontal dimension (x-axis).

    ris_num_els_ver : int
        Number of RIS elements in the vertical dimension (z-axis).

    ris_configs : ndarray of shape(num_configs, )
        Set of steering angles.

    Returns
    -------
    channel_gains_dl : ndarray of shape (num_ues, num_configs)
        Downlink channel for each UE and each configuration.

    """

    # Extract distances and angles
    bs_distance = np.linalg.norm(bs_pos)
    bs_angle = np.arctan2(bs_pos[0], bs_pos[1])

    ue_distances = np.linalg.norm(ue_pos, axis=0)
    ue_angles = np.arctan2(ue_pos[0, :], ue_pos[1, :])

    # Compute DL pathloss of shape (num_ues, )
    num = bs_gain * ue_gain * (ris_size_el * ris_size_el)**2
    den = (4 * np.pi * bs_distance * ue_distances)**2

    const = num/den

    pathloss_dl = const * np.cos(bs_angle)**2

    # Compute fundamental frequency
    fundamental_freq = ris_size_el / wavelength

    # Compute term 1 of shape (num_ues, )
    term1 = np.sqrt(pathloss_dl) * ris_num_els_ver

    # Compute term 2 of shape (num_ues, )
    term2 = np.exp(1j * 2 * np.pi * fundamental_freq * ((bs_distance + ue_distances) / ris_size_el))

    # Compute term 3 of shape (num_ues, )
    term3 = np.exp(-1j * 2 * np.pi * fundamental_freq * (ris_num_els_hor + 1) / 2 * (np.sin(bs_angle) - np.sin(ue_angles)))

    # Compute term 4  of shape (num_ues, num_configs)
    enumeration_num_els_hor = np.arange(1, ris_num_els_hor + 1)

    term4 = np.exp(1j * 2 * np.pi * fundamental_freq * enumeration_num_els_hor[:, None, None] * (np.sin(ue_angles)[:, None] - np.sin(ris_configs)[None, :]))
    term4 = term4.transpose(1, 0, 2)

    term4 = term4[:, :, :].sum(axis=1)

    # Compute channel gains
    channel_gains = term1[:, None] * term2[:, None] * term3[:, None] * term4

    return channel_gains