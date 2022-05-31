import numpy as np

import networkx as nx
from networkx.algorithms import bipartite

def collision_resolution(ue_choices, buffered_access_attempts, gamma_th):
    """Evaluates the number of successful access attempts of the random access method given the choices made by the UEs
    and the power received by the BS.

    Parameters
    ----------
    ue_choices : dict
        Dictionary containing choices of the UEs.

            keys:
                UE indexes

            values:

                RIS-assisted protocol:
                    Contains a list with the access frames (configurations) chosen by each active UE.

                Slotted ALOHA protocol:
                    Contains a list with the access frames chosen by each active UE.

    buffered_access_attempts : ndarray of shape (num_access_frames)
        Buffered UL received signal in each access attempt.

    gamma_th : float
        Threshold SNR for SIC.


    Returns
    -------
    num_successful_attempts : integer
        Number of successful access attempts.

    """

    # Get number of active UEs
    num_active_ues = len(list(ue_choices.keys()))

    # Get number of access frames
    num_access_frames = len(list(buffered_access_attempts.keys()))

    # Nothing to compute
    if num_active_ues == 1:
        return 1

    # Initialize number of successful access attempts
    num_successful_attempts = 0

    # Enumerate UEs
    enumeration_active_ues = np.arange(0, num_active_ues).astype(int)

    # Enumerate access frames
    enumeration_access_frames = [str('A') + str(access_frame) for access_frame in range(num_access_frames)]

    # Create edge list: each edge is a tuple
    edge_list = []

    # Go through all active UEs
    for ue in enumeration_active_ues:

        # Chosen access frames
        chosen_access_frames = ue_choices[ue]

        # Go through all access frames chosen by the current UE
        for access_frame in chosen_access_frames:
            edge_list.append((ue, enumeration_access_frames[access_frame]))

    # Create a Bipartite graph
    B = nx.Graph()

    # Add UEs' nodes
    B.add_nodes_from(enumeration_active_ues, bipartite=0)

    # Add access frames' nodes
    B.add_nodes_from(enumeration_access_frames, bipartite=1)

    # Add edges
    B.add_edges_from(edge_list)

    # Remove isolated nodes
    B.remove_nodes_from(list(nx.isolates(B)))

    # Capture effect
    while True:

        # Compute node degrees
        deg_access_frames, _ = bipartite.degrees(B, enumeration_active_ues)
        dict_deg_access_frames = dict(deg_access_frames)

        # No singletons, we cannot decode -> break
        if not (1 in dict_deg_access_frames.values()):
            break

        # Take the first singleton
        for access_frame_str_dagger, deg in dict_deg_access_frames.items():
            if deg == 1:
                break

        # Find respective UE
        ue_dagger = [ue for ue, aa in B.edges if aa == access_frame_str_dagger][0]

        # Get natural index
        access_frame_dagger = int(access_frame_str_dagger[1:])

        # Compute SNR of the buffered signal
        buffered_snr = np.abs(buffered_access_attempts[access_frame_dagger][ue_dagger])**2 / np.abs(buffered_access_attempts[access_frame_dagger]['noise'])**2

        # Check SIC condition
        if buffered_snr > gamma_th:

            # Update number of successful access attempts
            num_successful_attempts += 1

            # Identify other edges with dagger UE
            edge_list_ue_dagger = [(ue, aa) for ue, aa in B.edges if ue == ue_dagger]

            # Go through buffered signals that contain the dagger UE and update them
            for edge in edge_list_ue_dagger:

                if edge != (ue_dagger, access_frame_str_dagger):

                    # Other access frame indexes
                    other_access_frame_str = edge[1]
                    other_access_frame = int(other_access_frame_str[1:])

                    for other_ue in buffered_access_attempts[other_access_frame].keys():
                        buffered_access_attempts[other_access_frame][other_ue] -= buffered_access_attempts[access_frame_dagger][ue_dagger]

                    # Update noise
                    buffered_access_attempts[other_access_frame]['noise'] -= buffered_access_attempts[access_frame_dagger]['noise']

            # Remove edges
            B.remove_edges_from(edge_list_ue_dagger)

        # useless singleton
        else:

            # Remove edge
            B.remove_edge(ue_dagger, access_frame_str_dagger)

        # Remove isolated nodes
        B.remove_nodes_from(list(nx.isolates(B)))

    return num_successful_attempts

def collision_resolution_slotted(ue_access, buffered_access_attempts, gamma_th):
    """Evaluates the number of successful access attempts of the random access method given the choices made by the UEs
    and the power received by the BS.

    Parameters
    ----------
    ue_choices : dict
        Dictionary containing choices of the UEs.

            keys:
                UE indexes

            values:

                RIS-assisted protocol:
                    Contains a list with the access frames (configurations) chosen by each active UE.

                Slotted ALOHA protocol:
                    Contains a list with the access frames chosen by each active UE.

    buffered_access_attempts : ndarray of shape (num_access_frames)
        Buffered UL received signal in each access attempt.

    gamma_th : float
        Threshold SNR for SIC.


    Returns
    -------
    num_successful_attempts : integer
        Number of successful access attempts.

    """

    # Get total number of active UEs
    total_num_ues = len(list(ue_access.keys()))

    # Get number of access slots
    num_access_slots = len(list(buffered_access_attempts.keys()))

    # Nothing to compute
    if total_num_ues == 1:
        return 1

    # Initialize number of successful access attempts
    num_successful_attempts = 0

    # Enumerate access slots
    enumeration_access_slots = [str('A') + str(access_slot) for access_slot in range(num_access_slots)]

    # Create edge list: each edge is a tuple
    edge_list = []

    # Go through all active UEs
    for ue in ue_access.keys():

        # Chosen access slots
        chosen_access_slots = ue_access[ue]

        # Go through all access slots chosen by the current UE
        for access_slot in chosen_access_slots:
            edge_list.append((ue, enumeration_access_slots[access_slot]))

    # Create a Bipartite graph
    B = nx.Graph()

    # Add UE nodes
    B.add_nodes_from(ue_access.keys(), bipartite=0)

    # Add access slot nodes
    B.add_nodes_from(enumeration_access_slots, bipartite=1)

    # Add edges
    B.add_edges_from(edge_list)

    # Remove isolated nodes
    B.remove_nodes_from(list(nx.isolates(B)))

    # Capture effect
    while True:

        # Compute node degrees
        deg_access_frames, _ = bipartite.degrees(B, ue_access.keys())
        dict_deg_access_frames = dict(deg_access_frames)

        # No singletons, we cannot decode -> break
        if not (1 in dict_deg_access_frames.values()):
            break

        # Take the first singleton
        for access_frame_str_dagger, deg in dict_deg_access_frames.items():
            if deg == 1:
                break

        # Find respective UE
        ue_dagger = [ue for ue, aa in B.edges if aa == access_frame_str_dagger][0]

        # Get natural index
        access_frame_dagger = int(access_frame_str_dagger[1:])

        # Compute SNR of the buffered signal
        buffered_snr = np.abs(buffered_access_attempts[access_frame_dagger][ue_dagger])**2 / np.abs(buffered_access_attempts[access_frame_dagger]['noise'])**2

        # Check SIC condition
        if buffered_snr > gamma_th:

            # Update number of successful access attempts
            num_successful_attempts += 1

            # Identify other edges with dagger UE
            edge_list_ue_dagger = [(ue, aa) for ue, aa in B.edges if ue == ue_dagger]

            # Go through buffered signals that contain the dagger UE and update them
            for edge in edge_list_ue_dagger:

                if edge != (ue_dagger, access_frame_str_dagger):

                    # Other access frame indexes
                    other_access_frame_str = edge[1]
                    other_access_frame = int(other_access_frame_str[1:])

                    for other_ue in buffered_access_attempts[other_access_frame].keys():
                        buffered_access_attempts[other_access_frame][other_ue] -= buffered_access_attempts[access_frame_dagger][ue_dagger]

                    # Update noise
                    buffered_access_attempts[other_access_frame]['noise'] -= buffered_access_attempts[access_frame_dagger]['noise']

            # Remove edges
            B.remove_edges_from(edge_list_ue_dagger)

        # useless singleton
        else:

            # Remove edge
            B.remove_edge(ue_dagger, access_frame_str_dagger)

        # Remove isolated nodes
        B.remove_nodes_from(list(nx.isolates(B)))

    return num_successful_attempts

# def throughput_evaluation(ue_choices):
#     """Evaluates the throughput of the random access method given the choices made by the UEs.
#
#     Parameters
#     ----------
#     ue_choices : dict
#         Dictionary containing choices of the UEs.
#             keys -> UE indexes
#             values -> tuple,
#                 1st-dimension contains a list or an integer with the configurations chosen by a UE.
#                 2nd-dimension contains the pilot chosen by a UE.
#
#     Returns
#     -------
#     throughput : float
#         Ratio of success attempts per number of UEs trying to access.
#
#     """
#
#     # Get number of active UEs
#     num_active_ue = len(list(ue_choices.keys()))
#
#     # Nothing to compute
#     if num_active_ue == 1:
#         return 1
#
#     # Get mask of pilots
#     mask_pilots = np.array(list(ue_choices.values()), dtype=object)[:, 1]
#
#     # Get active pilots
#     active_pilots = np.unique(mask_pilots)
#
#     # Pool of success UEs
#     success_pool = []
#
#     # Go through all active pilots
#     for pilot in active_pilots:
#
#         # Create a Bipartite graph
#         B = nx.Graph()
#
#         # Get index of colliding UEs
#         colliding_ues = np.array(list(ue_choices.keys()))[mask_pilots == pilot]
#
#         # Add colliding UEs
#         B.add_nodes_from(colliding_ues, bipartite=0)
#
#         # Create edge list
#         edge_list = []
#
#         # Create list with chosen configurations
#         chosen_configs_list = []
#
#         # Go through all colliding UEs
#         for kk in colliding_ues:
#
#             # Chosen configurations
#             chosen_configs = ue_choices[kk][0]
#
#             if isinstance(chosen_configs, (int, np.int64)):
#                 edge_list.append((kk, 'S' + str(chosen_configs)))
#
#                 if not ('S' + str(chosen_configs) in chosen_configs_list):
#                     chosen_configs_list.append('S' + str(chosen_configs))
#
#             else:
#
#                 # Go through all configurations chosen by a UE
#                 for config in chosen_configs:
#                     edge_list.append((kk, 'S' + str(config)))
#
#                     if not ('S' + str(config) in chosen_configs_list):
#                         chosen_configs_list.append('S' + str(config))
#
#         # Add configuration nodes
#         B.add_nodes_from(chosen_configs_list, bipartite=1)
#
#         # Add edges
#         B.add_edges_from(edge_list)
#
#         # Capture effect
#         while True:
#
#             # Obtain node degrees
#             deg_ue, deg_config = bipartite.degrees(B, colliding_ues)
#             dict_deg_ue = dict(deg_ue)
#
#             # No singletons, we cannot decode
#             if not (1 in dict_deg_ue.values()):
#                 break
#
#             # Go through the degree dictionary, if not break
#             for config, deg in dict_deg_ue.items():
#
#                 # Is there a singleton?
#                 if deg == 1:
#
#                     # Find respective UE
#                     ue = [ue for ue, cc in B.edges if cc == config][0]
#
#                     # Remove edge
#                     B.remove_edge(ue, config)
#
#                     # Check UE
#                     if not (ue in success_pool):
#
#                         # Add UE to the pool
#                         success_pool.append(ue)
#
#     # Compute success attempts
#     success_attempts = len(success_pool)
#
#     # Compute performance metric: Throughput
#     assert success_attempts <= num_active_ue
#
#     return success_attempts / num_active_ue
#
#
# def throughput_evaluation_with_power(ue_choices, gamma_ul, gamma_th):
#     """Evaluates the throughput of the random access method given the choices made by the UEs and the power received by
#     the BS.
#
#     Parameters
#     ----------
#     ue_choices : dict
#         Dictionary containing choices of the UEs.
#             keys -> UE indexes
#             values -> tuple,
#                 1st-dimension contains a list or an integer with the configurations chosen by a UE.
#                 2nd-dimension contains the pilot chosen by a UE.
#
#     gamma_ul : ndarray of shape (num_chosen_configs, num_ues)
#         Received uplink SNR for each UE for each config.
#
#     gamma_th : float
#         Threshold SNR for SIC.
#
#
#     Returns
#     -------
#     throughput : float
#         Ratio of success attempts per number of UEs trying to access.
#
#     """
#
#     # Get number of active UEs
#     num_active_ue = len(list(ue_choices.keys()))
#
#     # Nothing to compute
#     if num_active_ue == 1:
#         return 1
#
#     try:
#         gamma_ul.shape[1]
#     except:
#         gamma_ul = gamma_ul[np.newaxis, :]
#
#     # Get mask of pilots
#     mask_pilots = np.array(list(ue_choices.values()), dtype=object)[:, 1]
#
#     # Get active pilots
#     active_pilots = np.unique(mask_pilots)
#
#     # Pool of success UEs
#     success_pool = []
#
#     # Go through all active pilots
#     for pilot in active_pilots:
#
#         # Create a Bipartite graph
#         B = nx.Graph()
#
#         # Get index of colliding UEs
#         colliding_ues = np.array(list(ue_choices.keys()))[mask_pilots == pilot]
#
#         # Add colliding UEs
#         B.add_nodes_from(colliding_ues, bipartite=0)
#
#         # Create edge list
#         edge_list = []
#
#         # Create list with chosen configurations
#         chosen_configs_list = []
#
#         # Go through all colliding UEs
#         for kk in colliding_ues:
#
#             # Chosen configurations
#             chosen_configs = ue_choices[kk][0]
#
#             if isinstance(chosen_configs, (int, np.int64)):
#                 edge_list.append((kk, 'S' + str(chosen_configs)))
#
#                 if not ('S' + str(chosen_configs) in chosen_configs_list):
#                     chosen_configs_list.append('S' + str(chosen_configs))
#
#             else:
#
#                 # Go through all configurations chosen by a UE
#                 for config in chosen_configs:
#                     edge_list.append((kk, 'S' + str(config)))
#
#                     if not ('S' + str(config) in chosen_configs_list):
#                         chosen_configs_list.append('S' + str(config))
#
#         # Add configuration nodes
#         B.add_nodes_from(chosen_configs_list, bipartite=1)
#
#         # Add edges
#         B.add_edges_from(edge_list)
#
#         # Obtain node degrees
#         deg_ue, deg_config = bipartite.degrees(B, colliding_ues)
#         dict_deg_ue = dict(deg_ue)
#
#         # Capture effect
#         while True:
#
#             # No singletons, we cannot decode -> break
#             if not (1 in dict_deg_ue.values()):
#                 break
#
#             # Go through the degree dictionary
#             for config, deg in dict_deg_ue.items():
#
#                 # Is there a singleton?
#                 if deg == 1:
#
#                     # Find respective UE
#                     ue = [ue for ue, cc in B.edges if cc == config][0]
#
#                     # Get absolute config index
#                     abs_index_config = int(config[1:])
#
#                     # Get relative config index
#                     rel_index_config = np.where(np.array(ue_choices[ue][0]) == abs_index_config)[0].item()
#
#                     # Check power
#                     if gamma_ul[rel_index_config, ue] > gamma_th:
#
#                         # Remove all edges with that UE
#                         for user, configuration in B.edges():
#
#                             if user == ue:
#                                 B.remove_edge(ue, configuration)
#
#                                 # Update entries
#                                 dict_deg_ue[configuration] -= 1
#
#                         # Check UE
#                         if not (ue in success_pool):
#
#                             # Add UE to the pool
#                             success_pool.append(ue)
#                     else:
#
#                         # Put a -1 to indicate we cannot that singleton decode
#                         dict_deg_ue[config] = -1
#
#     # Compute success attempts
#     success_attempts = len(success_pool)
#
#     # Compute performance metric: Throughput
#     assert success_attempts <= num_active_ue
#
#     throughput = success_attempts / num_active_ue
#
#     return throughput


