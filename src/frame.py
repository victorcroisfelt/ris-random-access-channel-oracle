import numpy as np

import networkx as nx
from networkx.algorithms import bipartite

########################################
# Private functions
########################################
def bigraph_degree(edge_list):
    """
    Compute the degress of nodes of type B stored in the second dimensions of a
    bipartite graph represented by a list of edges.

    Parameters
    ----------

    edge_list : array of tuples
        Each tuple connects a node type A to a node type B.

    degrees : dict
        Dictionary containing nodes of type B as keys and values representing
        their respective degrees.
    """

    degrees = {}

    for edge in edge_list:

        if edge[1] in degrees.keys():
            degrees[edge[1]] += 1

        else:
            degrees[edge[1]] = 1

    return degrees

########################################
# Classes
########################################
class Block:
    """
    """

    def __init__(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float
    ):

        self.num_slots = num_slots
        self.num_channel_uses = num_channel_uses
        self.num_silent_channel_uses = num_silent_channel_uses

        self.codebook = None

        if decoding_snr is None:
            self.decoding_snr = 1
        else:
            self.decoding_snr = decoding_snr



class Training(Block):
    """

    """

    def __init__(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float
    ):

        super().__init__(num_slots, num_channel_uses, num_silent_channel_uses, decoding_snr)

        self.set_codebook()

    def set_codebook(self):

        self.codebook = np.linspace(0, np.pi/2, self.num_slots)


class Access(Block):
    """

    """

    def __init__(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float,
    ):

        super().__init__(num_slots, num_channel_uses, num_silent_channel_uses, decoding_snr)

        # Set access codebook
        self.set_codebook()

        # Enumerate access slots
        self.enum_ac_slots = np.arange(self.num_slots)

    def set_codebook(self):
        """

        """

        self.codebook = np.linspace(0, np.pi/2, self.num_slots)


    def messages(self, num_ues):

        return np.sqrt(1/2) * (np.random.randn(num_ues, self.num_channel_uses) + 1j * np.random.randn(num_ues, self.num_channel_uses))

    def access_policy(self, ac_info, num_packets=1, access_policy=None, rng=None):
        """

        """

        if rng is None:
            rng = np.random.RandomState()

        # Extract number of UEs
        num_ues = ac_info.shape[0]

        # Number of repeated packets
        num_packets = num_packets if self.num_slots >= num_packets else self.num_slots

        # Prepare to save chosen access slots
        ue_choices = {ue: None for ue in range(num_ues)}

        # Go through all UEs
        for ue in range(num_ues):

            # Choose access policy
            if access_policy == 'RCURAP':

                # Uniformly sample w/o replacement
                ue_choices[ue] = list(rng.choice(self.enum_ac_slots, size=num_packets, replace=False))

            elif access_policy == 'RCARAP':

                # Get probability mass function
                #pmf = np.exp(np.abs(ac_info[ue, :])) / np.exp(np.abs(ac_info[ue, :])).sum()
                pmf = (np.abs(ac_info[ue, :])) / (np.abs(ac_info[ue, :])).sum()

                # Sample w/o replacement according to pmf
                ue_choices[ue] = list(rng.choice(self.enum_ac_slots, size=num_packets, replace=False, p=pmf))

            elif access_policy == 'RGSCAP':

                # Get channel qualities
                channel_qualities = np.abs(ac_info[ue, :])

                # Sorting
                argsort_channel_qualities = np.flip(np.argsort(channel_qualities))

                # Store choices
                ue_choices[ue] = list(argsort_channel_qualities[:num_packets])

            elif access_policy == 'SMAP':

                # Get channel qualities
                channel_qualities = np.abs(ac_info[ue, :])

                # Take the best
                best_idx = np.argmax(channel_qualities)

                # NaNing
                channel_qualities[best_idx] = np.nan

                if len(channel_qualities[~np.isnan(channel_qualities)]) != 0:

                    # Compute inequality
                    inequality = channel_qualities**2 - self.decoding_snr
                    inequality[inequality < 0.0] = np.nan

                    if len(inequality[~np.isnan(inequality)]) != 0:

                        # Get minimum idx
                        min_idx = np.nanargmin(inequality)

                        # Store choices
                        ue_choices[ue] = [best_idx, min_idx]

                    else:
                        ue_choices[ue] = [best_idx, ]

                else:
                    ue_choices[ue] = [best_idx, ]


        return ue_choices

    def ul_transmission(self, channels_ul, ue_messages, ue_choices):

        # Extract number of UEs
        num_ues = ue_messages.shape[0]

        # Generate noise
        noise = np.sqrt(1/2) * (np.random.randn(self.num_slots, self.num_channel_uses) + 1j * np.random.randn(self.num_slots, self.num_channel_uses))

        # Prepare to save access attempts
        access_attempts = np.zeros((self.num_slots, self.num_channel_uses), dtype=np.complex_)

        # Prepare to sabe bipartite graph
        bigraph = []

        # Go through each UE
        for ue in range(num_ues):

            # Go through UE's choice
            for ac in ue_choices[ue]:

                # Obtain received signal at the AP
                access_attempts[ac] += channels_ul[ue, ac] * ue_messages[ue, :]

                # Store in the graph
                bigraph.append((ue, ac))

        # Add noise
        access_attempts += noise

        return access_attempts, bigraph


    def decoder(self, channels_ul, ue_messages, access_attempts, bigraph, mvu_error_ul=0):
        """
        Evaluates the number of successful access attempts of the random access method given the choices made by the UEs
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

        # Extract number of UEs
        num_ues = ue_messages.shape[0]

        # Prepare to save deconding results
        access_result = {}

        while True:

            # Get graph degree as a dictionary
            degrees = bigraph_degree(bigraph)

            # No singletons, we cannot decode -> break
            if not (1 in degrees.values()):
                break

            # Get a singleton
            singleton = [(ue_idx, ac_idx) for (ue_idx, ac_idx) in bigraph if degrees[ac_idx] == 1][0]

            # Correspoding indexes
            (ue_idx, ac_idx) = singleton

            # Compute SNR of the buffered signal
            buffered_snr = np.linalg.norm(access_attempts[ac_idx])**2

            # Check SIC condition
            if buffered_snr >= (self.num_channel_uses * self.decoding_snr):

                # Store results
                if not ue_idx in access_result.keys():
                    access_result[ue_idx] = []

                access_result[ue_idx].append(ac_idx)

                # Reconstruct UE's signal
                reconstruction_noise = (np.sqrt(mvu_error_ul/2) * (np.random.randn() + 1j * np.random.randn()))
                reconstructed_signal = (channels_ul[ue_idx, ac_idx] * ue_messages[ue_idx]) + reconstruction_noise

                # Identify other edges with the UE of interest
                ue_edges = [(ue, aa) for ue, aa in bigraph if ue == ue_idx]

                # Apply SIC
                for edge in ue_edges:

                    # Extract access slot
                    other_ac_idx = edge[1]

                    # Update buffered signal
                    access_attempts[other_ac_idx] -= reconstructed_signal

                    # Remove edges
                    bigraph.remove(edge)

            else:

                bigraph.remove((ue_idx, ac_idx))

        return access_result


class ACK(Block):

    def __init__(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float
    ):

        super().__init__(num_slots, num_channel_uses, num_silent_channel_uses, decoding_snr)

        self.codebook = None

    def set_codebook(self, detected_directions, ack_method=None):

        if ack_method == 'rand':

            self.codebook = np.pi/2 * np.random.rand(detected_directions.size)

            _, multiplier = np.unique(self.codebook, return_counts=True)
            multiplier = (multiplier >= 1).sum()
            self.multiplier = int(multiplier)

        elif ack_method == 'prec':

            self.codebook = np.repeat(detected_directions.mean(), detected_directions.size)

        elif ack_method == 'tdma':

            self.codebook = detected_directions

            _, multiplier = np.unique(self.codebook, return_counts=True)
            multiplier = (multiplier >= 1).sum()
            self.multiplier = int(multiplier)

    def dl_transmission(self, dl_channels, ack_messages, noise):

        # Compute received signal
        rx_signals = dl_channels * ack_messages + noise

        # Compute received SNR
        rx_snr = np.linalg.norm(rx_signals, axis=-1)**2

        # Compare with the threshold
        ack_success = rx_snr >= (self.num_channel_uses * self.decoding_snr)

        return ack_success.sum()

class Frame():

    def __init__(self):

        self.tr = None
        self.ac = None
        self.ack = None


    def init_training(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float
    ):

        self.tr = Training(num_slots, num_channel_uses, num_silent_channel_uses, decoding_snr)


    def init_access(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float,

    ):

        self.ac = Access(num_slots, num_channel_uses, num_silent_channel_uses, decoding_snr)


    def init_ack(
        self,
        num_slots : int,
        num_channel_uses : int,
        num_silent_channel_uses : int,
        decoding_snr : float
    ):

        self.ack = ACK(num_slots, num_channel_uses, num_silent_channel_uses, decoding_snr)


    def compute_throughput(self, access_policy, num_successful_attempts, ack_method=None, switch_time=0):
        """

        """

        # Compute durations
        tr_duration = self.tr.num_slots * (self.tr.num_channel_uses + switch_time)
        ac_duration = self.ac.num_slots * (self.ac.num_channel_uses + switch_time)
        ack_duration = 0

        if ack_method == 'prec':
            ack_duration = (self.ack.num_slots * self.ack.num_channel_uses) + switch_time

        elif ack_method == 'rand' or ack_method == 'tdma':
            ack_duration = (self.ack.num_slots * self.ack.num_channel_uses) + (self.ack.multiplier * switch_time)

        if access_policy == 'RCURAP':
            throughput = num_successful_attempts / (ac_duration + ack_duration)

        else:
            throughput = num_successful_attempts / (tr_duration + ac_duration + ack_duration)

        return throughput
