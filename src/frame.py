import numpy as np

import networkx as nx
from networkx.algorithms import bipartite

########################################
# Private functions
########################################
def graph_degree(edge_list):
    """
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

        self.set_codebook()

    def set_codebook(self):
        """

        """

        self.codebook = np.linspace(0, np.pi/2, self.num_slots)


    def messages(self, num_ues):

        return np.sqrt(1/2) * (np.random.randn(num_ues, self.num_channel_uses) + 1j * np.random.randn(num_ues, self.num_channel_uses))

    def access_policy(self, ac_info, num_packets=1, access_policy=None):
        """

        """


        # Extract number of UEs
        num_ues = ac_info.shape[0]

        if num_packets > self.num_slots:
            num_packets = self.num_slots

        # Prepare to save chosen access slots
        chosen_access_slots = {ac_slot: [] for ac_slot in range(self.num_slots)}

        # Choose access policy
        if access_policy == 'RCURAP':

            # Go through all UEs
            for ue in range(num_ues):

                choices = []
                temp = 0

                while True:

                    # Flip a fair coin
                    test = np.random.rand() > 1/2

                    # Store
                    if test and not temp in choices:
                        choices.append(temp)

                    # Increment
                    temp += 1

                    # Stopping criterion
                    if len(choices) == num_packets:
                        break

                    # Start again
                    if temp == self.num_slots:
                        temp = 0

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        elif access_policy == 'RCARAP':

            # Go through all UEs
            for ue in range(num_ues):

                # Get probability mass function
                pmf = np.abs(ac_info[ue, :]) / np.abs(ac_info[ue, :]).sum()

                choices = []
                temp = 0

                while True:

                    # Flip a unfair coin
                    test = np.random.rand() > pmf[temp]

                    # Store
                    if test and not temp in choices:
                        choices.append(temp)

                    # Increment
                    temp += 1

                    # Stopping criterion
                    if len(choices) == num_packets:
                        break

                    # Start again
                    if temp == self.num_slots:
                        temp = 0

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        elif access_policy == 'RGSCAP':

            # Go through all UEs
            for ue in range(num_ues):

                # Get channel qualities
                channel_qualities = np.abs(ac_info[ue, :])

                # Sorting
                argsort_channel_qualities = np.flip(np.argsort(channel_qualities))

                # Choose
                choices = list(argsort_channel_qualities[:num_packets])

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        elif access_policy == 'SMAP':

            # Go through all UEs
            for ue in range(num_ues):

                choices = []

                # Get channel qualities
                channel_qualities = np.abs(ac_info[ue, :])

                # Take the best
                best_idx = np.argmax(channel_qualities)

                # Store and delete
                choices.append(best_idx)
                channel_qualities[best_idx] = np.nan

                if len(channel_qualities) != 0:

                    # Compute inequality
                    inequality = channel_qualities**2 - self.decoding_snr
                    inequality[inequality < 0.0] = np.nan

                    # Get minimum idx
                    try:
                        min_idx = np.nanargmin(inequality)
                        choices.append(min_idx)
                    except:
                        pass

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        return chosen_access_slots

    def ul_transmission(self, num_ues, channels_ul, messages, chosen_access_slots):

        # Prepare to save access attempts
        buffered_access_attempts = np.zeros((self.num_slots, self.num_channel_uses), dtype=np.complex_)

        # Go through each access slot
        for ac_slot in chosen_access_slots.keys():

            # Extract number of colliding UEs
            colliding_ues = chosen_access_slots[ac_slot]

            if len(colliding_ues) == 0:
                continue

            # Generate noise
            noise = np.sqrt(1/2) * (np.random.randn(self.num_channel_uses) + 1j * np.random.randn(self.num_channel_uses))

            # Obtain received signal at the AP
            buffered_access_attempts[ac_slot] = np.sum(channels_ul[colliding_ues, ac_slot][:, None] * messages[colliding_ues, :], axis=0) + noise

        return buffered_access_attempts


    def decoder(self, num_ues, chosen_access_slots, messages, buffered_access_attempts):

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

        # Initialize decoding results
        access_dict = {}

        # Enumerate UEs
        enumeration_ues = np.arange(0, num_ues).astype(int)

        # Enumerate access slots
        enumeration_ac_slots = [str('A') + str(ac_slot) for ac_slot in range(self.num_slots)]

        # Create edge list: each edge is a tuple
        edge_list = []

        # Go through all access slots
        for ac_slot in chosen_access_slots:

            # Extract colliding UEs
            colliding_ues = chosen_access_slots[ac_slot]

            if len(colliding_ues) == 0:
                continue

            # Go through all colliding UEs
            for ue in colliding_ues:
                edge_list.append((ue, enumeration_ac_slots[ac_slot]))

        while True:

            # Get graph degree as a dictionary
            degrees = graph_degree(edge_list)

            # No singletons, we cannot decode -> break
            if not (1 in degrees.values()):
                break

            # Get a singleton
            singleton = [(ue_idx, ac_slot) for (ue_idx, ac_slot) in edge_list if degrees[ac_slot] == 1][0]

            # Correspoding indexes
            (ue_idx, ac_slot) = singleton

            # Get actual index
            ac_slot_idx = int(ac_slot[1:])

            # Compute SNR of the buffered signal
            buffered_snr = np.linalg.norm(buffered_access_attempts[ac_slot_idx])**2

            # Check SIC condition
            if buffered_snr >= (self.num_channel_uses * self.decoding_snr):

                # Store results
                if not ac_slot_idx in access_dict.keys():
                    access_dict[ac_slot_idx] = []

                access_dict[ac_slot_idx].append(ue_idx)

                # Identify other edges with dagger UE
                edge_list_success = [(ue, aa) for ue, aa in edge_list if ue == ue_idx]

                # Reconstruct UE's signal
                hat_channel = buffered_access_attempts[ac_slot_idx].mean()
                hat_signal = hat_channel * np.squeeze(messages[ue_idx])

                # Go through buffered signals that contain the successful UE and update them
                for edge in edge_list_success:

                    if edge != (ue_idx, ac_slot):

                        # Other access slot index
                        other_ac_slot = edge[1]
                        other_ac_slot_idx = int(other_ac_slot[1:])

                        # Update buffered signal
                        buffered_access_attempts[other_ac_slot_idx] -= hat_signal

                    # Remove edges
                    edge_list.remove(edge)

            else:

                try:
                    edge_list.remove((ue_idx, ac_slot))
                except:
                    pass

        return access_dict


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

    def set_codebook(self, detected_directions):

        if self.num_slots == 1:

            self.codebook = np.linspace(0, np.pi/2, self.num_slots)


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


    def compute_throughput(self, access_policy, num_successful_attempts):
        """

        """

        # Compute durations
        tr_duration = self.tr.num_slots * self.tr.num_channel_uses + self.tr.num_slots * self.tr.num_silent_channel_uses
        ac_duration = self.ac.num_slots * self.ac.num_channel_uses + self.ac.num_slots * self.ac.num_silent_channel_uses

        # Compute throughput
        if access_policy == 'RCURAP':
            throughput = num_successful_attempts / ac_duration
        else:
            throughput = num_successful_attempts / (tr_duration + ac_duration)

        return throughput
