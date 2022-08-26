import numpy as np

import networkx as nx
from networkx.algorithms import bipartite


def graph_degree(edge_list):

    degrees = {}

    for edge in edge_list:

        if edge[1] in degrees.keys():
            degrees[edge[1]] += 1

        else:
            degrees[edge[1]] = 1

    return degrees


class Block:

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

    def access_policy(self, snr, ac_info, num_repetitions=1, policy='RCURAP'):

        # Extract number of UEs
        num_ues = ac_info.shape[0]

        if num_repetitions > self.num_slots:
            num_repetitions = self.num_slots

        # Prepare to save chosen access slots
        chosen_access_slots = {ac_slot: [] for ac_slot in range(self.num_slots)}

        # Choose access policy
        if policy == 'RCURAP':

            # Go through all UEs
            for ue in range(num_ues):

                choices = []
                temp = 0

                while True:

                    test = np.random.randn() > 1/2

                    if test:
                        if not temp in choices:
                            choices.append(temp)

                    temp += 1

                    if len(choices) >= num_repetitions:
                        break

                    if temp >= self.num_slots:
                        temp = 0

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        elif policy == 'RCARAP':

            # Go through all UEs
            for ue in range(num_ues):

                # Get probability mass function
                pmf = np.abs(ac_info[ue, :]) / np.abs(ac_info[ue, :]).sum()

                choices = []
                temp = 0

                while True:

                    test = np.random.randn() > pmf[temp]

                    if test:
                        if not temp in choices:
                            choices.append(temp)

                    temp += 1

                    if len(choices) >= num_repetitions:
                        break

                    if temp >= self.num_slots:
                        temp = 0

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        elif policy == 'RGSCAP':

            # Go through all UEs
            for ue in range(num_ues):

                # Get channel qualities
                channel_qualities = np.abs(ac_info[ue, :])

                # Sorting
                argsort_channel_qualities = np.flip(np.argsort(channel_qualities))

                # Choose
                choices = list(argsort_channel_qualities[:num_repetitions])

                for cc in choices:
                    chosen_access_slots[cc].append(ue)

        elif policy == 'SMAP':

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

                if len(channel_qualities) == 0:

                    for cc in choices:
                        chosen_access_slots[cc].append(ue)

                else:

                    # Compute inequality
                    inequality = (snr * channel_qualities**2) - self.decoding_snr
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


    def decoder(self, snr, num_ues, chosen_access_slots, buffered_access_attempts, messages):

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
        # Nothing to compute
        if num_ues == 1:
            return chosen_access_slots

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

            # Get singletons
            singletons = [ac_slot for ac_slot in degrees.keys() if degrees[ac_slot] == 1]

            # Go through all access-slot singletons
            for ac_slot in singletons:

                # Get index
                ac_slot_idx = int(ac_slot[1:])

                # Compute SNR of the buffered signal
                buffered_snr = np.linalg.norm(buffered_access_attempts[ac_slot_idx])**2

                # Get UE index
                ue_idx = chosen_access_slots[ac_slot_idx][0]

                if len(list(access_dict.values())) != 0:

                    if ue_idx in np.concatenate(list(access_dict.values())).flat:

                        # Update dict
                        chosen_access_slots[ac_slot_idx].remove(ue_idx)

                        continue

                # Check SIC condition
                if buffered_snr >= (self.num_channel_uses * self.decoding_snr):

                    # Store results
                    if not ac_slot_idx in access_dict.keys():
                        access_dict[ac_slot_idx] = []
                    access_dict[ac_slot_idx].append(ue_idx)

                    # Identify other edges with dagger UE
                    edge_list_success = [(ue, aa) for ue, aa in edge_list if ue == ue_idx]

                    # Reconstruct UE's signal
                    hat_channel_gain = (1 / (self.num_channel_uses * np.sqrt(snr))) * buffered_access_attempts[ac_slot_idx].sum()
                    hat_signal = np.sqrt(snr) * hat_channel_gain * np.squeeze(messages[ue_idx])

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

                    edge_list.remove((ue_idx, ac_slot))

                # Update
                chosen_access_slots[ac_slot_idx].remove(ue_idx)

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
        decoding_snr : float
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
