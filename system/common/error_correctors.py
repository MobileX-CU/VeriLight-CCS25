"""
Various error correction code: Reed Solomon, Viterbi, Concatenated Reed Solomon + Viterbi (used in paper)
"""
import unireedsolomon as rs
import random
import glob
from viterbi import bit_encoder_K2, bit_encoder_K5, stream_encoder, Decoder, Channel
from bitstring_utils import bitstring_to_ascii, ascii_to_bitstring
import numpy as np

#######################
# REED 
########################
class ReedSolomon(object):
   
    def __init__(self, n, k):
        """
        n : codeword size (a codeword consists of n bytes, k of which are data and n - k of which are parity)
        k : number of data bytes
        """
        self.n = n
        self.k = k
        self.coder = rs.RSCoder(n, k)
    
    def strength(self):
        """
        https://www.cs.cmu.edu/~guyb/realworld/reedsolomon/reed_solomon_codes.html
        """
        return (self.n - self.k)/2 #number of error symbols that can be corrected

    def bitstring_to_codewords(self, bitstring):
        chunk_size = self.n * 8
        codewords = []
        for i in range(0, len(bitstring), chunk_size):
            chunk = bitstring[i:i+chunk_size]
            codeword = bitstring_to_ascii(chunk)
            codewords.append(codeword)
        return codewords

    def codewords_to_bitstring(self, codewords):
        bitstring = ""
        for c in codewords:
            bitstring += ascii_to_bitstring(c)
        return bitstring

    def encode(self, message_bitstring):
        """
        message_bitsrting : str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                            a char "1" or "0"
        """

        """
        convert message bitstring to ascii for compatibility with the ReedSolomon coder from unireedsolomom (and other implementations),
        which expects input in the form of 8-bit symbols 
        """
        message_symbols = bitstring_to_ascii(message_bitstring)

        #split symbols into chunks of size n and obtain codeword for each 
        #if the last chunk has less than n bytes, pad it with 0x00 ASCII character until it is of size n
        codewords = []
        for i in range(0, len(message_symbols), self.k):
            if i + self.k > len(message_symbols):
                num_padding_symbols = self.k - (len(message_symbols) - i)
                chunk = message_symbols[i:]
                for n in range(num_padding_symbols):
                    chunk += chr(0)
            else:
                chunk = message_symbols[i:i+self.k]
            codeword = self.coder.encode(chunk)
            codewords.append(codeword)
        return codewords

    def encode_payload(self, message_bitsrting):
        """
        for feature_proc serialization to use - take in bitstring, output bitstring of concatenated codewords
        """
        codewords = self.encode(message_bitsrting)
        enc_bitstring = self.codewords_to_bitstring(codewords)
        return enc_bitstring

    def decode(self, codewords):
        recovered_bitstring = ""
        correctable = True
        for i, c in enumerate(codewords):
            try:
                recovered_chunk_symbols = self.coder.decode(c, nostrip = True)[0]
            except Exception as e:
                print(f"Reed Solomon can't corect codeword {i}. {e}. Recovered message will just be uncorrected data portion of codeword.")
                recovered_chunk_symbols = c[:self.k]
                correctable = False
            recovered_chunk_bitstring = ascii_to_bitstring(recovered_chunk_symbols)
            recovered_bitstring += recovered_chunk_bitstring
        return recovered_bitstring, correctable
    
    def decode_payload(self, encoded_bitstring):
        codewords = self.bitstring_to_codewords(encoded_bitstring)
        decoded_bitstring, correctable = self.decode(codewords)
        return decoded_bitstring, correctable

    def noise_codewords(self, codewords, flip_probability):
        #add noise to the codewords by flipping bits
        noised_codewords = []
        num_bit_flips = 0
        c_bitstrings = self.codewords_to_bitstrings(codewords)
        for c in c_bitstrings:
            noised_c_bitstring = ""
            for i in c:
                if random.randint(0, 100) < flip_probability*100:
                    num_bit_flips += 1
                    if i == "1":
                        noised_c_bitstring += "0"
                    else:
                        noised_c_bitstring += "1"
                else:
                    noised_c_bitstring += i
            noised_c = bitstring_to_ascii(noised_c_bitstring)
            noised_codewords.append(noised_c)
        return noised_codewords, num_bit_flips

    def add_payload_noise(self, encoded_payload, flip_probability):
        num_bit_flips = 0
        noised_payload = ""
        for i in encoded_payload:
            if random.randint(0, 100) < flip_probability*100:
                num_bit_flips += 1
                if i == "1":
                    noised_payload += "0"
                else:
                    noised_payload += "1"
            else:
                noised_payload += i
        return noised_payload, num_bit_flips


    def check(self, test_message_bitstring):
        codewords = self.encode(test_message_bitstring)
        recovered_bitstring = self.decode(codewords)
        err = sum(c1!=c2 for c1,c2 in zip(test_message_bitstring, recovered_bitstring))
        print("----- REED SOLOMON CHECK -----")
        print(f"Input bitstring:     {test_message_bitstring}")
        print(f"Recovered bitstring: {recovered_bitstring}")
        print(f"Total errors: {err} bits") 


#######################
# VITERBI
########################
class SoftViterbi(object):
    def __init__(self, k):
        self.k = k
        if k == 2:
            self.bit_encoder = bit_encoder_K2
        elif k == 5:
            self.bit_encoder = bit_encoder_K5

    def encode_payload(self, input_bitstring):
        """
        Modified version of stream_encoder in viterbi_utils.py, adapted to accept/output
        bitstrings rather than lists
        """
        input_stream = [int(i) for i in input_bitstring]
        list_output = stream_encoder(self.k, input_stream)
        encoded_bitstring = ""
        for el in list_output:
            encoded_bitstring += str(el[0])
            encoded_bitstring += str(el[1])
        return encoded_bitstring

    def decode_payload(self, input_probs):
        input_stream = []
        correctable = True
        if len(input_probs) % 2 != 0 or len(input_probs) < self.k:
            correctable = False
        else:
            for i in range(0, len(input_probs), 2):
                input_stream.append([input_probs[i], input_probs[i + 1]])
            try:
                dec_list = Decoder(self.k, input_stream, False)
            except:
                correctable = False
        if not correctable:
            print(f"Viterbi decoder failed. Returning {(len(input_stream) - self.k)} 0s")
            dec_bitstring = "0" * (len(input_probs) - self.k)
        else:
            dec_bitstring = ""
            for b in dec_list:
                dec_bitstring += str(b)
        return dec_bitstring, correctable


############################
# CONCATENATED RS + VITERBI
############################
class ConcatenatedViterbiRS(object):
    """
    Concatenated error correction: first Viterbi, then RS on Viterbi-encoded
    """
    def __init__(self, v_k, n, rs_k):
        """
        Initialize a concatenated error correction code with an inner Viterbi 
        encoder and an outer Reed Solomon (RS) encoder

        Parameters:
            v_k (int): Viterbi encoding k
            n (int): Reed Solomon codeword size (a codeword consists of n bytes, 
                     k of which are data and n - k of which are parity)
            rs_k (int): number of data bytes input to Reed Solomon encoder
        
        Returns:
            None
        """
        self.viterbi_coder = SoftViterbi(v_k)
        self.rs_coder = ReedSolomon(n, rs_k)
    
    def encode_payload(self, input_bitstring):
        """
        First do Viterbi encoding on input bitstring, then run Reed-Solomon 
        encoding on the Viterbi-encoded bitstring

        Parameters:
            input_bitstring (str): the input bitstring to be encoded. 
        
        Returns:
            final_encoded (str): the final encoded bitstring (after both Viterbi and RS encoding)
            rs_encoded (str): the Reed Solomon coded bitstring (i.e., pre-Viterbi encoding, post-RS encoding)

        Note: Here, a bitstring is a string of 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not a char
              e.g., "100010010010"
        """

        rs_encoded = self.rs_coder.encode_payload(input_bitstring)
        final_encoded = self.viterbi_coder.encode_payload(rs_encoded)
        return final_encoded, rs_encoded

    def decode_payload(self, input_probs):
        """
        First do soft Viterbi decoding on input probabilities to recover a Reed-Solomon codeword.
        Then run RS decoding on codeword to obtain final data

        Parameters:
            input_probs (list of floats): each float representing the probability of a bit being a 1
        
        Returns:
            decoded_bitstring (str): the decoded bitstring (after both Viterbi and RS decoding)
            pred_rs_encoded (str): the Reed Solomon coded bitstring (i.e., post-Viterbi decoding, pre-RS decoding)
            correctable (bool): whether the error correction was successful, considering any errors encountered Viterbi and/or RS decoding
        
        Note: Here, a bitstring is a string of 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not a char
              e.g., "100010010010"
        """
        pred_rs_encoded, vit_corectable = self.viterbi_coder.decode_payload(input_probs)
        decoded_bitstring, rs_correctable = self.rs_coder.decode_payload(pred_rs_encoded)
        correctable = vit_corectable and rs_correctable
        return decoded_bitstring, pred_rs_encoded, correctable

#######################
# TESTING/SIMULATION
########################
class ReedNoisyChannelSimulator(object):
    def __init__(self, error_corrector):
        """
        Accept either ReedMuller or ReedSolom class above
        """
        self.error_corrector = error_corrector
        if type(self.error_corrector) == ReedSolomon:
            self.error_corrector_name = "REED SOLOMON"
        else:
            self.error_corrector_name = "REED MULLER"
    
    def simulate(self, message_bitstring, error_probability):
        encoded_payload = self.error_corrector.encode_payload(message_bitstring)
        noised_payload, num_bit_flips = self.error_corrector.add_payload_noise(encoded_payload, error_probability)
        decoded_payload = self.error_corrector.decode_payload(noised_payload)
        err = sum(c1!=c2 for c1,c2 in zip(message_bitstring, decoded_payload))
        print(f"-----{self.error_corrector_name} NOISY CHANNEL SIMULATOR -----")
        print(f"Input bitstring:     {message_bitstring}")
        print(f"Recovered bitstring: {decoded_payload}")
        print(f"{num_bit_flips} bits flipped during noise addition.")
        print(f"Total errors: {err} bits")


def evaluate_error_correction(ec, payloads_path, n, exclude_seqs = None):
    """
    evaluate the impact of error correction on the decoded sequences from the core unit
    camera
    """
    unencoded_payload_paths = glob.glob(f"{payloads_path}/unencoded_*.txt")

    tot_raw_bers = 0
    tot_rec_bers = 0
    num_seqs_evaluated = 0
    for p in unencoded_payload_paths:
        seq_num = p.split("unencoded_")[1].split(".txt")[0]
        if exclude_seqs is not None:
            if int(seq_num) in exclude_seqs:
                continue

        f = open(p)
        unencoded = f.read()
        f.close()

        f = open(f"../embedding/trial_temp/embedded_data/{seq_num}.txt")
        encoded = f.read()[:(n*8)] #remove any padded added to payload to reach full bandwidth
        f.close()

        try:
            f = open(f"../embedding/trial_temp/embedded_data/pred_{seq_num}.txt")
            pred = f.read()[:(n*8)] #remove any padded added to payload to reach full bandwidth
            f.close()
        except:
            # print(f"No pred for seq {seq_num}")
            continue
        
        pred_message_len= len(pred)
        encoded = encoded[:pred_message_len]
        
        num_raw_bit_errors = sum(c1!=c2 for c1,c2 in zip(pred, encoded))
        tot_raw_bers += num_raw_bit_errors

        decoded = ec.decode_payload(pred)
        recovered_num_bit_errors = sum(c1!=c2 for c1,c2 in zip(decoded, unencoded))
        tot_rec_bers += recovered_num_bit_errors

        num_seqs_evaluated += 1
    
    print(f"{num_seqs_evaluated} seqs evaluated.")
    print(f"{tot_raw_bers} bit errors in raw payload.")
    print(f"{tot_rec_bers} bit errors after error correction.")

def concatenated_noisy_channel_simulator(ec, input_bitstring, eb_no_db):
    enc = ec.encode_payload(input_bitstring)
    temp = Channel([[int(enc[i]), int(enc[i + 1])] for i in range(0, len(enc), 2)], eb_no_db, False, 0.5)
    test_channel_output= []
    for t in temp:
        test_channel_output.append(t[0])
        test_channel_output.append(t[1])
    dec = ec.decode_payload(test_channel_output)
    err = sum(c1!=c2 for c1,c2 in zip(input_bitstring, dec))
    return err

#######################
# UTILS
########################
def get_rs_params(data_capacity, raw_payload_size, viterbi_k = None):
    """
    data_capacity : data capacity of a single window, in bits
    payload_size : size of raw payload (i.e., just window and hash bits), in bits
    viterbi_k : k associated with Viterbi encoder, if using a concatenated error correction scheme with RS then Viterbi. 
    If none, assume the error correction is purely Reed Solomon
    """
    rs_capacity = data_capacity
    if viterbi_k is not None:
        rs_capacity -= (viterbi_k*2)
        rs_capacity /= 2
    rs_n = int(rs_capacity / 8)
    rs_k = int(np.ceil(raw_payload_size / 8))
    return rs_n, rs_k
    