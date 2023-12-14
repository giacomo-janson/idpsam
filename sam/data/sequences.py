aa_list = list("QWERTYIPASDFGHKLCVNM")
aa_one_letter = set(aa_list)

aa_three_letters = ["GLN", "TRP", "GLU", "ARG", "THR",
                    "TYR", "ILE", "PRO", "ALA", "SER",
                    "ASP", "PHE", "GLY", "HIS", "LYS",
                    "LEU", "CYS", "VAL", "ASN", "MET"]

aa_one_to_three_dict = {
    "G": "GLY", "A": "ALA", "L": "LEU", "I": "ILE", "R": "ARG", "K": "LYS",
    "M": "MET", "C": "CYS", "Y": "TYR", "T": "THR", "P": "PRO", "S": "SER",
    "W": "TRP", "D": "ASP", "E": "GLU", "N": "ASN", "Q": "GLN", "F": "PHE",
    "H": "HIS", "V": "VAL", "X": "XXX"
}

aa_three_to_one_dict = {
    "GLY": "G", "ALA": "A", "LEU": "L", "ILE": "I", "ARG": "R", "LYS": "K",
    "MET": "M", "CYS": "C", "TYR": "Y", "THR": "T", "PRO": "P", "SER": "S",
    "TRP": "W", "ASP": "D", "GLU": "E", "ASN": "N", "GLN": "Q", "PHE": "F",
    "HIS": "H", "VAL": "V", "XXX": "X"
}


pos_aa = set(("K", "R"))
neg_aa = set(("D", "E"))

def get_net_q_res(seq):
    q = 0
    for r in seq:
        if r in pos_aa:
            q += 1
        elif r in neg_aa:
            q -= 1
    return q/len(seq)