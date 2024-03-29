data = [['H', '1', '2.10', '1.25'], ['Li', '1', '0.90', '1.61'], ['Be', '2', '1.45', '1.08'], ['B', '3', '1.90', '0.795'], ['C', '4', '2.37', '0.64'], ['N', '5', '2.85', '0.54'], ['O', '6', '3.32', '0.465'], ['F', '7', '3.78', '0.405'], ['Na', '1', '0.89', '2.65'], ['Mg', '2', '1.31', '2.03'], ['Al', '3', '1.64', '1.675'], ['Si', '4', '1.98', '1.42'], ['P', '5', '2.32', '1.24'], ['S', '6', '2.65', '1.10'], ['Cl', '7', '2.98', '1.01'], ['K', '1', '0.80', '3.69'], ['Ca', '2', '1.17', '3.00'], ['Sc', '3', '1.50', '2.75'], ['Ti', '4', '1.86', '2.58'], ['V', '5', '2.22', '2.43'], ['Cr', '6', '2.00', '2.44'], ['Mn', '7', '2.04', '2.22'], ['Fe', '8', '1.67', '2.11'], ['Co', '9', '1.72', '2.02'], ['Ni', '10', '1.76', '2.18'], ['Cu', '11', '1.08', '2.04'], ['Zn', '12', '1.44', '1.88'], ['Ga', '3', '1.70', '1.695'], ['Ge', '4', '1.99', '1.56'], ['As', '5', '2.27', '1.415'], ['Se', '6', '2.54', '1.285'], ['Br', '7', '2.83', '1.20'], ['Rb', '1', '0.80', '4.10'], ['Sr', '2', '1.13', '3.21'], ['Y', '3', '1.41', '2.94'], ['Zr', '4', '1.70', '2.825'], ['Nb', '5', '2.03', '2.76'], ['Mo', '6', '1.94', '2.72'], ['Tc', '7', '2.18', '2.65'], ['Ru', '8', '1.97', '2.605'], ['Rh', '9', '1.99', '2.52'], ['Pd', '10', '2.08', '2.45'], ['Ag', '11', '1.07', '2.375'], ['Cd', '12', '1.40', '2.215'], ['In', '3', '1.63', '2.05'], ['Sn', '4', '1.88', '1.88'], ['Sb', '5', '2.14', '1.765'], ['Te', '6', '2.38', '1.67'], ['I', '7', '2.76', '1.585'], ['Cs', '1', '0.77', '4.31'], ['Ba', '2', '1.08', '3.402'], ['La', '3', '1.35', '3.08'], ['Hf', '4', '1.73', '2.91'], ['Ta', '5', '1.94', '2.79'], ['W', '6', '1.79', '2.735'], ['Re', '7', '2.06', '2.68'], ['Os', '8', '1.85', '2.65'], ['Ir', '9', '1.87', '2.628'], ['Pt', '10', '1.91', '2.70'], ['Au', '11', '1.19', '2.66'], ['Hg', '12', '1.49', '2.41'], ['Tl', '3', '1.69', '2.235'], ['Pb', '4', '1.92', '2.09'], ['Bi', '5', '2.14', '1.997'], ['Po', '6', '2.40', '1.90'], ['At', '7', '2.64', '1.83'], ['Fr', '1', '0.70', '4.37'], ['Ra', '2', '0.90', '3.53'], ['Ac', '3', '1.10', '3.12'], ['Ce', '3', '1.1', '4.50'], ['Pr', '3', '1.1', '4.48'], ['Nd', '3', '1.2', '3.99'], ['Pm', '3', '1.15', '3.99'], ['Sm', '3', '1.2', '4.14'], ['Eu', '3', '1.15', '3.94'], ['Gd', '3', '1.1', '3.91'], ['Tb', '3', '1.2', '3.89'], ['Dy', '3', '1.15', '3.67'], ['Ho', '3', '1.2', '3.65'], ['Er', '3', '1.2', '3.63'], ['Tm', '3', '1.2', '3.60'], ['Yb', '3', '1.1', '3.59'], ['Lu', '3', '1.2', '3.37'], ['Th', '3', '1.3', '4.98'], ['Pa', '3', '1.5', '4.96'], ['U', '3', '1.7', '4.72'], ['Np', '3', '1.3', '4.93'], ['Pu', '3', '1.3', '4.91'], ['Am', '3', '1.3', '4.89']]

def core_lookup(atom):  #core radius - sum of s and p - orbital
    for i in data:
        if i[0] == str(atom):
            return i[3]
    return "None"
    
def valence_lookup(atom): #valence number of electrons
    for i in data:
        if i[0] == str(atom):
            return i[1]
    return "None"
    
def mbchi_lookup(atom):   #Matynov-Batsanovelectronegativity
    #print(atom)
    for i in data:
        if i[0] == str(atom):
            return i[2]
    return "None"