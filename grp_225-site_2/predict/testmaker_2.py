import pymatgen.core as mp
import sys
sys.path.append("/home/ruman/Desktop/codes/pymatgen/trainmaker")
import core_radii as cr
from shannon import shannon_lookup 
from math import log
import re
import random
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

orbitals = ["s","p","d","f"]
width = 10
sg = 225
ns = 2
print("formula".ljust(width) + "X_a0".ljust(width)+"X_b1".ljust(width) + "Z_a2".ljust(width)+"Z_b3".ljust(width) + "IE_a4".ljust(width)+"IE_b5".ljust(width) + "EA_a6".ljust(width)+"EA_b7".ljust(width) + "r_A8".ljust(width)+"r_B9".ljust(width) + "vale_A10".ljust(width)+"vale_B11".ljust(width) + "cr.r_A12".ljust(width)+"cr.r_B13".ljust(width) + "mb.x_A14".ljust(width)+"mb.x_B15".ljust(width) + "s_A16".ljust(width)+"p_A17".ljust(width)+"d_A18".ljust(width)+"f_A19".ljust(width) + "s_B20".ljust(width)+"p_B21".ljust(width)+"d_B22".ljust(width)+"f_B23".ljust(width) + "random24".ljust(width)+"random25".ljust(width) + "topoY".ljust(width))

formulae = []
topo = []
with open("../materiae.dat") as f:
    for line in f:
        words = line.strip().split(" ")
        formulae.append(words[5])
        topo.append("1" if words[4]=="TI" or words[4]=="TCI" else "0")

with MPRester("XUCu3DLWy99hFoeU3hAg0N62eyiEQENW") as mpr:
    props = ['material_id','formula_pretty','elements','structure']
    searc=mpr.summary.search(spacegroup_number=sg, num_sites=ns,fields=props)

for i in searc:
    structure = i.structure
    sg = SpacegroupAnalyzer(structure)
    sit = structure.sites
    wyk = sg.get_symmetry_dataset()['wyckoffs']
    line = i.formula_pretty.ljust(width)
    #print(structure.formula.replace(" ",""), end=' ')
    if sit[0].specie.X == sit[1].specie.X:
        continue
    
    if sit[0].specie.X < sit[1].specie.X:
        el = [sit[0].specie,sit[1].specie]
    else:
        el = [sit[1].specie,sit[0].specie]
        
    #electronegativity
    for j in range(len(el)):
        X=(str(el[j].X))
        line = line + X.ljust(width)
    #atomic number
    for j in range(len(el)):
        Z=(str(el[j].Z))
        line = line + Z.ljust(width)
    #ionization energy
    for j in range(len(el)):
        IE=(str(round(el[j].ionization_energy,5)))
        line = line + IE.ljust(width)
    #electron affinity
    for j in range(len(el)):
        EA=(str(round(el[j].electron_affinity,5)))
        line = line + EA.ljust(width)
        
    #atomic_radius
    for j in range(len(el)):
        r=(str(round(el[j].atomic_radius,5)))
        line = line + r.ljust(width)
        
    #valence electron
    for j in range(len(el)):
        line = line + str(cr.valence_lookup(el[j].symbol).ljust(width))
    
    #core radius - sum of s- and p- orbital radii
    for j in range(len(el)):    
        line = line + str(cr.core_lookup(el[0].symbol).ljust(width))
        
    #Martynov-Batsanov electronegativity scales
    for j in range(len(el)): 
        line = line + str(cr.mbchi_lookup(el[0].symbol).ljust(width))
        
    #electron count in outermost orbitals
    for j in el:
        for o in orbitals:
            count = re.findall(o+"\d+",j.electronic_structure)
            if len(count) == 0:
                line = line + "0".ljust(width)
            else:
                line = line + count[0][1:].ljust(width)       
                
    try:
        ind = formulae.index(str(i.material_id))
        line = line + str(topo[ind]).ljust(width)
    except:
        line = line + str(-1).ljust(width).ljust(width)
        
    line = line + str(i.material_id)

    print(line)
