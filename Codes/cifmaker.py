from mp_api.client import MPRester
import pymatgen.core as mp
from pymatgen.io import cif

with MPRester("XUCu3DLWy99hFoeU3hAg0N62eyiEQENW") as mpr:
    with open("../validation.dat") as f:
        for line in f:
            structure = mpr.get_structure_by_material_id(line.strip())
            cif_ready = cif.CifWriter(structure)
            cif_ready.write_file(line.strip()+".cif")
           
