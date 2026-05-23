import numpy as np
import re

def write_kicad_coil(pcb_file, coildata, net):
    if(type(coildata) == 'str'):
        coildata = np.load(coildata)

    def _find_coil(): #look for coil annotation comments
        coil_start_line = "##### Coil Start #####"
        coil_end_line =  "##### Coil End #####"
        coil_start = -1
        coil_end = -1
        with open(pcb_file) as f:
            for idx, line in enumerate(f):
                if(line.strip() == coil_start_line):
                    coil_start = idx
                elif(line.strip() == coil_end_line):
                    coil_end = idx
        
        return coil_start, coil_end
    
    print(_find_coil())

if __name__ == '__main__':
    pcb = 'Medium_Magnet/Rect_Grads/Rect_Grads_PCB/Rect_Grads_PCB.kicad_pcb'
    write_kicad_coil(pcb, np.zeros(10), 1)
