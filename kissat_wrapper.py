"""
Simple Python wrapper for Kissat.
"""
import os
import subprocess
import random

class Kissat:


    def __init__(self, dimacs_string : str):
        self.dimacs = dimacs_string

    def solve(self):
        """
        Solve the CNF in self.cnf
        """

        # 1. write cnf to (temp) DIMCACS cnf file
        tmp_cnf_file = '_tmp_' + str(random.randint(0,2**31)) + '.cnf'
        with open(tmp_cnf_file, 'w') as f:
            f.write(self.dimacs)

        # 2. run and parse result
        res = subprocess.run(["./extern/kissat/build/kissat", tmp_cnf_file], capture_output=True, text=True)
        self._parse_kissat_output(res)        

        # 3. remove temp cnf file
        os.remove(tmp_cnf_file)

        return self.is_sat


    def _parse_kissat_output(self, output):
        """
        Get relevant information from kissat console output
        """
        self.model = []
        lines = output.stdout.split('\n')
        for line in lines:
            if line.startswith('s'):
                if line[:5] == 's SAT':
                    self.is_sat = True
                elif line[:5] == 's UNS':
                    self.is_sat = False
                else:
                    print("Error parsing Kissat output")
            elif line.startswith('v'):
                for lit in line[1:].split():
                    if int(lit) != 0:
                        self.model.append(int(lit))
            elif line.startswith('c process-time'):
                self.solve_time = float(line.split()[-2])
