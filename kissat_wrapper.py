"""
Simple Python wrapper for Kissat.
"""
import os
import subprocess

class Kissat:


    def __init__(self, cnf):
        self.cnf = cnf


    def solve(self):
        """
        Solve the CNF in self.cnf
        """

        # 1. write cnf to (temp) DIMCACS cnf file
        tmp_cnf_file = '_tmp.cnf'
        with open(tmp_cnf_file, 'w') as f:
            f.write(self.cnf.dimacs(self.cnf.get_variable_map()))

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
        lines = output.stdout.split('\n')
        for line in lines:
            if line.startswith('s'):
                if line[:5] == 's SAT':
                    self.is_sat = True
                elif line[:5] == 's UNS':
                    self.is_sat = False
                else:
                    print("Error parsing Kissat output")
            elif line.startswith('c process-time'):
                self.solve_time = float(line.split()[2])
