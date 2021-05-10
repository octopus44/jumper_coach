import os
import unittest
from sys import argv

def recommend_cue(poss_cue):
    if poss_cue == "good job":
        print("Understand if we are going to have a good relationship then my suggestions may seem hard to achieve.  Together getting good requires looking hard at the jump and finding ways to improve.")

def main(arg):
    script, filename, outfile, debug = argv
    print("================================")
    script, debug = argv
    if debug == 'False': debug = bool(0)
    if debug == 'True': debug = bool(1)
    if debug: print({debug})
    os.system("pwd")
    print("================================")

if __name__ == "__main__":
    main(argv)
    unittest.main(argv)