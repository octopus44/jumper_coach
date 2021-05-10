import os
import unittest
from sys import argv


def main(arg):
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