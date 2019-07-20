'''
Created by JordiKai Watanabe-Inouye
Read & formats profile results from a file
The cmd to run this is:
    python3 read_profiles.py profile_uvd.txt
'''

import pstats
import sys

def main():
    print(" Profiling", sys.argv[1])
    p = pstats.Stats(sys.argv[1])
    p.strip_dirs().sort_stats('time').print_stats(20)

if __name__ == "__main__": main()
