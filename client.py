#
# client.py
#
# Client to the token processor server.
# Provide a text file for processing.
#
# Homework 1
# Course: CS 441, Fall 2024, UIC
# Author: Himanshu Dongre
#
import sys
import argparse

from omniORB import CORBA, PortableServer
import TokenProcessor


MASTER_IOR_FILE = "master_IOR.txt"


def main():
    orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)

    master_ior = read_master_ior()
    master_obj = orb.string_to_object(master_ior)._narrow(TokenProcessor.Master)

    # Parse command-line args
    parser = argparse.ArgumentParser(
        description="Client to the token processor server that takes a filename or string as input.",
        usage="client.py (-f FILE | -s STRING)"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help="Specify a file path as input.")
    group.add_argument('-s', '--string', help="Specify a string as input.")

    args = parser.parse_args()

    if args.file:
        master_obj.process_text("f", args.file)
    elif args.string:
        master_obj.process_text("s", args.string.encode("ascii", errors="ignore").decode())


def read_master_ior():
    """
    Read the master's IOR from a file (simplified IOR I/O).
    """
    with open(MASTER_IOR_FILE, "r") as mior:
        master_ior = mior.read()
    
    return master_ior


if __name__ == "__main__":
    main()