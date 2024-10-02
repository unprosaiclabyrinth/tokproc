#
# client.py
#
# Provide a text file for processing.
#
# Homework 1
# Course: CS 441, Fall 2024, UIC
# Author: Himanshu Dongre
#
import sys

from omniORB import CORBA, PortableServer
import TokenProcessor


def main():
    orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)
    # TODO add error checking
    master_obj = orb.string_to_object(sys.argv[1])._narrow(TokenProcessor.Master)

    # with open(sys.argv[2], "r") as f:
    #     text = f.read()
    
    text = "Hello, World!"
    master_obj.process_text(text)


if __name__ == "__main__":
    main()