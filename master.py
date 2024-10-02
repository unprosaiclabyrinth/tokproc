#
# master.py
#
# Local master object in the master-worker model.
# Preprocess raw text data and delegate processing to workers.
#
# Homework 1
# Course: CS 441, Fall 2024, UIC
# Author: Himanshu Dongre
#
import sys
import boto3
import nltk
import csv
import string
import configparser

import subprocess
from multiprocessing import Process, Queue

from omniORB import CORBA, PortableServer
import TokenProcessor, TokenProcessor__POA


class Master_i(TokenProcessor__POA.Master):
    def process_text(self, text):
        # with open(filename, "r") as text_file:
        #     text = text_file.read()

        tokens = tokenize_text(text)
        shard = tokens
        print(shard)
        # for shard in shards:
        worker_ior = WORKERS[0]
        print(f"\nRegistered worker {worker_ior}")
        worker_obj = orb.string_to_object(worker_ior)._narrow(TokenProcessor.Worker)
        if worker_obj is None:
            print("Object reference is not of type worker")
            sys.exit(1)
        else:
            print("Connected to worker")
            sent = "Hello, Worker #1!"
            print(f"Sent: {sent}")
            recv = worker_obj.echo_test(sent)
            print(f"Received: {recv}")
        
        token_ids = worker_obj.byte_pair_encode(shard)

        idmap = dict(zip(tokens, token_ids))
        freqmap = {}
        for token_id in token_ids:
            if token_id in freqmap:
                freqmap[token_id] += 1
            else:
                freqmap[token_id] = 1

        print(f"{token_ids}")

        sliding_window_data_samples = worker_obj.sample_sliding_window_data(token_ids)
        print(sliding_window_data_samples)

        embeddings = worker_obj.embed(token_ids, sliding_window_data_samples)
        print(embeddings)
        embeddings_dict = dict(zip(freqmap.keys(), embeddings))

        with open("stats.csv", "w") as out_file:
            csv_writer = csv.writer(out_file)

            headers = ["Token", "ID", "Frequency"] + [ f"embVal{i:03}" for i in range(EMBEDDING_DIM) ]
            csv_writer.writerow(headers)

            for token in idmap:
                row = [token, idmap[token], freqmap[idmap[token]]] + [ val for val in embeddings_dict[idmap[token]] ]
                csv_writer.writerow(row)
        
        return text

    def get_worker_ior(self, worker_ior):

        workers.append(worker_ior) # register worker
        return worker_ior


def main():
    configure()
    global orb, master_ior
    
    orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)
    poa = orb.resolve_initial_references("RootPOA")
    poa._get_the_POAManager().activate()

    master_ior = orb.object_to_string(Master_i()._this())
    print(f"{master_ior}")

    try:
        orb.run()
    except KeyboardInterrupt:
        print("Exited gracefully O:-)")


def split_tokens(tokens):
    shard_size = len(tokens) // n_nodes
    shards = [ tokens[i : i + shard_size] for i in range(0, len(tokens), shard_size) ]
    
    if len(tokens) % n_nodes != 0:
        shards[-1].extend(tokens[len(shards) * shard_size :])

    return shards


def tokenize_text(text, language="english"):
    tokens = []
    sentences = nltk.sent_tokenize(text, language=language)
    tokenizer = nltk.TreebankWordTokenizer()
    for sentence in sentences:
        tokens += tokenizer.tokenize(sentence)
    return tokens


def configure():
    """
    Set global parameters from an INI file
    """
    # Read INI config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Initialize global constants
    global WINDOW_SIZE, STRIDE, EMBEDDING_DIM, SHARD_SIZE, WORKER_AMI, WORKERS
    WINDOW_SIZE = config["global"].getint("window_size") # Size of the sliding window
    STRIDE = config["global"].getint("stride") # Stride by which sliding window shifts
    EMBEDDING_DIM = config["global"].getint("embedding_dim") # Dimension of embedding vector
    SHARD_SIZE = config["master"].getint("shard_size") # Size of each shard in terms of num tokens
    WORKER_AMI = config["master"].get("worker_ami")
    WORKERS[0] = config["master"].get("worker0")


if __name__ == "__main__":
    main()