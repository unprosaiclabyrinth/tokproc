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

from omniORB import CORBA, PortableServer
import TokenProcessor, TokenProcessor__POA

workers = []

class Master_i(TokenProcessor__POA.Master):
    def process_text(self, text):
        # with open(filename, "r") as text_file:
        #     text = text_file.read()

        tokens = tokenize_text(text)
        shard = tokens
        print(shard)
        # for shard in shards:
        worker_obj = orb.string_to_object(workers[0])._narrow(TokenProcessor.Worker)
        if worker_obj is None:
            print("Object reference is not of type worker")
            sys.exit(1)
        else:
            print("Connected to worker")
        
        token_ids = worker_obj.byte_pair_encode(shard)

        idmap = dict(zip(tokens, token_ids))
        freqmap = {}
        for token_id in token_ids:
            if token_id in freqmap:
                freqmap[token_id] += 1
            else:
                freqmap[token_id] = 1

        print(f"{token_ids}")

        sliding_window_data_samples = worker_obj.sample_sliding_window_data(token_ids, 100, 10)
        print(sliding_window_data_samples)

        embeddings = worker_obj.embed(token_ids, sliding_window_data_samples)
        print(embeddings)
        embeddings_dict = dict(zip(freqmap.keys(), embeddings))

        embedding_dim = 4
        with open("stats.csv", "w") as out_file:
            csv_writer = csv.writer(out_file)

            headers = ["Token", "ID", "Frequency"] + [ f"embVal{i}" for i in range(embedding_dim) ]
            csv_writer.writerow(headers)

            for token in idmap:
                row = [token, idmap[token], freqmap[idmap[token]]] + [ val for val in embeddings_dict[idmap[token]] ]
                csv_writer.writerow(row)
        
        return text

    def get_worker_ior(self, worker_ior):
        print(f"\nRegistered worker {worker_ior}")
        workers.append(worker_ior) # register worker
        return worker_ior


def main():
    global orb, master_ior1#, master_ior2
    orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)
    poa = orb.resolve_initial_references("RootPOA")

    master_ior1 = orb.object_to_string(Master_i()._this())
    print(f"For worker:-\n{master_ior1}")

    # master_ior2 = orb.object_to_string(Master()._this())
    # print(f"For client:-\n{master_ior2}")

    poa._get_the_POAManager().activate()

    try:
        orb.run()
    except KeyboardInterrupt:
        print("Exited gracefully O:-)")


def split_tokens(tokens):
    n_nodes = 10
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


if __name__ == "__main__":
    main()