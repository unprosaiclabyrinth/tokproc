#
# worker.py
#
# Source for remote worker object in the master-worker model.
# Provide functionality to process a shard of tokens.
#
# Homework 1
# Course: CS 441, Fall 2024, UIC
# Author: Himanshu Dongre
#
import sys
import tiktoken
import random
import configparser
import boto3

from tensorflow import convert_to_tensor, float32
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

from omniORB import CORBA, PortableServer
import TokenProcessor, TokenProcessor__POA

CONFIG_FILE = "config.ini"

class Worker_i(TokenProcessor__POA.Worker):
    def echo_test(self, message):
        """
        Function to test client-server communication
        """
        print(f'Got "{message}", sending it back...')
        return message
    
    def byte_pair_encode(self, shard):
        """
        Return list of token ID's for all tokens in the shard using byte pair encoding
        """
        bp_enc = tiktoken.encoding_for_model("gpt-4o")

        token_ids = []
        for token in shard:
            token_ids += bp_enc.encode(token)

        return token_ids

    def sample_sliding_window_data(self, token_ids):
        """
        Return sliding window data samples given window size and stride.
        """
        windows = (
            Dataset.from_tensor_slices(token_ids)
            .window(WINDOW_SIZE, shift=STRIDE, drop_remainder=False)
            .flat_map(lambda window: window.batch(WINDOW_SIZE))
        )

        # Convert any np arrays to lists
        samples = list(map(lambda sample: sample.tolist(), list(windows.as_numpy_iterator())))

        #print(f"Sliding window samples: {samples}")
        return samples
    
    def embed(self, token_ids, sliding_window_samples):
        """
        Return embeddings only for tokens in the shard.
        """
        # Define the BPE tokenizer to use its properties
        tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Pad sequences to the same length
        padded_seq = pad_sequences(
            sliding_window_samples,
            maxlen=max(list(map(len, sliding_window_samples))),
            padding="post"
        )

        # Define the model with an embedding layer
        model = Sequential([
            Embedding(input_dim=tokenizer.n_vocab, output_dim=EMBEDDING_DIM),
            GlobalAveragePooling1D(),
            Dense(1, activation="sigmoid") # Output layer
        ])

        # Compile the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Print the model summary
        model.summary()

        # Fit the model (random labels)
        random_labels_tensor = convert_to_tensor(
            list(map(lambda sample: random.randint(0,1), sliding_window_samples)),
            dtype=float32
        )
        padded_tensor = convert_to_tensor(padded_seq, dtype=float32)
        model.fit(padded_tensor, random_labels_tensor, epochs=10)

        # Extract and return the embeddings
        all_embeddings = list(
            map(
                lambda embedding: list(map(float, embedding)),
                model.layers[0].get_weights()[0]
            )
        )
        token_embeddings = list(
            map(lambda token_id: all_embeddings[token_id], list(dict.fromkeys(token_ids)))
        )
        return token_embeddings


def main():
    configure()

    worker_id = int(sys.argv[1])
    public_ip = sys.argv[2]

    orb = CORBA.ORB_init(["-ORBendPointPublish", f"giop:tcp:{public_ip}:0"], CORBA.ORB_ID)
    poa = orb.resolve_initial_references("RootPOA")
    poa._get_the_POAManager().activate()

    worker_ior = orb.object_to_string(Worker_i()._this())
    
    # Write worker IOR to persistent S3 storage
    # to communicate it to master (master reads)
    write_worker_ior(worker_id, worker_ior)

    try:
        orb.run()
    except KeyboardInterrupt:
        print("Exiting gracefully... O:)")


def write_worker_ior(id, ior):
    """
    Write the worker IOR to S3 for the master to read it. If worker_id is N
    then the IOR for this worker is in ior00N.txt/ior0N.txt/iorN.txt
    """
    # Write IOR locally
    with open(LOCAL_IOR_FILE, "w") as loc_ior:
        loc_ior.write(ior)
    
    # Create an S3 client
    s3 = boto3.client("s3")

    # Set the bucket name and file details
    local_key = LOCAL_IOR_FILE # Path to local file
    s3_key = f"ior{id:03}.txt" # Path to S3 file

    try:
        s3.upload_file(local_key, S3_IOR_BUCKET, s3_key)
        print(f"File uploaded successfully to {S3_IOR_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")


def configure():
    """
    Set global parameters from an INI file
    """
    # Read INI config file
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    # Initialize global constants
    global WINDOW_SIZE, STRIDE, EMBEDDING_DIM, S3_IOR_BUCKET
    WINDOW_SIZE = config["global"].getint("window_size") # Size of the sliding window
    STRIDE = config["global"].getint("stride") # Stride by which sliding window shifts
    EMBEDDING_DIM = config["global"].getint("embedding_dim") # Dimension of embedding vector
    S3_IOR_BUCKET = config["global"].get("s3_ior_bucket")

    global LOCAL_IOR_FILE
    LOCAL_IOR_FILE = config["worker"].get("local_ior_file")


if __name__ == "__main__":
    main()