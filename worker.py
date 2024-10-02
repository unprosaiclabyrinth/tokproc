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

from tensorflow import convert_to_tensor, float32
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

from omniORB import CORBA, PortableServer
import TokenProcessor, TokenProcessor__POA


class Worker_i(TokenProcessor__POA.Worker):
    def byte_pair_encode(self, shard):
        print("Received 'em words")
        bp_enc = tiktoken.encoding_for_model("gpt-4o")
        token_ids = []
        for token in shard:
            token_ids += bp_enc.encode(token)
        print(f"Token IDs: {token_ids}")
        return token_ids

    def sample_sliding_window_data(self, token_ids, window_size, stride):
        dataset = Dataset.from_tensor_slices(token_ids)
        windows = dataset.window(window_size, shift=stride, drop_remainder=False)
        windows = windows.flat_map(lambda window: window.batch(window_size))
        samples = [ sample.tolist() for sample in list(windows.as_numpy_iterator()) ]
        print(f"Sliding window samples: {samples}")
        return samples
    
    def embed(self, token_ids, sliding_window_samples):
        # Define the BPE tokenizer to use its properties
        tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Pad sequences to the same length
        maxlen = max([ len(sample) for sample in sliding_window_samples ])
        padded_seq = pad_sequences(sliding_window_samples, maxlen=maxlen, padding="post")

        # Create an embedding layer
        embedding_dim = 4 # dimension of the embedding vector

        # Define the model
        model = Sequential([
            Embedding(input_dim=tokenizer.n_vocab, output_dim=embedding_dim),
            GlobalAveragePooling1D(),
            Dense(1, activation="sigmoid") # Example output layer
        ])

        # Compile the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Print the model summary
        model.summary()
        print(padded_seq.shape)

        # Fit the model (random labels)
        random_labels = [ random.randint(0,1) for _ in sliding_window_samples ]
        labels_tensor = convert_to_tensor(random_labels, dtype=float32)
        padded_tensor = convert_to_tensor(padded_seq, dtype=float32)
        model.fit(padded_tensor, labels_tensor, epochs=10)

        # Extract and return the embeddings
        all_embeddings = [ [ float(value) for value in embedding ] for embedding in model.layers[0].get_weights()[0] ]
        token_embeddings = []
        uniq_token_ids = []
        for token_id in token_ids:
            if token_id not in uniq_token_ids:
                uniq_token_ids.append(token_id)
                token_embeddings.append(all_embeddings[token_id])
        return token_embeddings


def main():
    # public_ip = sys.argv[1]
    master_ior = sys.argv[1]

    orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)
    poa = orb.resolve_initial_references("RootPOA")

    worker_ior = orb.object_to_string(Worker_i()._this())

    master_obj = orb.string_to_object(master_ior)._narrow(TokenProcessor.Master)
    if master_obj is None:
        print("Object reference is not of type Master")
        sys.exit(1)

    # Send IOR to master (genius)
    try:
        master_obj.get_worker_ior(worker_ior)

        poa._get_the_POAManager().activate()
        orb.run()
    except KeyboardInterrupt:
        print("Exited gracefully 0:-)")


if __name__ == "__main__":
    main()