#
# worker.py
#
# Provide functionality to process a shard of tokens.
#
# Homework 1
# Course: CS 441, Fall 2024, UIC
# Author: Himanshu Dongre
#
import sys
import tiktoken
import tensorflow as tf
import random

from omniORB import CORBA, PortableServer
import ShardWorkerModule, ShardWorkerModule__POA


class TokenProcessor(ShardWorkerModule__POA.ShardWorker):
    def byte_pair_encode(shard):
        bp_enc = tiktoken.encoding_for_model("gpt-4o")
        return [ bp_enc.encode(token) for token in shard ]

    def sample_sliding_window_data(token_ids, window_size, stride):
        dataset = tf.data.Dataset.from_tensor_slices(token_ids)
        windows = dataset.window(window_size, shift=stride, drop_remainder=False)
        windows = windows.flat_map(lambda window: window.batch(window_size))
        return list(windows.as_numpy_iterator())
    
    def embed(token_ids):
        # Define the BPE tokenizer to use its properties
        tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Pad sequences to the same length
        maxlen = max([ len(token_id) for token_id in token_ids ])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(token_ids, maxlen=maxlen, padding="post")

        # Create an embedding layer
        embedding_dim = 8 # dimension of the embedding vector

        # Define the model
        model - tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=tokenizer.n_vocab, output_dim=embedding_dim, input_length=maxlen),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid") # Example output layer
        ])

        # Compile the model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Print the model summary
        model.summary()

        # Fit the model (random labels)
        random_labels = [ random.randint(0,1) for _ in token_ids ]
        model.fit(padded_seq, random_labels, epochs=10)

        # Extract and return the embeddings
        embeddings = model.layers[0].get_weights()[0]
        return embeddings


def main():
    public_ip = sys.argv[1]
    master_ior = sys.argv[2]

    orb = CORBA.ORB_init(["-ORBendPointPublish", "giop:tcp:" + public_ip + ":0"], CORBA.ORB_ID)
    poa = orb.resolve_initial_references("RootPOA")

    worker_ior = orb.object_to_string(ShardWorker()._this())

    master_obj = orb.string_to_object(master_ior)._narrow(TokenProcessorModule.TokenProcessor)
    if master_obj is None:
        print("Object reference is not of type Master")
        sys.exit(1)

    # Send IOR to master (genius)
    try:
        master_obj.get_worker_ior(worker_ior)
        orb.run()
    except KeyboardInterrupt:
        print("Exited gracefully 0:-)")


if __name__ == "__main__":
    main()