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
import boto3 # AWS SDK
import configparser # INI config I/O
import csv # CSV I/O
import nltk # Text tokenizer
import paramiko # SSH
import string
import sys

from functools import reduce
from multiprocessing import Process, Queue

from omniORB import CORBA, PortableServer
import TokenProcessor, TokenProcessor__POA


# Read-only
class Config:
    CONFIG_FILE = "config.ini"

    WINDOW_SIZE = 0
    STRIDE = 0
    EMBEDDING_DIM = 0
    S3_IOR_BUCKET = ""

    SHARD_SIZE = 0
    OUT_FILE = ""

    WORKER_AMI = ""
    INSTANCE_TYPE = ""
    MIN_COUNT = 0
    MAX_COUNT = 0
    KEYPAIR = ""
    S3_ROLE = ""
    SECURITY_GROUP = ""

    SSH_KEY_PATH = ""
    SSH_USER = ""
    SSH_PORT = 0

    @classmethod
    def configure(cls):
        """
        Set global parameters from an INI file such that they
        can be read by all processes (parent and children)
        """
        # Read INI config file
        config = configparser.ConfigParser()
        config.read(cls.CONFIG_FILE)

        # Initialize global constants
        cls.WINDOW_SIZE = config["global"].getint("window_size") # Size of the sliding window
        cls.STRIDE = config["global"].getint("stride") # Stride by which sliding window shifts
        cls.EMBEDDING_DIM = config["global"].getint("embedding_dim") # Dimension of embedding vector
        cls.S3_IOR_BUCKET = config["global"].get("s3_ior_bucket")

        cls.SHARD_SIZE = config["master"].getint("shard_size") # Size of each shard in terms of num tokens
        cls.OUT_FILE = config["master"].get("out_file") # Output filename

        cls.WORKER_AMI = config["instance"].get("aws_worker_ami")
        cls.INSTANCE_TYPE = config["instance"].get("instance_type")
        cls.MIN_COUNT = config["instance"].getint("min_count")
        cls.MAX_COUNT = config["instance"].getint("max_count")
        cls.KEYPAIR = config["instance"].get("aws_keypair")
        cls.S3_ROLE = config["instance"].get("aws_s3_permission_role")
        cls.SECURITY_GROUP = config["instance"].get("aws_security_group")

        cls.SSH_KEY_PATH = config["instance"].get("ssh_pem_key_path")
        cls.SSH_USER = config["instance"].get("ssh_username")
        cls.SSH_PORT = config["instance"].getint("ssh_port")


worker_registry = {}


class Master_i(TokenProcessor__POA.Master):
    def process_text(self, text):
        shards = shard_tokens(tokenize_text(text)) # shard_ID -> shard
        print(f"{shards}, {len(shards)}")

        shard_data = {}

        for shard_id, shard in shards.items():
            idmap, freqmap, embeddings_map = process_shard(shard_id, shard)
            shard_data[shard_id] = {"IDs": idmap, "Frequencies": freqmap, "Embeddings": embeddings_map}
            
        write_csv(*clean_data(shard_data))
        
        return text


def main():
    Config.configure()

    global orb
    orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)

    poa = orb.resolve_initial_references("RootPOA")
    poa._get_the_POAManager().activate()

    # Print the master's IOR
    print(orb.object_to_string(Master_i()._this()))

    try:
        orb.run()
    except KeyboardInterrupt:
        print("Exited gracefully O:)")


def tokenize_text(text, language="english"):
    """
    Split text into tokens.
    """
    tokenizer = nltk.TreebankWordTokenizer()
    token_lists = list(map(
        lambda sentence: tokenizer.tokenize(sentence),
        nltk.sent_tokenize(text, language=language)
    ))
    return reduce(lambda acc, curr: acc + curr, token_lists, [])


def shard_tokens(tokens):
    """
    Given a list of tokens, split it into a shards, each <= SHARD_SIZE.
    Return a dictionary of lists of tokens (each element list is a shard),
    and each key is a unique shard ID.
    """
    return dict(map(
        lambda i: (i, tokens[i : i + Config.SHARD_SIZE]),
        range(0, len(tokens), Config.SHARD_SIZE)
    ))


def process_shard(shard_id, shard):
    """
    Calculate (BPE) token IDs, sliding window data samples, and embeddings.
    Return three dictionaries: token->token_id, token_id->frequency, token_id->embedding.
    """
    # Spawn a worker to process the shard
    worker_obj = spawn_worker(worker_id=shard_id)

    token_ids = worker_obj.byte_pair_encode(shard) # list of BPEs
    idmap = dict(zip(shard, token_ids)) # token -> token_ID
    freqmap = {} # token_ID -> frequency
    for token_id in token_ids:
        freqmap[token_id] = freqmap[token_id] + 1 if token_id in freqmap else 1
    #print(token_ids)

    sliding_window_data_samples = worker_obj.sample_sliding_window_data(token_ids) # list of lists of BPEs
    #print(sliding_window_data_samples)

    embeddings = worker_obj.embed(token_ids, sliding_window_data_samples) # list of list of floats
    embeddings_map = dict(zip(freqmap.keys(), embeddings)) # token_ID -> embedding
    #print(embeddings)

    # Kill the worker and return data
    kill_worker(worker_id=shard_id)
    return idmap, freqmap, embeddings_map


def spawn_worker(worker_id):
    """
    Dynamically launch a remote EC2 instance and run a worker on it.
    Return a reference to the worker object.
    """
    # Launch an EC2 instance
    instance_id = launch_instance(worker_id)
    # instance_id = "i-0159bae03b8a0b3f4"

    # Run the worker on the launched instance and pass it worker_id.
    # Run worker in a separate process since it blocks for output.
    # No need to call join on the process; it terminates when instance is terminated.
    Process(target=run_worker, args=(worker_id, instance_id)).start()

    # Read the worker IOR from S3 (genius communication)
    worker_ior = None
    while worker_ior is None:
        worker_ior = read_worker_ior(worker_id)
    
    register_worker(worker_id, instance_id, worker_ior)

    # Get reference to remote worker object and test communication
    worker_obj = orb.string_to_object(worker_ior)._narrow(TokenProcessor.Worker)
    if worker_obj is None:
        print("Object reference is not of type worker")
        sys.exit(1)
    else:
        # Test master->worker communication
        message = f"Hello, Worker #{worker_id}!"
        print(f"\nSent: {message}")
        print(f"Received: {worker_obj.echo_test(message)}\n")
    
    return worker_obj


def launch_instance(worker_id):
    """
    Launch an EC2 instance and return the instance ID.
    """
    ec2 = boto3.resource("ec2")

    instance = ec2.create_instances(
        ImageId = Config.WORKER_AMI,
        MinCount = Config.MIN_COUNT,
        MaxCount = Config.MAX_COUNT,
        InstanceType = Config.INSTANCE_TYPE,
        KeyName = Config.KEYPAIR,
        SecurityGroupIds = [Config.SECURITY_GROUP],
        IamInstanceProfile = {
            "Name": Config.S3_ROLE
        },

        # Startup script that configures environment to be able to run the worker later
        UserData = f'''#!/bin/bash
        cd /home/ec2-user
        export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" # Python should be able to find the CORBA modules

        # Get public IP of the instance *from within the instance* (passed to worker later)
        export TOKEN=$(curl -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" http://169.254.169.254/latest/api/token)
        export IPV4=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)''',

    )[0]

    # Wait for the instance to be in running state and reload configurations
    instance.wait_until_running()
    instance.reload()

    print(f"Launched instance: {instance.id}")
    return instance.id


# Called as a target of a child
def run_worker(worker_id, instance_id):
    """
    SSH into the instance specified by the given instance_id and run the worker.
    Reading instance_id from the worker registry does not work because worker
    is not registered yet.
    """
    Config.configure() # re-configure in child

    ec2 = boto3.resource("ec2")
    instance = ec2.Instance(instance_id)
    hostname = f"ec2-{instance.public_ip_address.replace(".", "-")}.compute-1.amazonaws.com"

    # Create SSH client
    ssh = paramiko.SSHClient()

    # Automatically add host key (can be stricter with known_hosts files)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote host
    ssh.connect(hostname, port=Config.SSH_PORT, username=Config.SSH_USER, key_filename=Config.SSH_KEY_PATH)

    # Run the command to start the worker
    command = f"python3 worker.py {worker_id} $IPV4"
    stdin, stdout, stderr = ssh.exec_command(command) # Blocking

    capture = stdout.read().decode('utf-8')

    # Close the connection
    ssh.close()


def read_worker_ior(worker_id):
    """
    Read the worker IOR from S3 and return it. If worker_id is N
    then the IOR for this worker is in ior00N.txt/ior0N.txt/iorN.txt
    """
    # Create an S3 client
    s3 = boto3.client("s3")
    s3_key = f"ior{worker_id:03}.txt" # File path in S3

    try:
        worker_ior = s3.get_object(Bucket=Config.S3_IOR_BUCKET, Key=s3_key)['Body'].read().decode('utf-8')

        # Delete file after reading from it to ensure correct value the next time
        s3.delete_object(Bucket=Config.S3_IOR_BUCKET, Key=s3_key) 

        return worker_ior
    except Exception as e: # Includes case of file not found
        return None


def register_worker(worker_id, instance_id, worker_ior):
    """
    Add given worker data to the global worker registry.
    """
    worker_registry[worker_id] = {"iID": instance_id, "IOR": worker_ior}
    print(f"Registered worker #{worker_id:03} on {instance_id}.")


def kill_worker(worker_id):
    """
    Terminate the instance on which the worker specified by given worker_ID is running.
    """
    boto3.client("ec2").terminate_instances(InstanceIds=[worker_registry[worker_id]["iID"]])
    print(f"Terminated instance: {worker_registry.pop(worker_id)["iID"]}")


def clean_data(shard_data):
    """
    Gather data from all shards into cumulative data for all tokens.
    """
    idmap = reduce(lambda acc, curr: {**acc, **curr["IDs"]}, shard_data.values(), {})

    freqmap = {}
    embeddings_multimap = {}
    
    for data_elem in shard_data.values():
        for token_id, frequency in data_elem["Frequencies"].items():
            if token_id not in freqmap:
                freqmap[token_id] = 0
            freqmap[token_id] += frequency
        
        for token_id, embeddings in data_elem["Embeddings"].items():
            if token_id not in embeddings_multimap:
                embeddings_multimap[token_id] = []
            embeddings_multimap[token_id].append(embeddings)
    
    embeddings_map = dict(map(
        lambda item: (
            item[0],
            list(map(
                lambda e: sum(e) / len(e),
                zip(*item[1]))
            )
        ),
        embeddings_multimap.items()
    ))

    return idmap, freqmap, embeddings_map


def write_csv(idmap, freqmap, embeddings_map):
    """
    Compile all stats of the data into one CSV file called "CS441hw1Stats.csv".
    """
    with open(Config.OUT_FILE, "w") as stats:
        csv_writer = csv.writer(stats)

        headers = ["Token", "ID", "Frequency"] + [ f"embVal{i:03}" for i in range(Config.EMBEDDING_DIM) ]
        csv_writer.writerow(headers)

        for token in idmap:
            row = [token, idmap[token], freqmap[idmap[token]]] + [ val for val in embeddings_map[idmap[token]] ]
            csv_writer.writerow(row)


if __name__ == "__main__":
    main()