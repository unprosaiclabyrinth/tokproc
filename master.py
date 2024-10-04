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
import logging # logging module
import nltk # Text tokenizer
import paramiko # SSH
import string
import sys

from functools import reduce
from multiprocessing import Pool, Process, Value, Manager, cpu_count
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from omniORB import CORBA, PortableServer
import TokenProcessor, TokenProcessor__POA


# Read-only
class Config:
    CONFIG_FILE = "config.ini"

    WINDOW_SIZE = 0
    STRIDE = 0
    EMBEDDING_DIM = 0
    S3_BUCKET = ""

    SHARD_SIZE = 0
    MASTER_IOR_FILE = ""
    OUT_FILE = ""
    WORKER_LIMIT = 0

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
        cls.S3_BUCKET = config["global"].get("aws_s3_bucket")

        cls.SHARD_SIZE = config["master"].getint("shard_size") # Size of each shard in terms of num tokens
        cls.MASTER_IOR_FILE = config["master"].get("master_ior_file")
        cls.OUT_FILE = config["master"].get("out_file") # Output filename
        cls.WORKER_LIMIT = config["master"].getint("worker_limit")

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
    
    @classmethod
    def upload_config(cls):
        """
        Upload configuration file to S3 for workers to download and read.
        """
        # Create an S3 client
        s3 = boto3.client("s3")

        try:
            s3.upload_file(cls.CONFIG_FILE, cls.S3_BUCKET, cls.CONFIG_FILE)
            print("Uploaded config to S3.")
        except Exception as e: # Includes case of file not found
            print(e)
            sys.exit(1)


worker_registry = {}
worker_count = 0

registry_lock = Lock()
worker_count_lock = Lock()

orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)


class Master_i(TokenProcessor__POA.Master):
    def process_text(self, opt, arg):
        match opt:
            case "f":
                try:
                    with open(arg, "r") as in_file:
                        text = in_file.read().encode("ascii", errors="ignore").decode()
                except Exception as e:
                    print(f"*** Couldn't read file \"{arg}\": {e}")
            case "s":
                text = arg
        
        shards = shard_tokens(tokenize_text(text)) # shard_ID -> shard
        if len(shards) == 1:
            print(f"Distributing {len(shards)} shard across {len(shards)} nodes...")
        else:
            print(f"Distributing {len(shards)} shards across {len(shards)} nodes...")

        shard_data = {}

        # I/O bound => thread pool
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            # Submit tasks to the thread pool
            futures = [executor.submit(process_shard, shard_id, shard) for shard_id, shard in shards.items()]

            # Collect the results as the futures complete
            results = [future.result() for future in as_completed(futures)]

        for shard_id, idmap, freqmap, embeddings_map in results:
            shard_data[shard_id] = {"IDs": idmap, "Frequencies": freqmap, "Embeddings": embeddings_map}
            
        write_csv(*clean_data(shard_data))
        
        return text


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
        lambda i: (i // Config.SHARD_SIZE, tokens[i : i + Config.SHARD_SIZE]),
        range(0, len(tokens), Config.SHARD_SIZE)
    ))


def process_shard(shard_id, shard):
    """
    Calculate (BPE) token IDs, sliding window data samples, and embeddings.
    Return three dictionaries: token->token_id, token_id->frequency, token_id->embedding.
    """
    Config.configure()

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
    return shard_id, idmap, freqmap, embeddings_map


def spawn_worker(worker_id):
    """
    Dynamically launch a remote EC2 instance and run a worker on it.
    Return a reference to the worker object.
    """
    global orb

    # Launch an EC2 instance
    instance_id = launch_instance(worker_id)

    # Run the worker on the launched instance and pass it worker_id.
    # Run worker in a separate thread since it blocks for output.
    # No need to call join on the process; it terminates when instance is terminated.
    Process(target=run_worker, args=(worker_id, instance_id)).start()
    #Thread(target=run_worker, args=(worker_id, instance_id)).start()

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
    
    return worker_obj


def launch_instance(worker_id):
    """
    Launch an EC2 instance and return the instance ID.
    """
    global worker_count_lock, worker_count

    ec2 = boto3.resource("ec2")

    while worker_count >= Config.WORKER_LIMIT:
        pass # Busy wait

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

    with worker_count_lock:
        worker_count += 1

    print(f"Launched instance: {instance.id}")
    return instance.id


def run_worker(worker_id, instance_id):
    """
    SSH into the instance specified by the given instance_id and run the worker.
    Reading instance_id from the worker registry does not work because worker
    is not registered yet.
    """
    Config.configure()

    ec2 = boto3.resource("ec2")
    instance = ec2.Instance(instance_id)
    hostname = f"ec2-{instance.public_ip_address.replace(".", "-")}.compute-1.amazonaws.com"

    # Create SSH client
    ssh = paramiko.SSHClient()

    # Automatically add host key (can be stricter with known_hosts files)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote host
    connected = False
    while not connected:
        try:
            ssh.connect(hostname, port=Config.SSH_PORT, username=Config.SSH_USER, key_filename=Config.SSH_KEY_PATH)
            connected = True

            # Run the command to start the worker
            command = f"python3 worker.py {worker_id} $IPV4"
            stdin, stdout, stderr = ssh.exec_command(command)

            capture = stdout.read().decode('utf-8') # Blocking
        except paramiko.ssh_exception.NoValidConnectionsError:
            # Retry connection in case of connection error
            pass

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
        worker_ior = s3.get_object(Bucket=Config.S3_BUCKET, Key=s3_key)['Body'].read().decode('utf-8')

        # Delete file after reading from it to ensure correct value the next time
        s3.delete_object(Bucket=Config.S3_BUCKET, Key=s3_key) 

        return worker_ior
    except Exception as e: # Includes case of file not found
        return None


def register_worker(worker_id, instance_id, worker_ior):
    """
    Add given worker data to the global worker registry.
    """
    global registry_lock, worker_registry

    with registry_lock:
        worker_registry[worker_id] = {"iID": instance_id, "IOR": worker_ior}
    print(f"Registered worker #{worker_id:03} on {instance_id}.")


def kill_worker(worker_id):
    """
    Terminate the instance on which the worker specified by given worker_ID is running.
    """
    global worker_count_lock, worker_count

    boto3.client("ec2").terminate_instances(InstanceIds=[worker_registry[worker_id]["iID"]])
    print(f"Terminated instance: {worker_registry.pop(worker_id)["iID"]}")

    with worker_count_lock:
        worker_count -= 1


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
    
    print(f'Wrote stats to "{Config.OUT_FILE}". Thanks!')


def write_master_ior(master_ior):
    """
    Write master's IOR to a local file to cimplify IOR I/O for client.
    """
    with open(Config.MASTER_IOR_FILE, "w") as loc_ior:
        loc_ior.write(master_ior)
    
    print(f'Wrote the IOR to "{Config.MASTER_IOR_FILE}".')


def main():
    Config.configure()
    Config.upload_config()
    global orb

    poa = orb.resolve_initial_references("RootPOA")
    poa._get_the_POAManager().activate()

    # Write the master's IOR to a file for the client to read
    write_master_ior(orb.object_to_string(Master_i()._this()))

    try:
        orb.run()
    except KeyboardInterrupt:
        print("Exited gracefully O:)")


if __name__ == "__main__":
    main()