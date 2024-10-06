**Name:** Himanshu Dongre\
**Option:** Option 2 ([CORBA](https://omniorb.sourceforge.io/) + Python)\
**Video link:**


**This project is NOT replicable in this state because it uses a custom AMI (Amazon Machine Image) to launch EC2 instances. It could be replicated with explicit access to this AMI.**

# Overview

This Git repo contains the Python implementation using CORBA of HW1 for CS441, Fall 2024, UIC. The major pieces are:-

1. **client.py:** The client script. This sends a file or a string to the server to process.
2. **master.py:** I have implemented the server as a [master-worker](http://charm.cs.uiuc.edu/research/masterSlave) model. The master receives the request from the client, splits the input into shards, and delegates each shard to an individual worker.
3. **worker.py** This is the script that runs in the cloud on an AWS EC2 instance. Each worker processes one shard on one node. Workers are dynamically spawned by the master based on the number of shards.
4. **tokproc.idl:** This contains the interface definitions for the CORBA distributed objects in [IDL](https://en.wikipedia.org/wiki/Interface_description_language) (Interface Definition Language). It contains the definitions for both the master and the worker.
5. **config.ini:** All configuration parameters have been factored out into a config file. I have used an [INI file](https://en.wikipedia.org/wiki/INI_file) for ease of I/O using Python's API.

The prerequisite Python dependencies and required AWS configurations are listed in the subsequent sections.

# Python dependencies

This project has been tested on Python version 3.9 and higher.

+ **[boto3](https://pypi.org/project/boto3/):** Boto3 is the official Python SDK (Software Development Kit) for Amazon Web Services (AWS). Boto3 provides an intuitive API to access AWS resources and services, making tasks like launching EC2 instances, uploading files to S3, or managing cloud infrastructure straightforward. It is used in this project to dynamically launch instances to run workers on, and terminating them after use.
+ **[paramiko](https://www.paramiko.org/):** Paramiko is a Python library that provides an interface for working with SSH (Secure Shell) and SFTP (SSH File Transfer Protocol). It allows developers to create secure connections to remote machines, execute commands, and transfer files over the SSH protocol. Paramiko is widely used for automating remote server administration tasks, establishing encrypted tunnels, and securely communicating between systems. It is used in this project to access remote EC2 instances and start the worker scripts.
+ **[nltk](https://www.nltk.org/):** The Natural Language Toolkit (NLTK) is a powerful Python library used for natural language processing (NLP). It provides a suite of text-processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, among other tasks. NLTK also includes datasets, lexical resources, and tools to help linguists, developers, and researchers work with human language data. It is used in this project 
+ **[tiktoken]( https://pypi.org/project/tiktoken):** Tiktoken is a fast and efficient tokenization library designed for use with OpenAI’s language models. It focuses on breaking down text into tokens, which are the building blocks used by large language models (LLMs) such as GPT, BERT, etc. Tiktoken optimizes tokenization for specific model families (like GPT-3, GPT-4), ensuring compatibility and performance when feeding inputs into these models.
+ **[tensorflow](https://pypi.org/project/tensorflow/):** TensorFlow is an open-source machine learning framework developed by Google. It is designed to build and deploy machine learning models across a range of platforms, from desktops to mobile devices to distributed systems in the cloud. TensorFlow provides a comprehensive ecosystem for developing machine learning and deep learning models, from research and prototyping to production-level deployment.
+ **[configparser](https://pypi.org/project/configparser/):** This is a Python module used for working with configuration files. It provides a way to read, write, and modify configuration files in a format that is similar to Windows INI files, which contain sections and key-value pairs. This is particularly useful for storing settings for applications in a structured and easy-to-read way. It is used in this project for INI file I/O.
+ **logging:** The logging module in Python provides a flexible framework for emitting log messages from Python programs. It is used in this project for logging the master's execution.

Some other Python modules used in this project are:-

+ **functools:** The `reduce` function from `functools` is used.
+ **multiprocessing:** It is used in this project for implementing multicore parallelism using multiprocessing.
+ **threading:** `Lock` from `threading` is used for synchronization.

# Custom AWS configurations

This project relies heavily on AWS configurations. The following are required to successfully run the project:-

1. **Custom AMI:** This project launches EC2 instances from a custom AMI that has omniORB CORBA and all the necessary Python dependencies configured for the worker script to run, along with the worker.py script pre-installed.
2. **[S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) bucket:** This project uses an AWS S3 bucket for IOR communication between the worker and the master. Such a bucket must be set up before attempting to run the project. The customized bucket can be specified in `config.ini`.
3. **Key pair:** Launching an instance required a key pair. This project has used a private keypair, and a new one must be generated for attempting to run the project. The name of the key pair can be specified in `config.ini`.
4. **[IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html):** This project uses a custom IAM role that grants all EC2 instances permission to access S3. Such a role must be set up before attempting to run the project. The customized IAM role can be specified in `config.ini`.
5. **Security group:** This project uses a custom security group, which allows all incoming traffic on all ports, to launch EC2 instances. Such a security group must be set up before attempting to run the project. The new security group name can be specified in `config.ini`.
6. **SSH key:** The key pair used to launch instances can be downloaded as a PEM file, of which the private key is used to establish a SSH tunnel to the remote instance. A new SSH key path can be specified in `config.ini`.

# Getting Started

After the aforementioned set up *required* to be able to run the project, the project can be run by executing the master script:-

```python
python3 master.py
```

Requests can be sent to the master by running the client script locally on a different terminal. The system designed in the project requires master and client to be mutually local for it to run as expected. The motivation behind such a design is simplicity as greater focus lies on utlizing cloud services. The client can be run as:-

```python
python3 client.py
```

The client accepts command-line arguments speciying input. The `-h` option can be used to elicit a help message from the client, which lists the acceptable command-line options.

```python
python3 client.py -h
```

# Testing

This project has been tested on the dataset: plaintext version of the book *The Adventures of Sherlock Holmes* by **Sir Arthur Conan Doyle** downloaded from Project Gutenberg.

```sh
hw1/ $ wc sherlock.txt
   11922  104506  587719 sherlock.txt
```

The script `master-test.py` is a test verion of the master script. The instance to which it connects is hardcoded (the actual submission is `master.py`). Unit tests have been carried out for all operations by monitoring output on this remote instance. The instance ID is i-0f1b66547267d7f21 and this test instance is seen as running in the supplemental video. A toy dataset with around 100 words was used for testing dynamic launching of instances and running of workers.