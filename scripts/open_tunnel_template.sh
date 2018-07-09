#!/bin/sh
# This script is meant to be executed inside of a Docker container
# to connect to your Mongo database.
ssh -nNT -L my.host:my_port:mongo.host:mongo_port \
    user@mongo.host >> /path/to/log_file.log 2>&1
