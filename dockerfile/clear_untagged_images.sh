#!/bin/sh
# This script is meant to be used as a cleanup script
# when developing new images. Run it to get rid of any untagged images.
docker image rm $(docker image ls | grep none | awk '{print $3}')
