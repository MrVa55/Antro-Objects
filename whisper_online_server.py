#!/usr/bin/env python3
from whisper_module import *
import sys
import argparse
import os
import logging
import numpy as np


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")


# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args,logger,other="")


# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.host, args.port))
    s.listen(1)
    logger.info('Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logger.info('Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(connection, online, min_chunk)
        proc.process()
        conn.close()
        logger.info('Connection to client closed')
logger.info('Connection closed, terminating.')
