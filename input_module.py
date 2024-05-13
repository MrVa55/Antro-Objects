# input_module.py
import argparse
import logging
import os
import sys
from whisper_online import add_shared_args, asr_factory, set_logging, load_audio_chunk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", help="The path to a speech audio wav file to warm up Whisper.")
    add_shared_args(parser)
    return parser.parse_args()

def setup_logger():
    logger = logging.getLogger(__name__)
    return logger

def warmup_asr(args, logger):
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. " + msg)
            sys.exit(1)
    else:
        logger.warning(msg)

def init_asr(args):
    SAMPLING_RATE = 16000
    size = args.model
    language = args.lan
    asr, online = asr_factory(args)
    min_chunk = args.min_chunk_size
    return SAMPLING_RATE, asr, online, min_chunk
