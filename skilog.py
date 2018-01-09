import sys
import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log.addHandler(log_handler)
log_handler.setFormatter(logging.Formatter('[%(levelname)s]%(message)s'))


