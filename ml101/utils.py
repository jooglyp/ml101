"""Utilities"""
import os


def print_delimiter(char='='):
    # TODO: Deprecate as this will only work on linux
    rows, columns = os.popen('stty size', 'r').read().split()
    print(char * int(columns))
