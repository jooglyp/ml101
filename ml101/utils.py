"""Utilities"""
import os


def print_delimiter(char='='):
    rows, columns = os.popen('stty size', 'r').read().split()
    print(char * int(columns))