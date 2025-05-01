#!/usr/bin/env python3

import sys
from facefusion import core

if __name__ == '__main__':
    # Optional: allow command-line args to influence behavior
    command = sys.argv[1] if len(sys.argv) > 1 else 'run'
    core.run(command)
