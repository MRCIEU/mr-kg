#!/usr/bin/env python3
"""Sanity check script to test common_funcs import."""

import duckdb
from common_funcs import hello

if __name__ == "__main__":
    print("Testing common_funcs import...")
    hello()
    print("Import test successful!")

    duckdb.sql("SELECT 42").show()
