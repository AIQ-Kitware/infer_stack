#!/usr/bin/env bash
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics infer_stack
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics ./tests