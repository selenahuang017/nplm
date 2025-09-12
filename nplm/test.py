#!/usr/bin/env python

# nplm/test_install.py

"""
A simple smoke test for the NPLM assignment package.

Run with:
    python -m nplm.test_install

If everything is installed correctly, you should see a confirmation message.
"""

#############
## Imports ##
#############

import sys
# import transformers
# import datasets
import torch
import numpy
import tqdm
import nltk


##########
## Main ##
##########

def main():
    print("NPLM package import works. :)")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS backend is available (Apple Silicon GPU).")
    else:
        print("Running on CPU (no GPU detected).")

    # Tiny tensor op as a sanity check
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    print("Tensor test:", (x + y).tolist())

    print("Setup looks good! You're ready to start the assignment.")


#####################
## CLI Entry Point ##
#####################

if __name__ == "__main__":
    main()
