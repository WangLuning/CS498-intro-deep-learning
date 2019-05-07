# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from rnn.helpers import *
from rnn.model import *

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, device=None):
    hidden = decoder.init_hidden(1, device=device)
    prime_input = char_tensor(prime_str).unsqueeze(0).to(device)

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).unsqueeze(0).to(device)

    return predicted

