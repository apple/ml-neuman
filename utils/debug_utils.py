# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/utils/debug_utils.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


def embed_breakpoint(debug_info='', terminate=True):
    print('\nyou are inside a break point')
    if debug_info:
        print(f'debug info: {debug_info}')
    print('')
    embedding = ('import numpy as np\n'
                 'import IPython\n'
                 'import matplotlib.pyplot as plt\n'
                 'IPython.embed()\n'
                 )
    if terminate:
        embedding += (
            'exit()'
        )

    return embedding
