'''
    Use this to easily combine complex configurations.

    Add your args file to `args/` then specify the files you want to combine.
'''
import sys
import os
from unittest.mock import patch

from transformer_vae.train import main


args_with_content = {}


def format_args(txt):
    lines = txt.strip().split('\n')
    return [
        f'--{l}' for l in lines
    ]


def shorten(arg):
    name = ''
    words = arg.split('_')
    for wd in words:
        name += wd[0].upper() + wd[1:5]
    return name


for file in os.listdir('args'):
    if file.endswith(".args"):
        path = os.path.join('args', file)
        name = file[:-5]
        assert(name not in args_with_content)
        args_with_content[name] = format_args(open(path, 'r').read())


args = sys.argv
made_args_lines = []
short_names = []

for i, arg in enumerate(['base'] + args[1:]):
    if arg[:2] == '--':
        made_args_lines += args[i:]
        break
    made_args_lines += args_with_content[arg]
    if arg != 'base':
        short_names.append(shorten(arg))

short_names = sorted(short_names)

args_str = 'train.py\n' + '\n'.join(made_args_lines)

if '--run_name=' not in args_str:
    args_str += f'\n--run_name={"_".join(short_names)}'

print(f'''
Running With Arguments:
-----------------------
{args_str}
''')

import pdb; pdb.set_trace()

with patch.object(sys, "argv", args_str.split()):
    main()
