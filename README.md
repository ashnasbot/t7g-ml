# T7G-ml

Some experiements in training a model to play the The 7th Guest minigame,
'Microscope'. This game plays a little like orthello on a 7x7 Zero sum grid.

The goal is to develop an agent that wins against the game AI in as few moves
as possible.

## Environments

- `t7g_env.py` provides an environment for a model to play against Stauf himself.
    very slow, good for confirming performance (or not).
- `t7g_virt_env.py` Provides a virtual environment for faster training.
    This env can also move the opponent randomly for a basic training env.
    Otherwise this env can be used for playing.
- `t7g_virt_env_plus_C_shoxx.py` This mouthful works as `t7g_virt_env` except
    it implements a alhpa-beta pruning minimaxer to provide an opponent.
    The minimaxer was delveoped by [Darkshoxx](https://github.com/darkshoxx) and adapted to C for greater performance.

## Trainers

 TODO

## Utils

- `utils.py` contains any easily extractable code from the other envs, to
    reduce duplication.

## C module
`micro_3.c` contains a C impl of the alpha-beta pruning minimaxer for getting the next best green cell move.

Windows:
```
gcc -Ofast .\micro_3.c -o .\micro3.dll --shared
```
This module uses a numpy array converted to bytes directly and doesnt do much in the way of error checking,
be warned.
