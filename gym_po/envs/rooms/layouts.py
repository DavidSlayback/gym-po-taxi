__all__ = ['LAYOUTS', 'layout_to_np', 'np_to_grid', 'ENDS', 'STARTS']
from typing import Tuple
import numpy as np
# Various ROOMs layouts
LAYOUTS = {
    '1': '''xxxxxxxxxxxxx
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            xxxxxxxxxxxxx''',
    '2': '''xxxxxxxxxxxxx
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            x00000000000x
            xxxxxx1xxxxxx
            x11111111111x
            x11111111111x
            x11111111111x
            x11111111111x
            x11111111111x
            xxxxxxxxxxxxx''',
    '4': '''xxxxxxxxxxxxxxxxx
            x1111111x0000000x
            x1111111x0000000x
            x1111111x0000000x
            x1111111x0000000x
            x111111110000000x
            x1111111x0000000x
            x1111111x0000000x
            xx2xxxxxx0000000x
            x2222222xxxx3xxxx
            x2222222x3333333x
            x2222222x3333333x
            x2222222x3333333x
            x222222233333333x
            x2222222x3333333x
            x2222222x3333333x
            xxxxxxxxxxxxxxxxx''',
    '8': '''xxxxxxxxxxxxxxxxxxxxxxxxx
            x55555x11111144444x00000x
            x55555x11111x44444x00000x
            x55555x11111x44444x00000x
            x55555111111x44444x00000x
            x55555x11111x44444400000x
            x5xxxxx11111xxxxx4x00000x
            x22222xxxx3xx77777xx6xxxx
            x22222x33333777777x66666x
            x22222x33333x77777x66666x
            x22222x33333x77777x66666x
            x22222233333x77777666666x
            xxxxxxxxxxxxxxxxxxxxxxxxx''',
    '10': '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
             x88888x11111144444x55555500000x
             x88888x11111x44444x55555x00000x
             x88888x11111x44444x55555x00000x
             x88888111111x44444x55555x00000x
             x88888x11111x44444455555x00000x
             x8xxxxx11111xxxx7xx55555xxxx9xx
             x22222xxx1xxx77777xxxxx5x99999x
             x22222x33333377777x66666x99999x
             x22222x33333x77777x66666x99999x
             x22222x33333x77777x66666x99999x
             x22222233333x77777666666999999x
             xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx''',
    '16': '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
             x88888x11111144444x55555x:::::x;;;;;x??????00000x
             x88888x11111x44444x55555x:::::;;;;;;x?????x00000x
             x88888x11111x44444x55555x:::::x;;;;;x?????x00000x
             x88888111111x44444x555555:::::x;;;;;x?????x00000x
             x88888x11111x44444455555x:::::x;;;;;??????x00000x
             x8xxxxx11111xxxx7xx55555xxxx:xxxxx;xx?????xxxx>xx
             x22222xxx1xxx77777xxxxx5x99999x<<<<<xxxxx=x>>>>>x
             x22222x33333377777x66666x99999x<<<<<x=====>>>>>>x
             x22222x33333x77777x66666x99999x<<<<<======x>>>>>x
             x22222x33333x77777x66666x99999<<<<<<x=====x>>>>>x
             x22222233333x77777666666999999x<<<<<x=====x>>>>>x
             xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx''',
    '32': '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
             x^^^^^x!!!!!!"""""x/////x.....x,,,,,x``````00000x
             x^^^^^x!!!!!x"""""x/////x.....,,,,,,x`````x00000x
             x^^^^^x!!!!!x"""""x/////x.....x,,,,,x`````x00000x
             x^^^^^!!!!!!x"""""x//////.....x,,,,,x`````x00000x
             x^^^^^x!!!!!x""""""/////x.....x,,,,,``````x00000x
             x^xxxxx!!!!!xxxx(xx/////xxxx.xxxxx]xx`````xxxx|xx
             x-----xxx!xxx(((((xxxxx/x[[[[[x]]]]]xxxxx_x|||||x
             x-----x++++++(((((x)))))x[[[[[x]]]]]x_____||||||x
             x-----x+++++x(((((x)))))x[[[[[x]]]]]______x|||||x
             x-----x+++++x(((((x)))))x[[[[[]]]]]]x_____x|||||x
             x------+++++x((((())))))[[[[[[x]]]]]x_____x|||||x
             xxxx-xxxx+xxxxxx(xxxx)xxxx[xxxx]xxxxx_xxxxxxxxx|x
             x88888x11111144444x55555x:::::x;;;;;x??????&&&&&x
             x88888x11111x44444x55555x:::::;;;;;;x?????x&&&&&x
             x88888x11111x44444x55555x:::::x;;;;;x?????x&&&&&x
             x88888111111x44444x555555:::::x;;;;;x?????x&&&&&x
             x88888x11111x44444455555x:::::x;;;;;??????x&&&&&x
             x8xxxxx11111xxxx7xx55555xxxx:xxxxx;xx?????xxxx&xx
             x22222xxx1xxx77777xxxxx5x99999x<<<<<xxxxx=x>>>>>x
             x22222x33333377777x66666x99999x<<<<<x=====>>>>>>x
             x22222x33333x77777x66666x99999x<<<<<<=====x>>>>>x
             x22222x33333x77777x66666x999999<<<<<x=====x>>>>>x
             x22222233333x77777666666999999x<<<<<x=====x>>>>>x
             xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
}
WALL_CHAR = 'x'
ENDS = {
    '1': (11, 11),
    '2': (11, 11),
    '4': (15, 15),
    '8': (23, 11),
    '10': (29, 11),
    '16': (47, 11),
    '32': (47, 32),
}
STARTS = {
    '1': (1, 1),
    '2': (1, 1),
    '4': (1, 1),
    '8': (1, 1),
    '10': (1, 1),
    '16': (1, 1),
    '32': (1, 1),
}

def layout_to_np(layout: str) -> np.ndarray:
    """Convert layout string to numpy char array"""
    return np.asarray([t.strip() for t in layout.splitlines()], dtype='c').astype('U')


def np_to_grid(np_layout: np.ndarray) -> np.ndarray:
    """Convert numpy char array to state-abstracted integer grid"""
    state_aliases = np.unique(np_layout)
    state_aliases_without_wall = np.delete(state_aliases, np.nonzero(state_aliases == WALL_CHAR))
    state_alias_values = np.arange(len(state_aliases_without_wall))
    grid = np.full_like(np_layout, -1, dtype=int)
    for i, a in zip(state_alias_values, state_aliases_without_wall):
        grid[np_layout == a] = i
    return grid

