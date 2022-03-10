# Dm control ant entity
import dm_control.mjcf
from dm_control.locomotion.walkers import Ant
from pathlib import Path

pth = Path(__file__).parent.parent.resolve()

if __name__ == "__main__":
    test = dm_control.mjcf.from_path(pth / 'assets' / 'ant_tag.xml')
    print(3)