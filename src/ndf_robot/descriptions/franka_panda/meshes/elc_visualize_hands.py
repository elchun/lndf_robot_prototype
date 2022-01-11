from math import comb
import numpy as np
import trimesh

from airobot.utils import common

# hand = trimesh.load('visual/hand.obj')
# lf = trimesh.load('visual/finger.obj')
# rf = trimesh.load('visual/finger.obj')
hand = trimesh.load('collision/hand.obj')
full_hand = trimesh.load('panda_open_hand_full.obj')

combined_hands = trimesh.util.concatenate([hand, full_hand])
# combined_hands = hand
combined_hands.show()

# offset offset_z = 0.0584

# l_offset = [0, 0.015, offset_z]
# r_offset = [0, -0.015, offset_z]

# combined_hand = trimesh.util.concatenate([hand, rf, lf])

