# generate_data.py

import os
import json

from config import DATA_DIR, N_SPINS, N_REALIZATIONS, LATTICE_SIZE, ALPHA
from utils import generate_positions, initial_couplings
from sdrg import strongest_bond, decimate
from json_writer import build_step_json



def generate_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    for r in range(N_REALIZATIONS):
        print(f"Generating realization {r}")

        positions = generate_positions(N_SPINS, LATTICE_SIZE)
        J = initial_couplings(positions, ALPHA)
        active_spins = list(range(N_SPINS))

        rdir = os.path.join(DATA_DIR, f"realization_{r:03d}")
        os.makedirs(rdir, exist_ok=True)

        step = 0
        while len(active_spins) > 1:
            i, j = strongest_bond(J, active_spins)

            sample = build_step_json(J, positions, active_spins, (i, j))

            with open(os.path.join(rdir, f"step_{step:03d}.json"), "w") as f:
                json.dump(sample, f, indent=2)

            J, active_spins = decimate(J, active_spins, i, j)
            step += 1


if __name__ == "__main__":
    generate_dataset()

