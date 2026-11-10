"""Compatibility entry point; prefer ./run.sh radial-continue --help."""
from llsa.secondary.radial_continuation import (
    fixed_domain_seed, load_seed, main, secant_seed, transport_seed, transport_fixed_buffer, write_json,
)


if __name__ == '__main__':
    main()
