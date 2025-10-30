# landlab-pyvista-tools-examples
This is a repo for developing tools and examples around using PyVista to visualize Landlab grids and fields.

## Installation

It's recommended to use a virtual environment to avoid conflicts with other Python
packages:

```bash
python -m venv venv
source venv/bin/activate
```

To install **llpvtools**, run:

```bash
pip install -e .
```

If you want to run the examples, you can skip the above and instead run:

```bash
pip install -e .[examples]
```

> **Note:** You only need to run *one* of the above commands. The second
  includes everything from the first, plus additional dependencies for the examples.
