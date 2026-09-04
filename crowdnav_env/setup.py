"""Packaging for the simulator and evaluation harness.

Install in editable mode from the repository root::

    pip install -e crowdnav_env

Exact versions are pinned in requirements/main.txt; the bounds here only stop
this file from silently installing a stack the code cannot run on. Without
them, ``pip install -e crowdnav_env`` in an environment where the requirements
install had failed would happily pull numpy 2 and gym 0.26 and report success,
and the breakage would surface much later as an unrelated-looking error.

``rvo2`` (Python-RVO2, the ORCA implementation) is imported unconditionally by
``crowd_sim.envs.crowd_sim`` but is deliberately absent below: it has no PyPI
distribution and must be built from source. See docs/INSTALL.md.
"""

from setuptools import setup

setup(
    name='crowdnav',
    version='1.0.0',
    description='Crowd navigation simulator and evaluation harness for SoGuDiff',
    license='MIT',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    package_data={'crowd_nav': ['configs/*.config']},
    python_requires='>=3.8',
    install_requires=[
        # gym is held below 0.22: the environments use the pre-0.22 step/reset
        # API. numpy is held below 2: gym 0.21 and numba both break on its ABI.
        'gym==0.21.0',
        'matplotlib',
        'numpy>=1.22,<2',
        'scipy',
        'torch>=2,<3',
        'torchvision',
    ],
    extras_require={
        'test': ['pylint', 'pytest'],
    },
)
