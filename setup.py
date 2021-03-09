# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install script for setuptools."""

import datetime
from importlib import util as import_util
import sys

from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location('_metadata', 'mava/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

reverb_requirements = [
    'dm-reverb',
    'tensorflow>=2.4.0',
]

tf_requirements = [
    'tensorflow>=2.4.0',
    'tensorflow_probability',
    'dm-sonnet',
    'trfl',
]

env_requirements = [
    'pettingzoo'
]

testing_requirements = [
    'pytype',
    'pytest-xdist',
]

long_description = \
"""Mava is a library for building multi-agent reinforcement learning (MARL) systems. Mava builds off of Acme and in a similar way strives to expose simple, efficient, and readable components, as well as examples that serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research. 
For more information see [github repository](https://github.com/instadeepai/mava)."""

# Get the version from metadata.
version = _metadata.__version__

# If we're releasing a nightly/dev version append to the version string.
if '--nightly' in sys.argv:
  sys.argv.remove('--nightly')
  version += '.dev' + datetime.datetime.now().strftime('%Y%m%d')

setup(
    name='id-mava',
    version=version,
    description='A Python library for Multi-Agent Reinforcement Learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='InstaDeep',
    license='Apache License, Version 2.0',
    keywords='multi-agent reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm_env',
        'dm-tree',
        'numpy',
        'pillow',
    ],
    extras_require={
        'tf': tf_requirements,
        'envs': env_requirements,
        'reverb': reverb_requirements,
        'testing': testing_requirements,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)