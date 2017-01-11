#######################################################################
# Copyright (C) 2017 Artem Oboturov(oboturov@gmail.com)               #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import setuptools
from setuptools import setup
from setuptools.command.test import test as TestCommand

NAME = 'reinforcement-learning-an-introduction'
VERSION = '0.1.0'
DESCRIPTION = 'Python code for Sutton & Barto\'s book'
LICENSE = ''
URL = 'https://github.com/ShangtongZhang/reinforcement-learning-an-introduction'
AUTHOR = 'Shangtong Zhang'
EMAIL = 'zhangshangtong.cpp@gmail.com'
KEYWORDS = ''


class PyTest(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        sys.exit(0)


def setup_package():
    # Assemble additional setup commands
    cmdclass = {}
    cmdclass['test'] = PyTest

    setup(
        name=NAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        license=LICENSE,
        keywords=KEYWORDS,
        long_description='',
        classifiers=[],
        test_suite='',
        packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
        install_requires=['numpy', 'matplotlib', 'six', 'seaborn'],
        setup_requires=[],
        cmdclass=cmdclass,
        tests_require=['pytest'],
        command_options={},
        entry_points={},
    )


if __name__ == '__main__':
    setup_package()
