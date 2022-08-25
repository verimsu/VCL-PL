"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'CelebA'}
        assert(database in db_names)

        if database == 'CelebA':
            return '/home/mehmetyavuz/datasets/CelebA128/'
        else:
            raise NotImplementedError
