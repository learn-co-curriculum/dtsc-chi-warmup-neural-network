import os
import glob
import types

import pickle as pkl

import numpy as np
import pandas as pd
from IPython.display import Markdown, display


class Test():
    """
        Provides a way for students to check their objects against 
        objects created in solution cells

        How to use this class
        ---------------------

        In a solution cell: 

        once you have an object you would like students to replicate,
        call Test().save(object, object_nickname) to pkl the object
        in a class-defined directory

            This will: 
                - check to see if there's a directory at the level of the book called test_obj, 
                and create it if there's not.  (The directory name can be altered 
                with the attribute Test().dir)

                - check to see if there's a file at the path f"test_obj/{object_nickname+'.pkl'}"
                If there is, it is deleted.  (This allows for writing Test().save once and 
                re-saving the object every time the cell is run)

                - pkl the object and save it at the path f"test_obj/{object_nickname+'.pkl'}"

        In a student-facing cell: 

        call Test().run_test(student_object, object_nickname) and have students run the cell
        to check whether their object matches the pkl'd object

            This will: 
                - import the pkl'd object at the path f"test_obj/{object_nickname+'.pkl'}"

                - run an assert against the student-created object "student_object" and the pkl'd object

                - print "Hey, you did it.  Good job" if an AssertError is not thrown, "try again" if one is

                - the variable name of the first parameter is not bound by anything and need not be
                the original name of the object that was pkl'd.  It's up to the instructor whether
                to
                    - tell students to create an object with a specific name, and pre-populate
                    Test.run_test() with that name as the first parameter

                    - not specify a name for students to assign student_object, but rely on
                    the student to place their object as the first parameter for Test.run_test()


        Type-specific assert methods
        ----------------------------

        Some types need additional methods to be asserted.  

        Examples include numpy arrays, pandas series and dataframes.  

        These type-specific methods are run if an object of that type is the first
        parameter for Test().run_test

        A dictionary containing these type-specific assert methods is stored in Test().obj_tests_dict 
        with the type as the key.

        There is a Test().obj_tests_dict_kwargs attribute which contains parameters to pass to 
        type-specific assertion methods.  

        The dataframe assertion method has a "check_like" parameter which ignores the sort order
        and will assert True if identical frames sorted differently are compared.  This class 
        sets "check_like" to True by default.


        Is this robust for students with Windows filepaths?
        ---------------------------------------------------
        Not yet, working on it.
    """
    
    def __init__(self, directory='test_obj'):
                
        self.directory=directory
        
        self.obj_tests_dict = {
            np.ndarray: np.testing.assert_array_equal,
            pd.core.series.Series: pd.testing.assert_series_equal,
            pd.core.frame.DataFrame: pd.testing.assert_frame_equal,
            types.MethodType: lambda x, y: x.__code__.co_code == y.__code__.co_code
        }

        self.obj_tests_dict_kwargs = {
            np.ndarray: {},
            pd.core.series.Series: {},
            pd.core.frame.DataFrame: {'check_like': True},
            types.MethodType: {}
        }

        return

    def test_dir(self):
        '''
        check if test_obj dir is there; if not, create it
        '''

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        return

    def get_file_name(self, glob_listing):
        '''
        gets str of one file name in test_obj dir w/o .pkl extension

        Parameters:
            glob_listing: single returned object from glob

        Output:
            str of file name w/o .pkl extension
        '''

        #get filename
        directory, file_name = os.path.split(glob_listing)
        
        # remove .pkl
        listing, file_extension = os.path.splitext(file_name)
        return listing

    def save_ind(self, object, object_name):
        '''
        saves object to test_obj dir w/ object_name.pkl

        Parameters:
            object: object to pkl
            object_name: pkl file name
        '''
        
        with open(os.path.join(self.directory, f'{object_name}.pkl'), 'wb') as f:
            pkl.dump(object, f)

        return

    def save(self, object, object_name):
        '''
        parse test_obj dir to see if object_name.pkl prev saved
        if so, delete it

        save object under f'test_obj/{object_name}.pkl'

        Parameters:
            object: object to save as pkl file
            object_name: name to save pkl object as under f'test_obj/{object_name+".pkl"}'

        '''
        self.test_dir()

        files = glob.glob(
            os.path.join(
                self.directory, f'{object_name}.pkl'
            )
        )

        existing_files = [self.get_file_name(file) for file in files]

        if object_name+'.pkl' in existing_files:
            os.remove(
                os.path.join(
                    self.directory, f'{object_name}.pkl'
                )
            )

        self.save_ind(object, object_name)

        return

    def load_ind(self, object_name):
        '''
        loads and unpkls object from f"self.dir/{object_name+'.pkl'}"

        returns: unpkl'd object
        '''

        with open(os.path.join(self.directory, f'{object_name}.pkl'), 'rb') as f:
            obj = pkl.load(f)

        return obj

    def output(self, result=True):

        if result:
            display(Markdown('✅ **Hey, you did it.  Good job.**'))
        else:
            display(Markdown('❌ **Try Again**'))

    def run_test(self, obj, name):
        '''
        runs assert against obj and f"self.dir/{name+'.pkl'}"

        checks type of obj and, if type has assert method in self.obj_tests_dict, runs
        that assert method instead.  Any kwargs for that asssert method in 
        obj_tests_dict_kwargs are also passed.
        '''

        kind = type(obj)

        test_obj = self.load_ind(name)

        try:
            if kind in self.obj_tests_dict.keys():
                self.obj_tests_dict[kind](
                    obj, test_obj, **self.obj_tests_dict_kwargs[kind])

            else:
                assert obj == test_obj

            self.output()

        except AssertionError:

            self.output(result=False)
