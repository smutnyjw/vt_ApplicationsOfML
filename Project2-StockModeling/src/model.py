#!/usr/bin/env python3
'''
@module qcommodel.model
@info   Parent module for running end-to-end stock price modeling for the ECE
        5984 group K project2 effort. More to come
'''

# Python libraries
import sys
import os
import yaml
import traceback

# Third party libraries
import pandas as pd
from importlib import import_module

# Local libraries
from utils.logger import Logger
from utils.dataframe import Summarize
from utils.dataframe import Write

## Model implementation class
class Model():
    ''' This class performs configuration loading, data processing, model
        training and final model storage with abstracted methods for
        intermediate states
    '''

    def __init__(self, config_file):
        ''' Constructor
        :param config_file: String path to the configuration file
        '''
        # Parse the configuration file
        self._config = self.readConfig(config_file)
        assert self._config != None, 'Failed to parse configuration data'
        
        # Initialize class member data
        self.initVariables()

        # Setup logging
        self.initLogger()

        # Initialization complete
        self.logger.info('Initialization complete')

    ## Data modeling functionality
    ############################################################################
    @staticmethod
    def loadSteps(steps):
        ''' Load and validate modeling steps
        :param steps: List of step configuration dicts
        :return List: Step objects
        '''
        result = []
        for idx, step in enumerate(steps):
            print(f'Loading step-{idx}')
            result.append(Step(step))
        return result

    def run(self):
        ''' Data prepration wrapper method
        This function uses configuration fields to load, analyze, and prepare
            input data for further modeling
        :return None:
        '''
        # Setup input data/preparation settings
        self.logger.info(f"Processing input files: {self._config_input.keys()}")
        steps = self.loadSteps(self._config_steps)

        outputs = {}
        for key in self._config_input.keys():
            outputs[f'input_{key}'] = self._config_input[key]

        for idx, step in enumerate(steps):
            # Run the step
            self.logger.info('')
            self.logger.info(f'Running step-{idx}: {step.module.__name__}')

            # Handle multiple input objects prior to positional arguments
            inputs = []
            if type(step.input.name) in [list, tuple]:
                inputs += step.input.name
            else:
                inputs = [step.input.name]

            # Get the object mapping to the input for each specified input
            for idx in range(len(inputs)):
                print(inputs[idx])
                inputs[idx] = outputs[inputs[idx]]

            # Call the function with kwargs if specified
            if step.args:
                output = step.module(*inputs, **step.args)
            else:
                output = step.module(*inputs)
            self.logger.info('==================================================')

            # Process the output of the step
            outputs[step.output.name] = output
            if step.output.write:
                ofn = self._config_output['prefix'].replace('%suffix', step.output.name.lower())
                self.logger.info(f'Writing data to: {ofn}')
                Write(output, ofn)
            if step.output.summarize:
                Summarize(output, prefix=step.output.name)
            if step.output.print:
                print(output)
            self.logger.info('==================================================')

    ## Class support functions
    ############################################################################
    def initVariables(self):
        ''' Class member variable initialization
        Initializes all class member variables with defaults/config fields
            => Configuration field requirements are defined by the method of
                access. .get for optional, direct access for requried
        :return None:
        '''
        # Utilities
        self.name = self.__class__.__name__
        self._config_logging = self._config.get('logging', {})

        # Debug
        self._config_debug = self._config.get('debug', {})
        self._summarize = self._config_debug.get('summarize', False)

        # Input data configuration
        self._config_input = self._config['input']

        # Output data configuration
        self._config_output = self._config['output']

        # Modeling steps configuration
        self._config_steps = self._config['steps']

    def initLogger(self):
        ''' Logger initialization function 
        Initializes class member 'logger'
        :return None:
        '''
        # Just pass config file logging params through
        self._config_logging['name'] = self.name
        self.logger = Logger(**self._config_logging)

    @staticmethod
    def readConfig(fn):
        ''' Configuration file parser
        :param fn: String path to the config file
        :return dict: Config file contents
        '''
        result = None
        assert os.path.exists(fn), 'Missing/invalid configuration: {fn}'
        try:
            with open(fn, 'r') as fd:
                result = yaml.safe_load(fd)
        except Exception as exc:
            print(f'Failed to parse configuration: {fn}')
            print(f'\n{traceback.format_exc()}')
        return result

## Modeling 'Step' wrapper class/object
class Step:
    class Input:
        def __init__(self, config):
            self.validate(config)
            self.name = config['name']
        
        def validate(self, config):
            ''' Validate input settings '''
            assert type(config['name']) in [str, list, tuple], f"Invalid input name: {config['name']}"

    class Output:
        def __init__(self, config):
            self.validate(config)
            self.name = str(config['name']).lower()
            self.write = bool(config.get('write', False))
            self.print = bool(config.get('print', False))
            self.summarize = bool(config.get('summarize', False))

        def validate(self, config):
            ''' Validate input settings '''
            assert isinstance(config['name'], str), f"Invalid output name: {config['name']}"

    def __init__(self, config):
        ''' Constructor
        :param method: String path to the python module
        :param input: Python dict with data descriptor fields
            - name: 
        :param output: Python dict with data descriptor fields
            - write: Bool write to file flag
            - print: Bool print object flag
            - summarize: Bool pd.dataframe summary flag
            - name: String name of the output for mapping to other steps
        '''
        self.validate(config)
        module = config['module'].split('.')
        method = module[-1]
        module = '.'.join(module[:-1])
        try:
            self.module = import_module(module)
            self.module = getattr(self.module, method)
        except Exception:
            print(traceback.format_exc())
            print(f"Step - Failed to load module: {config['module']}")
            raise ImportError(f"Failed to load module: {config['module']}")
        self.args = config.get('args', None)
        self.input = self.Input(config['input'])
        self.output = self.Output(config['output'])

    def validate(self, config):
        ''' Step configuration validation '''
        assert 'module' in config, 'Step missing required field "module"'
        assert 'input' in config, 'Step missing required field "input"'
        assert 'output' in config, 'Step missing required field "output"'

if __name__ == '__main__':
    # Create the model class
    if len(sys.argv[1:]):
        fn = sys.argv[1]
    else:
        #fn = os.path.dirname(os.path.realpath(__file__)) + '/conf/ann.yaml'
        #fn = os.path.dirname(os.path.realpath(__file__)) + '/conf/model.yaml'
        fn = os.path.dirname(os.path.realpath(__file__)) + '/conf/ensemble.yaml'
    print(f'Using configuration file: {fn}')
    model = Model(fn)

    # Run the modeling steps
    model.run()