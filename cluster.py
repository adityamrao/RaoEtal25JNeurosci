from IPython.display import display, clear_output
from cmldask import CMLDask
from time import sleep, time
from cmldask.CMLDask import get_exceptions


def get_exceptions_quiet(futures, params):
    try:
        exceptions = get_exceptions(futures, params)
        print('Exceptions occurred during cluster run!')
        return exceptions
    except Exception as e:
        return None


def wait(futures, client, check_delay=10, cancel_prop=1.0, checkpoint_file=None, visualize_func=None):
    # periodically get results, check for errors, save checkpoint, and display updated visualization
    assert isinstance(check_delay, (float, int))
    start = time()
    while True:
        sleep(check_delay)
        clear_output()
        # TODO: should probably reserve full result gathering if visualize_func is specified to reduce network load/overhead
        finished_futures = CMLDask.filter_futures(futures)
        n_finished = len(finished_futures)
        
        # check for errors and cancel simulation if over cancel_prop proportion of jobs are errors
        errors = None
        try:  # CMLDask throws an error if there were no exceptions in any Dask jobs
            errors = CMLDask.get_exceptions(futures, range(len(futures)))
            print("Dask Errors:")
            print(errors.head())
            n_errors = len(errors)
        except:
            n_errors = 0

        dur = time() - start
        rate = -1.0 if not n_finished else n_finished/dur
        print(f'Simulations finished after {dur:0.3} s: {n_finished + n_errors} / {len(futures)} ({rate:0.3} iterations/s). {n_errors} job errors')
        
        if checkpoint_file or (visualize_func is not None):
            results = client.gather(finished_futures)
        
        if visualize_func is not None:
            display(visualize_func(results))

        if checkpoint_file:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
                
        if n_finished + n_errors == len(futures):
            print('Simulation complete. Shutting down jobs.')
            break
