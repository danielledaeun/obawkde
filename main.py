import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sota import Baseline, Sliding, OOB, AREBA
from proposing import OB_awKDE
from standard import NN_standard
from utils import simulation_data
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm 
import multiprocessing
import gc
import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n
    return s, n, metric

def update_delayed_metric(prev, flag, forget_rate):
    return (1.0 - forget_rate) * flag + forget_rate * prev

def run(data, config, models, method):
    time_steps = config['time_steps']
    time_drift_start = config['time_drift_at']
    input_layer = config['layer_dims'][0]

    #------------ initialisation ------------#
    preq_rec0s = np.zeros(time_steps)
    preq_rec1s = np.zeros(time_steps)
    preq_gmeans = np.zeros(time_steps)

    preq_rec0, preq_rec1 = (1.0,) * 2
    preq_rec0_s, preq_rec0_n, preq_rec1_s, preq_rec1_n = (0.0,) * 4
    w_neg, w_pos = (0.0,) * 2

    #------------ method specification ------------#
    technique = Baseline(models[0])

    if method == 'sliding':
        technique = Sliding(models[0], sliding_window_size=100)
    elif method == 'oob_single' or method == 'oob':
        technique = OOB(models)
    elif method == 'areba':
        technique = AREBA(models[0], queue_size_budget=20)
    elif method == 'mine':
        technique = OB_awKDE(models, queue_size=20)

    for t in range(time_steps):
        if t == time_drift_start:
            preq_rec1, preq_rec1_s, preq_rec1_n = (0.0,) * 3
            preq_rec0, preq_rec0_s, preq_rec0_n = (0.0,) * 3

        # Get current sample
        x = tf.convert_to_tensor(
            data.iloc[t,:-1].values.reshape(1, -1), 
            dtype=tf.float32
        )
        y = tf.convert_to_tensor(
            [[data.iloc[t,-1]]], 
            dtype=tf.float32
        )

        # Predict
        _, y_hat_class = technique.predict(x)
        
        # Check correctness
        example_neg = (y.numpy() == 0)
        correct = int(y.numpy().item()==y_hat_class[0].item())

        # Update metrics
        if example_neg:
            preq_rec0_s, preq_rec0_n, preq_rec0 = update_preq_metric(
                preq_rec0_s, preq_rec0_n, correct, 0.99)
        else:
            preq_rec1_s, preq_rec1_n, preq_rec1 = update_preq_metric(
                preq_rec1_s, preq_rec1_n, correct, 0.99)

        preq_gmean = np.sqrt(preq_rec0 * preq_rec1)
        
        # Store results
        preq_rec0s[t] = preq_rec0
        preq_rec1s[t] = preq_rec1
        preq_gmeans[t] = preq_gmean

        # Update delayed metrics
        w_neg = update_delayed_metric(w_neg, example_neg, 0.99)
        w_pos = update_delayed_metric(w_pos, not example_neg, 0.99)

        # Method-specific updates
        if method == 'sliding':
            x, y = technique.append_to_win(x.numpy(), y.numpy(), input_layer)
        elif method in ['oob_single', 'oob']:
            imbalance_rate = 1.0
            if (y.numpy() == 1) and (w_pos < w_neg) and (w_pos != 0.0):
                imbalance_rate = w_neg / w_pos
            elif (y.numpy() == 0) and (w_neg < w_pos) and (w_neg != 0.0):
                imbalance_rate = w_pos / w_neg
            technique.oob_oversample(np.random.RandomState(config['seed']), imbalance_rate)
        elif method == 'areba':
            technique.append_to_queues(x.numpy(), y.numpy())
            technique.adapt_queues(w_pos, w_neg)
            x, y = technique.get_training_set(input_layer)
        elif method == 'mine':
            technique.append_to_queues(x, y)
            if (w_pos < w_neg) and (y==1):
                #technique.kde_oversamping(self, target, n_features, k, seed)
                technique.kde_oversamping(1, input_layer, 3, 0)
            elif (w_pos > w_neg) and (y==0):
                technique.kde_oversamping(0, input_layer, 3, 0)

        # Train
        if method == 'mine':
            technique.train(x, y, input_layer, np.random.RandomState(config['seed']))
        else:
            technique.train(x, y)

    return preq_rec0s, preq_rec1s, preq_gmeans

def process_repeat(r, config):
    try:
        # Device selection - round-robin across available devices
        available_devices = ['/CPU:0']  # Start with CPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            available_devices.extend([f'/GPU:{i}' for i in range(len(gpus))])
        
        # Select device based on repeat number (round-robin)
        device = available_devices[r % len(available_devices)]
        
        # Create models with device specification
        if config['method'] == 'oob':
            ensemble_size = 30
        elif config['method'] == 'oob_single':
            ensemble_size = 1
        elif config['method'] == 'mine':
            ensemble_size = 1
        else:
            ensemble_size = None

        # Base model parameters
        model_params = {
            'layer_dims': config['layer_dims'],
            'learning_rate': config['learning_rate'],
            'output_activation': 'sigmoid',
            'loss_function': tf.keras.losses.BinaryCrossentropy(),
            'num_epochs': 1,
            'weight_init': "he",
            'class_weights': {0: 1.0, 1: 1.0},
            'minibatch_size': 1,
            'device': device  # Add device parameter
        }

        # Create initial model
        models = [NN_standard(**model_params)]

        # Create ensemble if needed using list comprehension
        if ensemble_size is not None:
            models.extend([
                NN_standard(**{**model_params, 'seed': config['seed'] + i + 1})
                for i in range(ensemble_size - 1)
            ])

        # Generate data
        df = simulation_data(
            random_state=np.random.RandomState(r),
            time_steps=config['time_steps'],
            time_drift=config['time_drift_at'],
            boundary=config['boundary'],
            ir=config['ir'],
            imbtype=config['imbtype']
        )

        # Run experiment
        result = run(
            data=df,
            config=config,
            models=models,
            method=config['method']
        )
        
        # Clear memory
        del df, models
        gc.collect()
        tf.keras.backend.clear_session()
        
        return result
        
    except Exception as e:
        print(f"Error in repeat {r}: {e}")
        raise

def main(config):
    # Initialize TensorFlow device handling
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    else:
        print("No GPUs found. Running on CPU.")

    # Validate configuration
    valid_methods = ['baseline', 'sliding', 'oob_single', 'oob', 'areba', 'mine']
    valid_boundaries = ['sine', 'sea', 'circle']
    valid_imbtypes = ['safe', 'borderline', 'noise']

    if config['method'] not in valid_methods:
        raise ValueError(f"Invalid method. Must be one of {valid_methods}")
    if config['boundary'] not in valid_boundaries:
        raise ValueError(f"Invalid boundary. Must be one of {valid_boundaries}")
    if config['imbtype'] not in valid_imbtypes:
        raise ValueError(f"Invalid imbtype. Must be one of {valid_imbtypes}")
    
    print(f"Experiments: {config['method']}-{config['boundary']}-{config['imbtype']}-{config['ir']}-{config['repeats']} times")

    # Calculate optimal number of workers
    optimal_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free for system
    print(f"Running with {optimal_workers} workers")
    
    # Set random seeds
    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup output directory - add error handling
    out_dir = os.path.join('res', config['boundary'], config['imbtype'], 
                          str(int(config['ir'] * 100)))
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving results to: {out_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        raise

    # Initialize results storage
    rec0, rec1, gmu = [], [], []
    
    # Modify the progress bar for cleaner output
    with tqdm(total=config['repeats'], 
             desc=f"Processing {config['method']}", 
             unit="repeat",
             ncols=80,
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
             ) as pbar:
        
        results = []
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            process_with_config = partial(process_repeat, config=config)
            
            for result in executor.map(process_with_config, range(config['repeats'])):
                results.append(result)
                pbar.update(1)
    
    print("\nProcessing complete. Saving results...")
    
    # Unpack results
    rec0, rec1, gmu = zip(*results)

    # Save results
    header = np.arange(config['time_steps'])
    np.savetxt(os.path.join(out_dir, f'{config["method"]}_rec0.csv'), rec0, header=','.join(map(str, header)),delimiter=',')
    np.savetxt(os.path.join(out_dir, f'{config["method"]}_rec1.csv'), rec1, header=','.join(map(str, header)), delimiter=',')
    np.savetxt(os.path.join(out_dir, f'{config["method"]}_gmu.csv'), gmu, header=','.join(map(str, header)), delimiter=',')

if __name__ == "__main__":
    config = {
        'seed': 0,
        'time_steps': 5000,
        'time_drift_at': 2500,
        'repeats': 100, #100
        'boundary': 'sea', #'sine', 'sea', 'circle'
        'ir': 0.1, # 0.1, 0.01, 0.001
        'imbtype': 'noise', #'safe', 'borderline', 'noise'
        'learning_rate': 0.01,
        'layer_dims': [2, 8, 1],
        'method': 'mine' # 'baseline', 'sliding', 'oob_single', 'oob', 'areba', 'mine'
        }
    
    # # a)run one experiment
    # main(config)

    # # b) run all methods for current dataset; NOTE: oob takes a long time so we don't run it here
    # for method in ['baseline', 'sliding', 'oob_single', 'areba', 'mine']:
    #     config['method'] = method
    #     main(config)

    # c) run all methods for all datasets and imbalance ratios
    # Define all possible values for each parameter
    boundaries = ['sine', 'sea', 'circle']
    imbtypes = ['safe', 'borderline', 'noise']
    irs = [0.1, 0.01, 0.001]
    methods = ['baseline', 'sliding', 'oob_single', 'areba', 'mine']  # Excluding 'oob' as it takes too long

    # Run all combinations
    total_experiments = len(boundaries) * len(imbtypes) * len(irs) * len(methods)
    current_experiment = 0

    print(f"Starting {total_experiments} experiments...")
    
    for boundary in boundaries:
        for imbtype in imbtypes:
            for ir in irs:
                for method in methods:
                    current_experiment += 1
                    print(f"\nExperiment {current_experiment}/{total_experiments}")
                    print(f"Configuration: {boundary}-{imbtype}-{ir}-{method}")
                    
                    # Update config
                    config.update({
                        'boundary': boundary,
                        'imbtype': imbtype,
                        'ir': ir,
                        'method': method
                    })
                    
                    try:
                        main(config)
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        continue

    print("\nAll experiments completed!")