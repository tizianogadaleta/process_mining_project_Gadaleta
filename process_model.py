import os
import pathlib
import pandas as pd
import pm4py
import streamlit as st

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator


# PATH CONFIG
current_dir = pathlib.Path(__file__).parent
DATA_PATH = current_dir / "data"
PROCESSED_CSV = DATA_PATH / "processed_event_log.csv"
INPUT_CSV_PATH = "data/raw_event_log.csv"


# -------------------------------------------------------
#                1. NORMALIZE EVENT LOG
# -------------------------------------------------------
def normalize_event_log(
    input_path: str,
    session_col: str = "user_session",
    user_col: str = "user_id",
    time_col: str = "event_time",
    event_col: str = "event_type",
    max_sessions: int = 10000,
    min_events_per_session: int = 2
) -> pd.DataFrame:
    """
    Normalize an event log CSV into an XES-like format.

    Parameters:
        - input_path: path to the input csv file

        (Adapted for this project but modifiable for others in the call in main)
        - session_col: column name for session id
        - user_col: column name for user/resource
        - time_col: column name for timestamp
        - event_col: column name for event/activity name

        (Parameters for saving the new dataset to operate on, optional)
        - max_sessions: maximum number of sessions to keep
        - min_events_per_session: minimum number of events per session

    Returns:
        Normalized, cleaned, and filtered CSV event log dataset ("return df" for future applications)
    """
    # Input CSV event log dataset
    df = pd.read_csv(input_path)

    # Validate required columns
    required_columns = {session_col, user_col, time_col, event_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    # Remove rows without id session
    df = df.dropna(subset=[session_col])

    # Filter sessions with at least min_events_per_session events
    session_counts = df[session_col].value_counts()
    valid_sessions = session_counts[session_counts >= min_events_per_session].index
    df = df[df[session_col].isin(valid_sessions)].copy()

    # Select first max_sessions valid sessions
    individual_sessions = df[session_col].unique()[:max_sessions]
    df = df[df[session_col].isin(individual_sessions)].copy()
    
    # Convert time column to datetime standard format
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Sort by session and timestamp
    df = df.dropna(subset=[time_col])
    df = df.sort_values(by=[session_col, time_col])
    
    # Rename columns (XES-like standard)
    rename_map = {
        session_col: "case:concept:name",
        user_col: "org:resource",
        time_col: "time:timestamp",
        event_col: "concept:name"
    }
    df = df.rename(columns=rename_map)

    # Save final CSV
    df.to_csv(PROCESSED_CSV, index=False)

    print("Normalized event log successfully generated")
    return df


# -------------------------------------------------------
#                 2. LOAD EVENT LOG
# -------------------------------------------------------
@st.cache_resource
def load_event_log(csv_path: str,
                   time_column: str = "time:timestamp",
                   case_column: str = "case:concept:name",
                   activity_column: str = "concept:name"):
    """
    Loads a CSV file into a pandas DataFrame and converts it to a PM4Py Event Log.

    Steps:
        1. Reads the CSV into a DataFrame.
        2. Filters out rows where the activity column is empty or NaN.
        3. Converts the timestamp column to datetime, dropping invalid entries.
        4. Sorts the DataFrame by case ID and timestamp.
        5. Converts the DataFrame to a PM4Py-compatible Event Log.
        6. Returns both the cleaned DataFrame and the Event Log.

    If an error occurs, displays an error message in Streamlit and returns an empty DataFrame and None.
    """
    try:
        df = pd.read_csv(csv_path)

        # Ensure activity column is string and filter out empty values
        df[activity_column] = df[activity_column].astype(str)
        df = df[df[activity_column].notna() & (df[activity_column] != "")]

        # Convert timestamp column to datetime and drop invalid entries (double check)
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df = df.dropna(subset=[time_column])

        # Sort the DataFrame by case ID and timestamp
        df = df.sort_values(by=[case_column, time_column])

        # Parameters for PM4Py Event Log conversion
        parameters = {
            "case_id_key": case_column,
            "activity_key": activity_column,
            "timestamp_key": time_column
        }

        # Convert the DataFrame to a PM4Py Event Log
        log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

        return df, log

    except Exception as e:
        st.error(f"[ERROR] Unable to load event log: {e}")
        return pd.DataFrame(), None


# -------------------------------------------------------
#                 3. CALCULATE METRICS
# -------------------------------------------------------
def calculate_quality_metrics(net, initial_marking, final_marking, event_log):
    """
    Calculate key quality metrics for a Petri net given an event log.

    Metrics computed:
    1. Fitness: How well the model can reproduce the observed behavior in the log.
    2. Precision: How much the model allows behavior not seen in the log.
    3. Generalization: How well the model generalizes to unseen but possible behavior.
    4. Simplicity: A measure of model complexity (fewer elements = simpler).

    Returns:
        tuple: (fitness_value, precision_value, generalization_value, simplicity_value)
    """
    try:
        # Token-based replay fitness evaluation
        fitness_value = fitness_evaluator.apply(
            event_log, net, initial_marking, final_marking,
            variant=fitness_evaluator.Variants.TOKEN_BASED
        )['log_fitness']

        # Precision evaluation using ET Conformance Token method
        precision_value = precision_evaluator.apply(
            event_log, net, initial_marking, final_marking,
            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
        )

        # Generalization evaluation
        generalization_value = generalization_evaluator.apply(
            event_log, net, initial_marking, final_marking
        )

        # Simplicity metric: inverse of the total number of places, transitions, and arcs
        simplicity_value = 1 / (len(net.places) + len(net.transitions) + len(net.arcs))

        return fitness_value, precision_value, generalization_value, simplicity_value

    except Exception as e:
        # Warn in case metric calculation fails
        print(f"[WARN] Metric calculation failed: {e}")
        return 0, 0, 0, 0


# -------------------------------------------------------
#                 4. PROCESS DISCOVERY
# -------------------------------------------------------
@st.cache_resource
def discover_process_models(csv_path: str):
    """
    Discover process models from an event log CSV and compute quality metrics.

    Steps:
    1. Load event log (DataFrame + PM4Py Event Log).
    2. Initialize storage for metrics and Petri net images.
    3. Discover process models using Alpha Miner, Heuristic Miner, and Inductive Miner.
    4. For each model:
        - Compute fitness, precision, generalization, and simplicity metrics.
        - Save Petri net visualization as PNG in .\img.
    5. Return metrics dictionary, image paths dictionary, and the original DataFrame for future applications.
    """
    df, event_log = load_event_log(csv_path)
    if df.empty or event_log is None:
        return {}, {}, pd.DataFrame()
    
    metrics_results = {}
    images_paths = {}
    
    # Folder to save Petri net visualizations
    IMAGES_PATH = pathlib.Path("img")
    IMAGES_PATH.mkdir(exist_ok=True)
    
    # Helper function to compute quality metrics
    def compute_metrics(net, im, fm, log_data):
        return calculate_quality_metrics(net, im, fm, log_data)
    
    # ----- Alpha Miner -----
    try:
        net_alpha, im_alpha, fm_alpha = pm4py.discover_petri_net_alpha(event_log)
        alpha_img = IMAGES_PATH / 'alpha_petri_net.png'
        pm4py.save_vis_petri_net(net_alpha, im_alpha, fm_alpha, str(alpha_img))
        fit, prec, gen, simp = compute_metrics(net_alpha, im_alpha, fm_alpha, event_log)
        metrics_results['Alpha_miner'] = {'fitness': fit, 'precision': prec, 'generalization': gen, 'simplicity': simp}
        images_paths['Alpha_miner'] = str(alpha_img)
    except Exception as e:
        print(f"[WARN] Alpha miner failed: {e}")
    
    # ----- Heuristic Miner -----
    try:
        heu_net = pm4py.discover_heuristics_net(event_log)
        net_heuristic, im_heuristic, fm_heuristic = pm4py.convert_to_petri_net(heu_net)
        heuristic_img = IMAGES_PATH / 'heuristic_petri_net.png'
        pm4py.save_vis_petri_net(net_heuristic, im_heuristic, fm_heuristic, str(heuristic_img))
        fit, prec, gen, simp = compute_metrics(net_heuristic, im_heuristic, fm_heuristic, event_log)
        metrics_results['Heuristic_miner'] = {'fitness': fit, 'precision': prec, 'generalization': gen, 'simplicity': simp}
        images_paths['Heuristic_miner'] = str(heuristic_img)
    except Exception as e:
        print(f"[WARN] Heuristic miner failed: {e}")
    
    # ----- Inductive Miner -----
    try:
        tree_inductive = pm4py.discover_process_tree_inductive(event_log)
        net_inductive, im_inductive, fm_inductive = pm4py.convert_to_petri_net(tree_inductive)
        inductive_img = IMAGES_PATH / 'inductive_petri_net.png'
        pm4py.save_vis_petri_net(net_inductive, im_inductive, fm_inductive, str(inductive_img))
        fit, prec, gen, simp = compute_metrics(net_inductive, im_inductive, fm_inductive, event_log)
        metrics_results['Inductive_miner'] = {'fitness': fit, 'precision': prec, 'generalization': gen, 'simplicity': simp}
        images_paths['Inductive_miner'] = str(inductive_img)
    except Exception as e:
        print(f"[WARN] Inductive miner failed: {e}")
    
    return metrics_results, images_paths, df


# -------------------------------------------------------
#          5. PROCESS_DISCOVERY WRAPPER (MAIN)
# -------------------------------------------------------
def process_discovery():
    """
    Main wrapper function that orchestrates the full Process Discovery pipeline
    used in the Streamlit dashboard.

    Summary of steps:
        1. Checks if the normalized event log CSV already exists.
        2. If not, triggers the normalization process starting from the raw input log.
        3. Executes the process discovery algorithms (Alpha, Heuristic, Inductive).
        4. Collects discovered models, evaluation metrics, and generated visualizations.
        5. Returns all results to be rendered in the dashboard interface.

    Output:
        - metrics_results: dictionary containing quality metrics for each discovered model.
        - images_paths: dictionary with file paths to generated Petri Net visualizations.
        - df_discovery: cleaned and processed DataFrame used for discovery.
    """
    if not os.path.exists(PROCESSED_CSV):
        print("\n[INFO] Normalizing event log ...\n")
        normalize_event_log(input_path=INPUT_CSV_PATH)

    metrics_results, images_paths, df_discovery = discover_process_models(PROCESSED_CSV)
    return metrics_results, images_paths, df_discovery
