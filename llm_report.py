from numpy import log
import pandas as pd
import pm4py
import pathlib
from openai import OpenAI

# PATH CONFIG
current_dir = pathlib.Path(__file__).parent
DATA_PATH = current_dir / "data"
OUTPUT_FILE = DATA_PATH / "processed_event_log.csv"
API_KEY_FILE = current_dir / "openrouter_api_key.txt"

api_key_file = pathlib.Path(API_KEY_FILE)
if not api_key_file.exists():
    raise FileNotFoundError("API key file non trovato.")

with open(api_key_file, "r") as f:
    api_key = f.read().strip()

# CONFIG - Client OpenAI on OpenRouter
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

#MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"

SYSTEM_PROMPT = """
    You are a senior Process Mining Analyst specialized in E-commerce.

    Goals:
    - Analyze the provided process mining metrics.
    - Identify bottlenecks or UX issues.
    - Suggest strategic business improvements.
    - Explain reasoning clearly and concisely.
    - Provide output in a structured format:

    Reasoning:
    Observations:
    Recommendations:
    """

def process_report():
    """
    Process Mining + Strategic AI Report

    Steps:
        1. Load processed event log
        2. Convert DataFrame to PM4Py EventLog
        3. Extract variants
        4. Compute case durations
        5. Compute additional behavioral metrics
        6. Build textual summary
        7. Send prompt to LLM (OpenAI/OpenRouter)
        8. Returns AI strategic report
    """

    try:
        print("\n[INFO] Starting Process Mining Report ...\n")

        # -----------------------------
        # 1. LOAD EVENT LOG
        # -----------------------------
        df = pd.read_csv(OUTPUT_FILE)

        if "time:timestamp" in df.columns:
            # Converti in datetime con timezone aware (UTC)
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
            # Rendi naive (tolgo timezone) per PM4Py
            df["time:timestamp"] = df["time:timestamp"].dt.tz_localize(None)


        print("[INFO] Event log loaded correctly.")

        # -----------------------------
        # 2a. Convert timestamps for PM4Py
        # -----------------------------
        from pm4py.objects.log.util import dataframe_utils
        df = dataframe_utils.convert_timestamp_columns_in_df(df)

        # -----------------------------
        # 2b. Convert DataFrame -> PM4Py EventLog
        # -----------------------------
        from pm4py.objects.conversion.log import converter as log_converter
        parameters = {
            "case_id_key": "case:concept:name",
            "activity_key": "concept:name",
            "timestamp_key": "time:timestamp"
        }
        log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

        if log is None or len(log) == 0:
            print("[WARNING] Empty event log. Skipping variant analysis and case duration..")
            variants = {}
            durations = []
            sorted_variants = []
        else:
            # -----------------------------
            # 3. VARIANT EXTRACTION
            # -----------------------------
            variants = pm4py.get_variants(log) or {}
            sorted_variants = sorted(
                variants.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            print("[INFO] Process variants extracted.")

        # -----------------------------
        # 5. ADDITIONAL METRICS
        # -----------------------------
        total_cases = df["case:concept:name"].nunique() if "case:concept:name" in df.columns else 0

        # Eventi per sessione
        events_per_case = df.groupby("case:concept:name").size()
        avg_events_per_case = events_per_case.mean() if not events_per_case.empty else 0
        max_events_per_case = events_per_case.max() if not events_per_case.empty else 0
        min_events_per_case = events_per_case.min() if not events_per_case.empty else 0

        # Attività più frequente
        most_common_activity = (
            df["concept:name"].value_counts().idxmax()
            if "concept:name" in df.columns and not df.empty
            else "N/A"
        )

        # Attività di uscita più comune
        if "time:timestamp" in df.columns and not df.empty:
            last_activities = (
                df.sort_values("time:timestamp")
                  .groupby("case:concept:name")
                  .last()
            )
            most_common_exit = last_activities["concept:name"].value_counts().idxmax()
        else:
            most_common_exit = "N/A"

        # Numero varianti uniche
        num_variants = len(variants)

        # -----------------------------
        # 6. BUILD VARIANTS SUMMARY
        # -----------------------------
        top_variants_text = []
        for i, (variant, cases) in enumerate(sorted_variants[:5] if variants else []):
            path = " → ".join(variant)
            freq = len(cases)
            top_variants_text.append(f"{i+1}. {path} ({freq} occurrences)")

        variants_summary = "\n".join(top_variants_text) if top_variants_text else "No variants available."

        # -----------------------------
        # 7. PROMPT ENGINEERING
        # -----------------------------
        prompt = f"""
            DATA SUMMARY:
            - Total Sessions: {total_cases}
            - Average Events per Session: {round(avg_events_per_case, 2)}
            - Max Events in a Session: {max_events_per_case}
            - Min Events in a Session: {min_events_per_case}
            - Most Frequent Activity: {most_common_activity}
            - Most Common Exit Activity: {most_common_exit}
            - Unique Process Variants: {num_variants}

            TOP PROCESS VARIANTS:{variants_summary}

            Tasks for this specific request:
            1. Analyze the main user paths in detail.
            2. Identify bottlenecks or potential UX issues.
            3. Suggest 3-5 actionable strategic business improvements.
            4. Rate your confidence in each suggestion (High/Medium/Low).

            Output format:
            All informations: give info about the process mining analysis (DATA SUMMARY and TOP PROCESS VARIANTS).
            Reasoning:
            Observations:
            Recommendations:
        """

        # -----------------------------
        # 7. LLM CALL
        # -----------------------------
        ai_report = "[ERROR] No response from AI."
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            if response is not None and hasattr(response, "choices") and response.choices:
                ai_report = str(response.choices[0].message.content)

        except Exception as e:
            ai_report = f"[ERROR] Error during AI call: {e}"

        print("\n[INFO] Process Mining Report Completed.\n")

    except FileNotFoundError as e:
        print(f"[ERROR] File missing: {e}")

    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")

    return ai_report
