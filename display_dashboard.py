import os
import pathlib
import pandas as pd
import streamlit as st

from openai import OpenAI

from process_model import process_discovery
from llm_report import process_report

st.set_page_config(
    page_title="Process Mining AI Dashboard",
)

# PATH CONFIG
current_dir = pathlib.Path(__file__).parent
DATA_PATH = current_dir / "data"
API_KEY_FILE = current_dir / "openrouter_api_key.txt"

# LOAD API KEY
if not API_KEY_FILE.exists():
    raise FileNotFoundError(f"API key file not found: {API_KEY_FILE}")

with open(API_KEY_FILE, "r") as f:
    api_key = f.read().strip()

# CONFIG OPENAI with OpenRouter API
ai_client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

#MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"

# PROMPTS
SYSTEM_PROMPT = """
You are a senior Process Mining and Business Intelligence analyst specialized in e-commerce analytics and user behavior modeling.

Global Context:
- The system is an online cosmetics e-commerce platform.
- You work primarily with event logs representing user interaction sequences.
- Typical user activities include: view, cart, remove_from_cart, purchase.
- Event logs may contain timestamps, case IDs, and activity sequences.
- Your objective is to analyze behavioral patterns, predict next activities, and extract strategic business insights.

Analytical Responsibilities:
- Analyze user action sequences (paths) using process mining principles.
- Explain why a predicted next activity is the most likely based on behavioral patterns, transition probabilities, and historical trends.
- Detect bottlenecks, inefficiencies, drop-off points, or anomalous behaviors.
- Provide data-driven business recommendations to optimize user journeys, increase conversion rates, and improve customer experience.
- Evaluate and clearly state the confidence level of predictions (High / Medium / Low).

Communication Style:
- Maintain a serious, professional, and analytical tone.
- Be concise but precise.
- Avoid vague statements; justify conclusions with logical reasoning.
- Focus on actionable and business-oriented insights.

Response Structure Guidelines:
- Reasoning: Detailed explanation of the analytical logic behind predictions or findings.
- Business Actions: At least 3 concrete and actionable recommendations.
- Confidence: Explicit confidence assessment when predictions are involved.

General Rules:
- Do not invent activities outside the defined domain unless explicitly stated.
- Base reasoning on patterns, probabilities, and process logic rather than intuition.
- Prioritize clarity, consistency, and practical usefulness over verbosity.
"""

CHATBOT_SYSTEM_PROMPT = """
You are an expert Process Mining model evaluation analyst specialized in process discovery quality assessment.

Scope and Knowledge Limits:
- You ONLY analyze process mining quality metrics.
- Available metrics are strictly limited to: Fitness, Precision, Generalization, Simplicity.
- Available process discovery algorithms are strictly limited to: Alpha Miner, Heuristic Miner, Inductive Miner.
- Do not introduce additional metrics, algorithms, or external concepts unless explicitly provided.

Your Tasks (when the user requests):
- Analyze the provided process mining quality metrics.
- Compare algorithm performance based on the given metrics.
- Identify weaknesses, trade-offs, or inconsistencies in the discovered models.
- Detect potential anomalies or overfitting/underfitting patterns.
- Explain your reasoning clearly and concisely.
- Provide actionable insights based only on the observed metrics.

Communication Style:
- Maintain a professional, analytical, and objective tone.
- Be concise but precise.
- Avoid speculation not supported by the provided metrics.
- Focus strictly on metric-driven evaluation.
"""


# -------------------------------------------------------
#  DISPLAY METRICS, PETRI NETS + NEXT EVENT PREDICTION
# -------------------------------------------------------
def display_process_mining_dashboard(
    metrics_results: dict,
    images_paths: dict,
    df: pd.DataFrame,
    decimal_places: int = 3
):
    """
    Display the process mining feature (described in process_model.py) on viewable dashboard in Streamlit.
    Adds next event prediction and AI reasoning assistant.

    Summary:
        1. Metrics and Petri Nets: Display quality metrics and Petri net images.
        2. Next Event Prediction: User inputs a sequence of activities, predicts next activity, and explains via AI.
        3. AI Strategic Report: Button to generate an AI report analyzing the process mining results (described in llm_report.py).
        4. AI Analyst Chatbot: Interactive chat with AI analyst about the process mining results.
    """

    # SIDEBAR / QUICK NAVIGATION DROPDOWN
    st.sidebar.markdown("## Quick Navigation")

    # CSS
    st.sidebar.markdown(
        """
        <style>
        /* Sidebar selectbox container */
        .sidebar .stSelectbox>div>div>div>select {
            background-color: #1f1f1f;   /* sfondo scuro */
            color: #ffffff;               /* testo bianco */
            border: 2px solid #ff4b4b;    /* bordo rosso */
            border-radius: 8px;           /* angoli arrotondati */
            padding: 5px 10px;
            font-size: 14px;
        }

        /* Dropdown options hover */
        .sidebar .stSelectbox>div>div>div>select option:hover {
            background-color: #333333;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Navigation list options
    navigation_options = {
        "Metrics and Petri Nets": "#quality-metrics-and-petri-nets",
        "Next Event Prediction": "#next-event-prediction",
        "AI Strategic Report": "#ai-strategic-process-report",
        "AI Chatbot Conversation": "#ai-chatbot"
    }

    # Dropdown Navigation
    selected_nav = st.sidebar.selectbox(
        "Sections",
        options=list(navigation_options.keys())
    )

    # Redirect to the corresponding section
    st.sidebar.markdown(f"[Go → {selected_nav}]({navigation_options[selected_nav]})")


    # ---- MAIN CONTENT ---- 

    # -------------------------------------------------------
    #               1. METRICS AND PETRI NET
    # -------------------------------------------------------
    st.markdown("### Quality Metrics and Petri Nets") 
    st.write("This section presents the discovered process models with their quality " \
        "metrics and corresponding Petri Net visualizations. " \
        "It allows you to compare different mining algorithms in terms " \
        "of fitness, precision, generalization, and simplicity.")

    for miner_name in metrics_results.keys():
        metrics = metrics_results[miner_name]

        with st.expander(f"{miner_name.replace('_', ' ')}", expanded=False):
            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.metric("Fitness", f"{metrics['fitness']:.{decimal_places}f}")
                st.metric("Precision", f"{metrics['precision']:.{decimal_places}f}")
                st.metric("Generalization", f"{metrics['generalization']:.{decimal_places}f}")
                st.metric("Simplicity", f"{metrics['simplicity']:.{decimal_places}f}")

            with right_col:
                if miner_name in images_paths and os.path.exists(images_paths[miner_name]):
                    st.image(
                        images_paths[miner_name],
                        caption=f"{miner_name} Petri Net",
                        width="stretch"
                    )
                else:
                    st.warning("Immagine Petri Net non trovata.")


    # -------------------------------------------------------
    #               2. NEXT EVENT PREDICTION 
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### Next Event Prediction")
    st.write(
        "This section predicts the most likely next user activity based on the selected sequence of events. "
        "It analyzes historical session patterns and provides an AI-generated explanation with business insights and confidence level."
    )

    if 'case:concept:name' not in df.columns or 'concept:name' not in df.columns:
        st.error("Required columns 'case:concept:name' or 'concept:name' not found in DataFrame.")
        return

    # ---------------- INIT SESSION STATE ----------------
    if "selected_path" not in st.session_state:
        st.session_state.selected_path = []

    if "predicted_event" not in st.session_state:
        st.session_state.predicted_event = None

    if "ai_prediction_report" not in st.session_state:
        st.session_state.ai_prediction_report = None

    if "run_prediction" not in st.session_state:
        st.session_state.run_prediction = False

    # Lista attività
    activities = df['concept:name'].unique().tolist()

    # ---------------- STEP LAYOUT ----------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.caption("STEP 1")
        st.write("**Select User Activities**")

        subcol1, subcol2 = st.columns([3, 1])
        with subcol1:
            activity_choice = st.selectbox(
                "Choose activity",
                options=activities,
                label_visibility="collapsed"
            )

        with subcol2:
            if st.button("Add"):
                st.session_state.selected_path.append(activity_choice)

        if st.session_state.selected_path:
            chips = " → ".join(st.session_state.selected_path)
            st.markdown(f"**Current Sequence:** {chips}")

            st.button(
                "Clear Path",
                on_click=lambda: st.session_state.update({
                    "selected_path": [],
                    "predicted_event": None,
                    "ai_prediction_report": None,
                    "run_prediction": False
                })
            )

    with col2:
        st.caption("STEP 2")
        st.write("**Run Analysis**")
        if st.button("Predict Next Event"):
            st.session_state.run_prediction = True

    # ---------------- PREDICTION LOGIC ----------------
    if st.session_state.run_prediction and st.session_state.selected_path:

        selected_path = st.session_state.selected_path

        # Session dictionary construction -> activity list
        traces = df.groupby('case:concept:name')['concept:name'].apply(list)

        # Count which activities follow the choosen sequence exactly
        next_event_counts = {}
        for trace in traces:
            for i in range(len(trace) - len(selected_path)):
                window = trace[i:i + len(selected_path)]
                if window == selected_path:
                    next_event = trace[i + len(selected_path)]
                    next_event_counts[next_event] = next_event_counts.get(next_event, 0) + 1

        if next_event_counts:
            next_event_prediction = max(next_event_counts, key=next_event_counts.get)
        else:
            next_event_prediction = "None"

        st.session_state.predicted_event = next_event_prediction

        # ---------------- RESULT PANEL ----------------
        if next_event_prediction == "None":
            st.write(
                "No further events are predicted. "
                "This indicates that the selected sequence is not found in the event log."
            )
        else:
            st.markdown("### Predicted Next Activity")
            st.write(
                f"The **next predicted event** based on the selected sequence is: **{next_event_prediction}**. "
                "This means that, given the pattern of activities chosen, this event is the most likely to occur immediately afterwards."
            )

        # ---------------- AI EXPLANATION ----------------
        prompt = f"""
        You are a Process Mining expert analyzing the following scenario:
        User path: {selected_path} and Predicted next activity for that Path: {next_event_prediction}

        Your tasks for this specific request are:
        1. Explain in detail why this activity is predicted as the most likely next step, considering patterns in user behavior and process flows.
        2. Provide from 3 to 5 actionable business recommendations to optimize the process or improve efficiency.
        3. Assess and state your confidence in this prediction (High / Medium / Low), providing reasoning.

        Please structure your response in clear sections:

        Reasoning:
        Explain your analysis and rationale for the predicted next activity.

        Business Actions:
        List actionable recommendations to enhance the process based on your analysis.

        Confidence:
        State your confidence level and justify it.
        """


        try:
            ai_response = ai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            answer = None
            if ai_response and hasattr(ai_response, "choices") and len(ai_response.choices) > 0:
                answer = ai_response.choices[0].message.content

            st.session_state.ai_prediction_report = answer

        except Exception as e:
            st.error(f"AI Error: {e}")
            st.session_state.ai_prediction_report = None

    # ---------------- DISPLAY AI PREDICTION REPORT ----------------
    if st.session_state.ai_prediction_report:
        st.markdown("### AI Insight")
        st.write(
            "This AI-generated insight explains the behavioral logic behind the selected sequence of activities, "
            "clarifies why the predicted next event is the most likely outcome, and lists actionable "
            "business recommendations to improve and optimize the process performance."
        )

        st.info(st.session_state.ai_prediction_report)


    # -------------------------------------------------------
    #                     3. AI STRATEGIC REPORT
    # -------------------------------------------------------
    st.divider()
    st.markdown("### AI Strategic Process Report") 
    st.write(
        "This AI strategic report is based on the overall data summary, including total user sessions, "
        "average session duration, and the most frequent process variants. It analyzes behavioral patterns, "
        "identifies potential bottlenecks, and provides actionable business recommendations with confidence levels."
    )

    # init session state 
    if "ai_strategic_report" not in st.session_state:
        st.session_state.ai_strategic_report = None

    # Report generation button
    if st.button("Generate AI Strategic Report"):
        with st.spinner("Generating AI report..."):
            try:
                ai_report = process_report()

                if ai_report:
                    st.session_state.ai_strategic_report = ai_report
                else:
                    st.warning("No report generated.")

            except Exception as e:
                st.error(f"Error generating report: {e}")

    # display report
    if st.session_state.ai_strategic_report:
        st.markdown(st.session_state.ai_strategic_report)


    # -------------------------------------------------------
    #                   4. AI ANALYST CHAT
    # -------------------------------------------------------
    st.divider()
    st.markdown("### AI Chatbot") 
    st.write("Interact with the AI analyst to ask questions about quality metrics and mining algorithm results.")

    # ---- CONTEXT ----
    context_lines = []
    for algo, data in metrics_results.items():
        context_lines.append(
            f"{algo} -> Fitness {data['fitness']:.2f}, "
            f"Precision {data['precision']:.2f}, "
            f"Generalization {data['generalization']:.2f}, "
            f"Simplicity {data['simplicity']:.2f}"
        )

    context = "\n".join(context_lines)

    # ---- SESSION INIT ----
    if "pm_messages" not in st.session_state:
        st.session_state.pm_messages = [
            {
                "role": "system",
                "content": CHATBOT_SYSTEM_PROMPT + f"\n\nMetrics Context:\n{context}"
            }
        ]

    # ---- DISPLAY CHAT HISTORY ----
    for msg in st.session_state.pm_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # ---- USER INPUT ----
    user_input = st.chat_input("Ask a question")

    if user_input:
        # 1. save user message
        st.session_state.pm_messages.append(
            {"role": "user", "content": user_input}
        )

        # 2. display input user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3. placeholder for Chatbot assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                response = ai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.pm_messages,
                    temperature=0.4
                )

                if response and hasattr(response, "choices") and response.choices:
                    answer = str(response.choices[0].message.content)
                else:
                    answer = "[ERROR] No response from AI, please try again."

            except Exception as e:
                answer = f"[ERROR] AI Error: {e}"

            # show answer in placeholder
            message_placeholder.markdown(answer)

        # 4. Save produced message
        st.session_state.pm_messages.append(
            {"role": "assistant", "content": answer}
        )


# -------------------------------------------------------
#                       MAIN APP
# -------------------------------------------------------
def main():
    # Introduction: Title & Description
    st.title("Process Mining AI Dashboard")
    st.write("This dashboard presents process mining results with AI-powered analysis and next event prediction.")
    st.divider()

    try:
        with st.spinner("Loading proess models..."):
            metrics_results, images_paths, df = process_discovery()

        if df is None or df.empty:
            st.error("No data available. Please check the dataset.")
            return

        display_process_mining_dashboard(
            metrics_results=metrics_results,
            images_paths=images_paths,
            df=df
        )

    except Exception as e:
        st.error(f"Error during dashboard display: {e}")

if __name__ == "__main__":
    main()