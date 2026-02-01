# Process Mining AI Dashboard

## Overview

This project is an **AI-powered process mining dashboard** designed to analyze user behavior in a cosmetics e-commerce site, discover process patterns and generate **actionable business insights** through Large Language Models (LLMs).

By combining **event log analysis**, **process mining techniques**, and **AI-driven explanations**, the dashboard enables both technical and non-technical stakeholders to understand how users navigate through a digital product and how the process can be optimized.

The application is built using **Streamlit** for interactive visualization and **PM4Py** for process mining, with AI capabilities integrated via **LLM APIs**.

---

## Key Objectives

- Analyze user sessions from event logs  
- Discover and visualize the most frequent process variants  
- Predict the most likely next user action  
- Explain process behavior using AI  
- Generate strategic, data-driven business recommendations
- Detect anomalies and bottleneck

---

## Core Features

### Process Mining Analysis

- Process discovery with algorithm Alpha Miner, Heuristic Miner and Inductive Miner
- Calculate key quality metrics for different models
- Automatic extraction of **process variants**
- Computation of key behavioral metrics:
  - Average session duration
  - Events per session
  - Most frequent activities
  - Most common exit points

---

### Next Event Prediction

- Interactive selection of user activity sequences
- Prediction of the **most likely next event** based on historical patterns
- AI-generated explanation of:
  - Why the prediction occurs
  - What it means from a process perspective
  - How it impacts business performance

---

### AI-Powered Insights

Three AI-driven reports are generated:

#### 1. AI Prediction Insight

- Explains why a specific user path (e.g. `view → cart`) behaves as observed
- Justifies the predicted next activity
- Provides actionable recommendations to optimize the flow

#### 2. AI Strategic Process Report

- High-level strategic analysis of the entire process
- Based on:
  - Total sessions
  - Average session duration
  - Top process variants
 
#### 3. AI Chatbot
- Analyze the provided process mining quality metrics
- Identifies bottlenecks, UX issues, and optimization opportunities
- Outputs structured recommendations with confidence levels

---

## Architecture Overview

- **Data Layer**: CSV-based event logs 
- **Process Mining Engine**: PM4Py  
- **Frontend**: Streamlit (interactive dashboard)  
- **AI Layer**: LLM-based reasoning and report generation  
- **State Management**: Streamlit session state for user interaction continuity  

---

## How to test the project?

- Install requirements on CMD with "pip install -r requirements"
- Enter you OpenRouter API KEY in openrouter_api_key.txt
- Change LLM Model on script in MODEL_NAME in display_dashboard.py and llm_report.py
- Move on main directory of the project and execute "Streamlit run display_dashboard.py" in CMD 

---

## Why This Project Matters

Traditional process mining tools often stop at visualization.  
This project goes one step further by **explaining the process**, **predicting user behavior**, and **translating insights into business actions** using AI.

It turns process mining from an analytical exercise into a **decision-support system**.
