# csv_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
import json
from io import StringIO
from logger import get_logger

logger = get_logger(__name__)

OLLAMA_ENDPOINT = "http://localhost:11434"
OLLAMA_TIMEOUT = 180  # seconds

def ask_csv_question(df: pd.DataFrame, prompt: str, model: str = "mistral") -> str:
    """
    Sends a prompt and dataframe to the LLM to get chart-generating Python code.

    Args:
        df (pd.DataFrame): The dataframe from uploaded CSV.
        prompt (str): User's chart-related question.
        model (str): Model name (default: "mistral").

    Returns:
        str: Python code to generate the chart.
    """
    try:
        sample = df.head(100).to_csv(index=False)
        full_prompt = (
            f"Given the following dataframe (first 100 rows):\n{sample}\n\n"
            f"Generate a Python (matplotlib or pandas) chart based on this user request: {prompt}"
        )

        response = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()["response"]
        return parse_generated_code(result)

    except Exception as e:
        logger.error(f"CSV chart error: {e}")
        return f"# Error: {e}"

def parse_generated_code(response: str) -> str:
    """
    Extracts code from the LLM response. Strips code block formatting.

    Args:
        response (str): Raw response from LLM.

    Returns:
        str: Cleaned Python code.
    """
    try:
        code = re.sub(r"^```(?:python)?|```$", "", response.strip(), flags=re.MULTILINE)
        return code
    except Exception as e:
        logger.error(f"Code parsing error: {e}")
        return f"# Failed to parse code: {e}"

def run_chart_code(code: str, df: pd.DataFrame):
    """
    Executes the generated code using matplotlib and pandas, returns the plot figure.

    Args:
        code (str): Code string to execute.
        df (pd.DataFrame): DataFrame to be passed as context.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    try:
        fig = plt.figure()
        exec_globals = {"df": df, "pd": pd, "plt": plt, "fig": fig}
        exec(code, exec_globals)
        return fig
    except Exception as e:
        logger.error(f"Chart execution error: {e}")
        raise RuntimeError(f"Chart execution failed: {e}")
