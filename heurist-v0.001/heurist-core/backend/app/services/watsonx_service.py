# app/services/watsonx_service.py

from langchain_ibm import ChatWatsonx
from app.config import Config
from app.database import save_universe, get_universe_by_title
import json
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Watsonx client
chat = ChatWatsonx(
    model_id="meta-llama/llama-3-70b-instruct",
    url=Config.WATSONX_URL,
    project_id=Config.WATSONX_PROJECT_ID,
    params={"temperature": 0.7, "max_tokens": 2000},
)

# Temporary conversation state
conversation_state = {"decision": None, "variables": {}, "context": None}

def initiate_conversation():
    """Starts a new conversation and collects the initial decision."""
    return {
        "message": "Welcome! What decision would you like to explore? Please describe your initial decision."
    }

def validate_input(parameter: str, parameter_name: str):
    """
    Validates if the provided parameter is not empty or null.
    """
    if not parameter or not parameter.strip():
        raise ValueError(f"The parameter '{parameter_name}' cannot be empty.")

def detect_context(decision_text: str) -> str:
    """
    Detects the decision context based on the provided text.
    """
    try:
        messages = [
            ("system", "You are an assistant that analyzes the type of user decision."),
            ("human", f"Based on the text '{decision_text}', identify if it is a personal, business, educational, health, or other type of decision.")
        ]
        response = chat.invoke(messages)
        context = response.content.strip().lower()
        logger.info(f"Detected context: {context}")
        return context if context in ["personal", "business", "educational", "health", "travel"] else "other"
    except Exception as e:
        logger.error(f"Error detecting context: {str(e)}")
        return "other"  # Fallback to avoid failures

def collect_variables(context: str) -> dict:
    """
    Returns relevant variables based on the context.
    """
    contexts = {
        "business": {
            "budget": "What is the available budget?",
            "timeline": "What is the estimated timeline (in days)?",
            "target_audience": "Who is the target audience for this decision?",
            "resources": "What human or material resources are available?",
            "risks": "What are the possible risks?",
            "expected_roi": "What is the expected return on investment (ROI)?",
        },
        "personal": {
            "personal_budget": "How much can you spend?",
            "available_time": "How much time do you have available?",
            "priorities": "What are your personal priorities?",
            "location": "Where do you intend to implement this decision?",
            "emotional_impact": "What would be the emotional impact of this decision?",
            "long_term_consequences": "What are the expected long-term consequences?",
        },
        "educational": {
            "goal": "What is the goal of the course or learning?",
            "cost": "What is the estimated cost?",
            "duration": "How much time can you dedicate?",
            "flexibility": "How flexible are the schedules?",
            "format": "Do you prefer in-person or online?",
            "career_impact": "What is the expected impact on your career?",
        },
        "health": {
            "current_condition": "What is your current health condition?",
            "cost": "What is the estimated cost of treatment?",
            "duration": "How much time can you dedicate to care?",
            "impact_on_life": "How will this affect your quality of life?",
            "risks": "Are there associated risks?",
        },
        "travel": {
            "budget": "What is the travel budget?",
            "destination": "What is the intended destination?",
            "duration": "How many days do you plan to travel?",
            "purpose": "What is the purpose of the travel (leisure, business, culture)?",
            "risks": "Are there risks, such as weather conditions or restrictions?",
        },
    }
    variables = contexts.get(context, {"general_details": "Please provide more details about your decision."})
    logger.info(f"Variables for context '{context}': {variables}")
    return variables

def collect_input(user_input: dict):
    """
    Collects user input and updates the conversation state.
    """
    decision_text = user_input.get("decision", "")
    details = user_input.get("details", {})

    # Reset state if a new decision is provided
    if decision_text and decision_text != conversation_state.get("decision"):
        conversation_state["decision"] = decision_text
        conversation_state["variables"] = {}
        context = detect_context(decision_text)
        conversation_state["context"] = context

        variables_questions = collect_variables(context)
        return {
            "message": "Understood. Please provide the following information:",
            "questions": variables_questions,
        }

    # Collect variables if not already defined
    if not conversation_state["variables"]:
        if not details:
            return {"message": "Please enter the requested variables."}

        conversation_state["variables"] = details
        return {
            "message": "Thank you! Data complete. Ready to start the simulation.",
            "state": conversation_state,
        }

    # If data is already collected
    return {"message": "All data has been collected. Ready to simulate."}

def simulate_decision(num_scenarios=9):
    """
    Simulates multiple scenarios based on the collected data.
    """
    decision = conversation_state.get("decision")
    variables = conversation_state.get("variables")
    context = conversation_state.get("context", "other")

    if not decision or not variables:
        return {"error": "Insufficient data for simulation. Ensure the decision and variables are provided."}

    try:
        variable_text = "\n".join([f"{key}: {value}" for key, value in variables.items()])
        prompt = (
            f"Based on the decision '{decision}' and the following variables:\n{variable_text}\n"
            f"Create {num_scenarios} unique scenarios in the context '{context}'. For each scenario:\n"
            "- Start with the scenario title.\n"
            "- Present a detailed and complete narrative.\n"
            "- Highlight specific and realistic benefits (prefix: 'Benefits:').\n"
            "- Describe clear challenges (prefix: 'Challenges:').\n"
            "- Include practical recommendations (prefix: 'Recommendations:').\n"
            "Clearly separate each scenario with the word 'SCENARIO' followed by a number."
        )

        messages = [
            ("system", "You are an assistant that creates detailed scenarios based on decisions."),
            ("human", prompt)
        ]

        response = chat.invoke(messages)
        raw_response = response.content.strip()

        scenarios_raw = raw_response.split("\nSCENARIO ")
        scenarios = []

        for i, scenario_text in enumerate(scenarios_raw):
            if i == 0:
                continue  # Skip introduction
            scenario_parts = scenario_text.split("\n", 1)
            scenario_title = scenario_parts[0].strip()
            scenario_details = scenario_parts[1].strip() if len(scenario_parts) > 1 else ""

            scenarios.append({
                "universe": f"Universe {i}",
                "title": scenario_title,
                "narrative": scenario_details
            })

        # Save to database
        universe_data = {
            "decision": decision,
            "context": context,
            "variables": variables,
            "scenarios": scenarios,
        }
        save_universe(universe_data)

        return {
            "message": "Simulation completed successfully!",
            "scenarios": scenarios
        }

    except Exception as e:
        logger.error(f"Error simulating scenarios: {str(e)}")
        return {"error": f"Error simulating scenarios: {str(e)}"}
