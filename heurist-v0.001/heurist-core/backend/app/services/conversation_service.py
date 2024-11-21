# app/services/conversation_service.py

from app.services.watsonx_service import query_watsonx

conversation_state = {
    "step": 0,
    "data": {}
}

def initiate_conversation() -> str:
    """Starts the conversation with the user."""
    conversation_state["step"] = 1
    return "What is the main decision you want to analyze? (e.g., 'Increase the marketing budget by 20%')"

def collect_input(user_input: dict) -> str:
    """Collects information from the user based on the current state."""
    step = conversation_state["step"]
    
    if step == 1:
        # Stores the main decision
        conversation_state["data"]["decision"] = user_input.get("decision", "")
        conversation_state["step"] = 2
        return "What are the relevant variables for this decision? (e.g., 'budget: 50000, timeline: 6 months, target_market: Young adults')"
    
    elif step == 2:
        # Stores the variables
        conversation_state["data"]["variables"] = user_input.get("variables", {})
        conversation_state["step"] = 3
        return "Great! We will generate scenarios based on this data. Please confirm if we can proceed."

    elif step == 3:
        # Finalizes the collection and starts the simulation
        return "Data collected. Please send a request to the '/simulate/' endpoint to generate the scenarios."

    else:
        return "Conversation already completed. Restart if necessary."

def simulate_decision(input_data: dict) -> list:
    """Simulates scenarios using Watsonx."""
    decision = conversation_state["data"].get("decision", "")
    variables = conversation_state["data"].get("variables", {})
    response = query_watsonx(decision, variables)
    return response
