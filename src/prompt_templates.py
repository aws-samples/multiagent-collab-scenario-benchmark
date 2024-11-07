USER_GSR_PROMPT = """Determine whether the conversation between the user and agent satisfies a list of assertions. 
Pay attention to dates, time, location, and other relevant information about the user.
The judgement should be based on the given user scenario and the conversation history.
The user scenario provides the background information of the conversation.
The conversation history shows the interaction between user and agent.
 
Scenario:
{scenario}
 
Conversation History:
{history}
 
Assertions:
{assertions}
 
Answer TRUE or FALSE for each assertion. Provide answers in JSON array format with keys "assertion", "answer", and "evidence".
Please address every assertion. 
"""

SYSTEM_GSR_PROMPT = """Determine whether the conversation between the user and agent satisfies a list of assertions. 
Pay attention to dates, time, location, and other relevant information about the user.
The judgement should be based on the given user scenario, the conversation history, and the tool invocations.
The user scenario provides the background information of the conversation.
The conversation history shows the interaction between user and agent.
The tool invocations shows tool actions and observations from the agents during the conversation.
 
Scenario:
{scenario}
 
Conversation History:
{history}

Tool Invocations:
{invocations}
 
Assertions:
{assertions}
 
Answer TRUE or FALSE for each assertion. Provide answers in JSON array format with keys "assertion", "answer", and "evidence".
Please address every assertion.
"""

ISSUES_PROMPT = SYSTEM_GSR_PROMPT + "\nThis conversation has been judged as {judgement} which is either caused by the user, primary agent, sub-agents, or tool failures. If the conversation has failed, please take this into account when determining reliability of agents and tools."

