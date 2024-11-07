import os
import json
import re

def parse_conversation(conversation, primary_agent_id, human_id):
    # parse primary agent's trajectory
    primary_traj = conversation["trajectories"][primary_agent_id]
    primary_traj_string = "\n".join(
        [
            "[{} -> {}]: {}".format(
                turn["source"], turn["destination"], turn["content"]
            ) for turn in primary_traj
        ]
    )
    # parse subagents trajectory
    subagent_traj_strings = []
    for subagent_id, subagent_traj in conversation["trajectories"].items():
        if subagent_id == primary_agent_id or subagent_id == human_id:
            continue
        def parse_subagent_turn(row):
            subagent_string = ""
            if row["role"] == "Action":
                for action in row['actions']:
                    action_dict = action['parameters']['mock_fn_input']
                    subagent_string += f"[{row['source']} {row['role']}]: {action_dict}\n"
            if row["role"] == "Observation":
                subagent_string = f"[{row['source']} {row['role']}]: {row['observation']}\n"
            return subagent_string.rstrip("\n")

        subagent_traj_strings += [parse_subagent_turn(row) for row in subagent_traj]
    subagents_traj_string = "\n".join(subagent_traj_strings)
    return primary_traj_string, subagents_traj_string

def parse_assertions(assertions, gsr_type):
    """
    Assertions may have `User:` or `Agent:` in front of them to
    distinguish between user and agent assertions.

    Removes the prefix and returns type of assertions.
    """
    clean_assertions = []
    user_prefix = "user:"
    system_prefix = "agent:"
    for assertion in assertions:
        assertion = assertion.lstrip()
        if gsr_type == "user" and assertion.lower().startswith(user_prefix):
            clean_assertion = assertion[5:].strip()
            clean_assertions.append(clean_assertion)
        elif gsr_type == "system" and assertion.lower().startswith(system_prefix):
            clean_assertion = assertion[6:].strip()
            clean_assertions.append(clean_assertion)
        elif gsr_type == "user" and not assertion.lower().startswith(system_prefix):
            # assume that no prefix means user assertion
            clean_assertions.append(assertion)
    return clean_assertions

def parse_llm_judge_response(raw_response):
    if type(raw_response) == str:
        # Remove the first line if the first line begins with "Here"
        if raw_response.startswith("Here"):
            raw_response = "\n".join(raw_response.split("\n")[1:])

        substrings_to_remove = ["\n", "```json", "```" ]
        pattern = "|".join(substrings_to_remove)
        parsed_response = re.sub(pattern, "", raw_response)
        dict_list_response = json.loads(parsed_response)
    else:
        dict_list_response = raw_response
    # handle wrapper keys
    for k in ["assertions", "results", "duplicate_or_irrelevant_messages"]:
        if k in dict_list_response:
            dict_list_response = dict_list_response[k]
            break
    if type(dict_list_response) == dict:
        # wrap with list (happens for single-item responses) 
        dict_list_response = [dict_list_response]
    # concatenate any items that are lists
    for row in dict_list_response:
        for key, value in row.items():
            if type(value) == list:
                row[key] = " ".join(value) 
            if type(value) != str:
                row[key] = str(value)
    return dict_list_response
