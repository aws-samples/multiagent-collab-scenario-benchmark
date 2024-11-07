import json
from argparse import ArgumentParser
from tqdm import tqdm
from litellm import completion
from pathlib import Path

from src.prompt_templates import USER_GSR_PROMPT, SYSTEM_GSR_PROMPT
from src import utils

def compute_gsr(response):
    # count the number of true and false
    true_count = [row["answer"].lower() for row in response].count("true")
    false_count = [row["answer"].lower() for row in response].count("false")
    # binary score whether the conversation satisfies all assertions
    gsr = float(false_count == 0)  
    # partial score equal to the percentage of true assertions
    partial_gsr = true_count / (true_count + false_count)
    return gsr, partial_gsr

def evaluate_gsr(conversation, scenario, primary_agent_id, human_id, gsr_type, llm_judge_id):
    assert gsr_type in ["user", "system"], "gsr_type must be 'user' or 'system'"
    primary_string, subagent_string = utils.parse_conversation(conversation, primary_agent_id, human_id)
    clean_assertions = utils.parse_assertions(scenario["assertions"], gsr_type)
    if not clean_assertions:
        # no assertions for this type of GSR
        # provide default assertion to not break code
        clean_assertions = ["User goals are achieved with help from the agent."]

    if gsr_type == "user":
        prompt = USER_GSR_PROMPT.format(
            scenario=scenario, 
            history=primary_string, 
            assertions="\n".join(clean_assertions)
        )

    else:
        prompt = SYSTEM_GSR_PROMPT.format(
            scenario=scenario, 
            history=primary_string, 
            assertions="\n".join(clean_assertions),
            invocations=subagent_string
        )

    raw_response = completion(
        model=llm_judge_id,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    ).choices[0].message.content

    response = utils.parse_llm_judge_response(raw_response)

    gsr, partial_gsr = compute_gsr(response)
    for row in response:
        row["assertion_type"] = gsr_type
    return gsr, partial_gsr, response


def evaluate_conversation(conversation, scenario, primary_agent_id, human_id, llm_judge_id):
    # evaluate user-side GSR
    user_gsr, user_partial_gsr, user_llm_report = evaluate_gsr(
        conversation, scenario, primary_agent_id, human_id, gsr_type="user", llm_judge_id=llm_judge_id  
    )
    # evaluate system-side GSR
    system_gsr, system_partial_gsr, system_llm_report = evaluate_gsr(
        conversation, scenario, primary_agent_id, human_id, gsr_type="system",
        llm_judge_id=llm_judge_id
    )
    # compute overall GSR
    overall_report = user_llm_report + system_llm_report
    overall_gsr, partial_gsr = compute_gsr(overall_report)
    result = {
        "trajectory_index": i,
        "user_gsr": user_gsr,
        "system_gsr": system_gsr,
        "overall_gsr": overall_gsr,
        "partial_gsr": partial_gsr,
        "report": overall_report,
    }
    return result

def save_results(evals, num_scenarios, results_file):
    results = {
        "user_gsr": sum([e["user_gsr"] for e in evals]) / len(evals),
        "system_gsr": sum([e["system_gsr"] for e in evals]) / len(evals),
        "overall_gsr": sum([e["overall_gsr"] for e in evals]) / len(evals),
        "partial_gsr": sum([e["partial_gsr"] for e in evals]) / len(evals),
        "scenario_count": num_scenarios,
        "conversation_count": len(evals),
        "conversation_evals": evals,
    }
    with open(results_file, "w") as fp:
        json.dump(results, fp, indent=4)
    print(f"Results: {json.dumps(results, indent=4)}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default="datasets/travel")
    parser.add_argument("--scenario_filename", type=str, default="scenarios_30.json")
    parser.add_argument("--conversations_dir", type=Path, default="sample_conversations/travel")
    parser.add_argument("--llm_judge_id", type=str, default="gpt-4o")
    args = parser.parse_args()

    # load scenarios and agents 
    with open(args.dataset_dir / args.scenario_filename) as f:
        scenarios = json.load(f)["scenarios"]
    
    with open(args.dataset_dir / "agents.json") as f:
        agents = json.load(f)
    primary_agent_id = agents["primary_agent_id"]
    human_id = agents["human_id"]

    # evaluate conversations
    evals = []
    for i in tqdm(range(len(scenarios))):
        conversation_file = args.conversations_dir / f"conversation_{i}.json"
        if not conversation_file.exists():
            print(f"{conversation_file} does not exist. Skipping...")
            continue
        with open(args.conversations_dir / f"conversation_{i}.json") as f:
            conversation = json.load(f)
        result = evaluate_conversation(conversation, scenarios[i], primary_agent_id, human_id, args.llm_judge_id)
        evals.append(result)

    # save results
    save_results(evals, len(scenarios), args.conversations_dir / "results.json")

