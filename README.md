## Meta-agent LLM Scenario Benchmarking

### Data

Benchmarking data is in the `datasets` directory where there are 30 hypothetical scenarios for three domains: travel planning, mortgage financing, and software development. 

Each entry in the scenarios file contains:
- `scenario`: The user background and goals.
- `input_problem`: A description of the problem to be solved by the agent.
- `assertions`: A list of assertions that must be true to judge the interaction between user and the agent. 

In each dataset, there is also a `agents.json` file that contains the agent's name and description, as well as their corresponding tools. The scenarios are collected based on these agent profiles and tool schemas.

### Pre-requisites

Create a Python 3.12 virtual environment and install requirements in `requirements.txt`.

Next, prepare the conversations that you want to benchmark. Each conversation should be in its own JSON file titled `conversation_0.json`, `conversation_1.json`, etc. where the index corresponds to the scenario index. The `conversation_{i}.json` file should be formatted as follows:

```
{
    "trajectories": {
        "agent_id_1": [
            {
                "role": null, # null, User, Action, Observation
                "source": "", # agent_id of the agent who sent this message
                "destination": "", # agent_id of the user who received this message
                "content": "", # content of the message
                "actions": [], # list of action objects executed by the agent
                "observation": null, # observation of the agent
            }
        ],
        "agent_id_2": [...],
        ...
    }
}
```

See `sample_conversations` for examples.


### How to use 

First, export any environment variables needed for LLM providers (Bedrock, OpenAI, Anthropic, etc) to support the LLM judge. See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for setting up LLMs.

Run the benchmarking script on a sample travel conversation:

```
{export env variables}

python -m src.benchmark
```

Customize the benchmarking parameters as needed:
```
python -m src.benchmark \ 
    --dataset_dir <path_to_dataset>  \
    --scenario_filename <filename of scenarios> \
    --conversations_dir <path_to_conversations> \
    --llm_judge_id <LiteLLM llm_judge_id> \
```



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

The dataset is licensed under the CC-BY-4.0 license.
