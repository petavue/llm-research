from datetime import datetime, timezone
import os
import pathlib
from openai import AsyncOpenAI
import asyncio
from anthropic import Anthropic


from common_functions import (
    get_datasets_info,
    initialize_files,
    generate_gold_file,
    write_to_file,
    log,
    get_elapsed_time,
    sql_match,
    get_parsed_args,
    get_instruction_shot_specific_prompt,
    multi_process_setup,
)
from typing import Tuple, List, Any
from common_constants import Defaults, Environments, AnthropicModels

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.ANTHROPIC

# Get environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


supported_models = {
    "cl3-op": AnthropicModels.MODEL_ANTHROPIC_OPUS,
    "cl3-son": AnthropicModels.MODEL_ANTHROPIC_SONNET,
    "cl3-hai": AnthropicModels.MODEL_ANTHROPIC_HAIKU
}


async def run_queries_on_anthropic(
    total_user_query: Tuple[str, str, str],
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    client: Any,
    instruction_size: int,
    dataset_length: int,
    shot_size: str,
) -> None:
    try:
        for context, question, hardness, db_id, evidence in total_user_query:
            data_to_log = {
                "environment": HOST_ENV,
                "model": model_name,
                "instruction_size": instruction_size,
                "dataset_length": dataset_length,
                "severity": "info",
                "is_sql": 0,
            }

            system_prompt = get_instruction_shot_specific_prompt(
                instruction_size, shot_size, db_id
            )

            req = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.replace("[context]", context).replace(
                        "[question]", f"Question: {question}"
                    ).replace("[hint]",str(evidence))
                }
            ]
        },
    ]
            data_to_log["request"] = req
            response_time_start = datetime.now(timezone.utc)
            anthropic_response = client.messages.create(
                model=model_name,
                messages=req,
                max_tokens=Defaults.MAX_TOKENS_TO_GENERATE,
            )
            response_time_stop = datetime.now(timezone.utc)
            data_to_log["response"] = str(anthropic_response)

            llm_response_content = anthropic_response.content[0].text
            llm_response_tokens = anthropic_response.usage.output_tokens
            llm_prompt_tokens = anthropic_response.usage.input_tokens

            if "select" in llm_response_content.lower():
                is_sql_match, sql_response = sql_match(llm_response_content)
            else:
                is_sql_match = False

            if not is_sql_match:
                data_to_log["severity"] = "warn"
                log("No SQL Output detected", data_to_log, log_file_path)
                write_to_file(
                    output_file_path,
                    metrics_file_path,
                    "I don't know\n\n",
                    f"{0},{llm_prompt_tokens},{llm_response_tokens},{hardness}\n",
                )
                continue
            

            data_to_log["response_time_start"] = response_time_start.strftime('%Y-%m-%d %H:%M:%S')
            data_to_log["response_time_stop"] = response_time_stop.strftime('%Y-%m-%d %H:%M:%S')
            data_to_log["is_sql"] = 1
            data_to_log["sql_response"] = sql_response
            log("SQL Response successful", data_to_log, log_file_path)

            response_time = response_time_stop - response_time_start
            write_to_file(
                output_file_path,
                metrics_file_path,
                f"{sql_response}\n\n",
                f"{response_time},{llm_prompt_tokens},{llm_response_tokens},{hardness}\n",
            )
    except Exception as ex:
        data_to_log["severity"] = "error"
        log(ex, data_to_log, log_file_path)
        print("exception: ", ex)
        write_to_file(
            output_file_path,
            metrics_file_path,
            f"An error occurred: {ex}\n\n",
            f"{0},{0},{0},{hardness}\n",
        )


async def multi_process(
    instruction_size: int,
    datasets_info: list,
    model_name: str,
    shot_size_list: List[str],
    target_dir: str,
) -> None:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)


    for shot_size in shot_size_list:
        if "cot" in shot_size:
            file_shot_size = shot_size.split("-")[0] + "_shot_cot"
        else:
            file_shot_size = shot_size + "_shot"

        for dataset_length, query_list, gold_file_list in datasets_info:
            model_file_path = f"{target_dir}/{HOST_ENV}/{file_shot_size}/{model_name}/{instruction_size}_Instructions/{dataset_length}_Inferences"
            if os.path.exists(model_file_path) and os.path.isfile(f"{model_file_path}/execution-log.jsonl"):
                count = 0
                with open(f"{model_file_path}/execution-log.jsonl", 'r') as file:
                    for _ in file:
                        count += 1
                print(count)
                
                if count == dataset_length:
                    continue
                else:
                    log_file_path = f"{model_file_path}/execution-log.jsonl"

                    output_file_path = f"{model_file_path}/predicted.txt"
                    metrics_file_path = f"{model_file_path}/metrics.csv"
                    print(f"Starting loop for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences - resuming from {count}")
                    loop_start_time = datetime.now()
                    loop_start_time = datetime.now()
                    await run_queries_on_anthropic(
                        query_list[count:],
                        output_file_path,
                        metrics_file_path,
                        log_file_path,
                        model_name,
                        client,
                        instruction_size,
                        dataset_length,
                        file_shot_size,
                    )
                    generate_gold_file(gold_file_list, model_file_path,dataset_length)
                    loop_end_time = datetime.now()
                    total_secs = (loop_end_time - loop_start_time).total_seconds()
                    print(
                        f"Time taken for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences: {get_elapsed_time(total_secs)}"
                    )
            else:

                output_file_path, metrics_file_path, log_file_path = initialize_files(
                    model_file_path
                )

                print(
                    f"Starting loop for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences"
                )
                loop_start_time = datetime.now()
                await run_queries_on_anthropic(
                    query_list,
                    output_file_path,
                    metrics_file_path,
                    log_file_path,
                    model_name,
                    client,
                    instruction_size,
                    dataset_length,
                    file_shot_size,
                )
                generate_gold_file(gold_file_list, model_file_path,dataset_length)
                loop_end_time = datetime.now()
                total_secs = (loop_end_time - loop_start_time).total_seconds()
                print(
                    f"Time taken for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences: {get_elapsed_time(total_secs)}"
                )


async def main() -> None:
    args, model_instructions = get_parsed_args(supported_models, HOST_ENV)
    inference_length_in_args = [int(inst) for inst in args.inf_length.split(",")]
    dataset_length_list = inference_length_in_args or Defaults.INFERENCE_LENGTH_LIST

    datasets_info = get_datasets_info(dataset_length_list)

    for model_name_from_args in args.models.split(","):
        if model_instructions:
            instruction_size_list = model_instructions[model_name_from_args]
        elif args.inst:
            instruction_size_list = [int(inst) for inst in args.inst.split(",")]
        else:
            instruction_size_list = Defaults.INSTRUCTION_SIZE_LIST

        model_name = supported_models[model_name_from_args]

        await multi_process_setup(
            multi_process,
            instruction_size_list,
            datasets_info,
            model_name,
            args.shot_size.split(","),
            args.target_dir,
        )


if __name__ == "__main__":
    asyncio.run(main())


