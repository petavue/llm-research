import os
import boto3
import json
import asyncio
from datetime import datetime, timezone
import pathlib
from common_functions import (
    get_datasets_info,
    get_instruction_shot_specific_prompt,
    initialize_files,
    generate_gold_file,
    write_to_file,
    log,
    get_elapsed_time,
    sql_match,
    get_parsed_args,
    multi_process_setup,
)
from typing import Any, Tuple, List
from common_constants import Defaults, Environments, BedrockModels

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.AMZ_BEDROCK

# Get environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

supported_models = {
    "claude2": BedrockModels.MODEL_ANTHROPIC_CLAUDE,
    "llama": BedrockModels.MODEL_META_LLAMA,
    "claude3-sonnet":BedrockModels.MODEL_ANTHROPIC_CLAUDE_3_SONNET,
    "claude3-haiku":BedrockModels.MODEL_ANTHROPIC_CLAUDE_3_HAIKU,
    "mistral":BedrockModels.MODEL_ANTHROPIC_MISTRAL_7B,
    "mixtral":BedrockModels.MODEL_ANTHROPIC_MIXTRAL
}


def initialize_amz_bedrock() -> Any:
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def run_queries_on_bedrock(
    total_user_query: Tuple[str, str, str],
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    bedrock_runtime_client: Any,
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

            prompt = system_prompt.replace("[context]", context).replace(
                    "[question]", "Question: "+question
                ).replace("[hint]",str(evidence))
            
            if model_name in [ BedrockModels.MODEL_ANTHROPIC_CLAUDE_3_SONNET, BedrockModels.MODEL_ANTHROPIC_CLAUDE_3_HAIKU]:
                
                body = {
                    "anthropic_version": Defaults.anthropic_version_const,
                    "max_tokens": Defaults.MAX_TOKENS_TO_GENERATE,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }

            elif model_name == BedrockModels.MODEL_ANTHROPIC_CLAUDE:
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                body = {"prompt": prompt}
                body["max_tokens_to_sample"] = Defaults.MAX_TOKENS_TO_GENERATE
            elif model_name == BedrockModels.MODEL_ANTHROPIC_MISTRAL_7B:
                prompt = f"<s>[INST] {prompt}\n\n[/INST]"
                body = {"prompt": prompt}
                body["max_tokens"] = Defaults.MAX_TOKENS_TO_GENERATE
            elif model_name == BedrockModels.MODEL_ANTHROPIC_MIXTRAL:
                prompt = f"<s>[INST]  {prompt}\n\n[/INST]"
                body = {"prompt": prompt}
                body["max_tokens"] = Defaults.MAX_TOKENS_TO_GENERATE

            else:
                body = {"prompt": prompt}

            data_to_log["request"] = body

            response_time_start = datetime.now(timezone.utc)
            bedrock_response = bedrock_runtime_client.invoke_model(
                modelId=model_name, body=json.dumps(body)
            )
            response_time_stop = datetime.now(timezone.utc)
            response_body = json.loads(bedrock_response["body"].read())

            # remove 'body' entry from bedrock_response due to 'StreamingBody is not JSON serializable' exception while logging
            data_to_log["response"] = {
                key: bedrock_response[key] for key in bedrock_response if key != "body"
            }
            data_to_log["response_body"] = response_body

            bedrock_response_headers = bedrock_response["ResponseMetadata"][
                "HTTPHeaders"
            ]
            llm_response_tokens = bedrock_response_headers[
                "x-amzn-bedrock-output-token-count"
            ]
            llm_prompt_tokens = bedrock_response_headers[
                "x-amzn-bedrock-input-token-count"
            ]

            if model_name == BedrockModels.MODEL_ANTHROPIC_CLAUDE:
                llm_response_content = response_body["completion"]
            elif model_name == BedrockModels.MODEL_ANTHROPIC_MISTRAL_7B or model_name == BedrockModels.MODEL_ANTHROPIC_MIXTRAL:
                llm_response_content = response_body["outputs"][0]['text']
            elif model_name == BedrockModels.MODEL_ANTHROPIC_CLAUDE_3_SONNET or model_name ==BedrockModels.MODEL_ANTHROPIC_CLAUDE_3_HAIKU:
                llm_response_content = response_body['content'][0]['text']
            elif model_name==BedrockModels.MODEL_META_LLAMA:
                llm_response_content = response_body["generation"]

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
    bedrock_runtime_client = initialize_amz_bedrock()

    for shot_size in shot_size_list:
        if "cot" in shot_size:
            file_shot_size = shot_size.split("-")[0] + "_shot_cot"
        else:
            file_shot_size = shot_size + "_shot"

        for dataset_length, query_list, gold_file_list in datasets_info:
            model_file_path = f"{target_dir}/{HOST_ENV}/{file_shot_size}/{model_name}/{instruction_size}_Instructions/{dataset_length}_Inferences"

            print(f"{model_file_path}/execution-log.jsonl")
            if os.path.exists(model_file_path) and os.path.isfile(f"{model_file_path}/execution-log.jsonl"):
                count = 0
                with open(f"{model_file_path}/execution-log.jsonl", 'r') as file:
                    for _ in file:
                        count += 1
                
                if count == dataset_length:
                    continue
                else:
                    log_file_path = f"{model_file_path}/execution-log.jsonl"

                    output_file_path = f"{model_file_path}/predicted.txt"
                    metrics_file_path = f"{model_file_path}/metrics.csv"
                    print(f"Starting loop for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences - resuming from {count}")
                    loop_start_time = datetime.now()
                    run_queries_on_bedrock(
                        query_list[count:],
                        output_file_path,
                        metrics_file_path,
                        log_file_path,
                        model_name,
                        bedrock_runtime_client,
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
                print(
                    f"Starting loop for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences"
                )
                output_file_path, metrics_file_path, log_file_path = initialize_files(
                    model_file_path
                )
                loop_start_time = datetime.now()
                run_queries_on_bedrock(
                    query_list,
                    output_file_path,
                    metrics_file_path,
                    log_file_path,
                    model_name,
                    bedrock_runtime_client,
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
