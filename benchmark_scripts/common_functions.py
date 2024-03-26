import pandas as pd
import os
import math
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re
import argparse
from typing import Any, List, Tuple, Dict, Callable
import json
from ast import literal_eval
import pathlib
from common_constants import SelfHostedModels, Environments

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()


def async_wrapper(
    multi_process: Callable,
    instruction_size: int,
    datasets_info: list,
    model_name: str,
    shot_size_list,
    target_dir: str,
) -> Any:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        multi_process(
            instruction_size,
            datasets_info,
            model_name,
            shot_size_list,
            target_dir,
        )
    )
    return result


async def multi_process_setup(
    multi_process: Any,
    instruction_size_list: list,
    datasets_info: list,
    model_name: str,
    shot_size_list: List[str],
    target_dir: str,
) -> None:
    loop = asyncio.get_running_loop()
    tasks = []

    with ProcessPoolExecutor() as executor:
        for instruction_size in instruction_size_list:
            partial_func = partial(
                async_wrapper,
                multi_process,
                instruction_size,
                datasets_info,
                model_name,
                shot_size_list,
                target_dir,
            )
            tasks.append(loop.run_in_executor(executor, partial_func))
        for done in asyncio.as_completed(tasks):
            await done


def get_datasets_info(
    dataset_length_list: List[int],
) -> Tuple[List, List]:
    datasets_info = []
    for dataset_length in dataset_length_list:
        output_file_path = (
            f"../spider_data/bird_equal_split_{str(dataset_length)}.csv"
        )
        spider_data_frame = pd.read_csv(output_file_path)

        query_list = list(
            zip(
                spider_data_frame.schema,
                spider_data_frame.question,
                spider_data_frame.difficulty,
                spider_data_frame.db_id,
                spider_data_frame.evidence
            )
        )
        gold_file_list = (spider_data_frame["sql_query"], spider_data_frame["db_id"])

        datasets_info.append((dataset_length, query_list, gold_file_list))

    return datasets_info


def get_shot_samples_data(file_shot_size: str) -> List:
    with open(f"../spider_data/spider_{file_shot_size}_samples.txt") as samples_file:
        contents = samples_file.read()
        few_shot_samples = literal_eval(contents)

    return few_shot_samples


def get_few_shot_sample_string(shot_size: str, db_id: str, prompt: str) -> str:
    if shot_size == "0_shot":
        return prompt.replace("[examples]", "")

    few_shot_sample_dict = get_shot_samples_data(shot_size)
    samples_list = few_shot_sample_dict[db_id]

    if "cot" in shot_size:
        parsed_str = list(
            map(
                lambda sample_dict: f"\n    Q: {sample_dict['question']}\n    A: {sample_dict['sql_query']}\n    Explanation: \n{sample_dict['cot']}\n",
                samples_list,
            )
        )
    else:
        parsed_str = list(
            map(
                lambda sample_dict: f"\n    Q: {sample_dict['question']}\n    A: {sample_dict['sql_query']}",
                samples_list,
            )
        )
    samples_prompt = prompt.replace("[examples]", "".join(parsed_str))
    return samples_prompt

def get_instruction_shot_specific_prompt(
    instruction_size: int, shot_size: str, db_id: str
) -> str:
    instructions_prompt = initialize_system_prompt(instruction_size)
    return get_few_shot_sample_string(shot_size, db_id, instructions_prompt)

def generate_model_specific_prompt_for_self_hosted_model(
    model_name: str, system_prompt: str, context: str, question: str,evidence:str
) -> str:
    if model_name == SelfHostedModels.MODEL_META_CODELLAMA_70B:
        system_prompt = system_prompt.replace("[context]", context).replace(
            "[question]", ""
        ).replace("[hint]",str(evidence))
        prompt = f"<s>Source: system\n\n {system_prompt} <step> Source: user\n\n {question} <step> Source: assistant\nDestination: user\n\n "
    elif model_name == SelfHostedModels.MODEL_WIZARDLM_WIZARD_CODER_33B:
        system_prompt = system_prompt.replace("[context]", context).replace(
            "[question]", ""
        ).replace("[hint]",str(evidence))
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{system_prompt} QUESTION: {question} \n\n### Response:"
    elif model_name in [
        SelfHostedModels.MODEL_MISTRALAI_MISTRAL_7B,
        SelfHostedModels.MODEL_MISTRALAI_MIXTRAL_8X7B,
        SelfHostedModels.MODEL_MISTRALAI_MISTRAL_7B_V2,
    ]:
        system_prompt = system_prompt.replace("[context]", context).replace(
            "[question]", question
        ).replace("[hint]",str(evidence))
        prompt = f"<s> [INST] {system_prompt} [/INST]"
    elif model_name in [
        SelfHostedModels.MODEL_DEFOG_SQLCODER_70B,
        SelfHostedModels.MODEL_DEFOG_SQLCODER_7B_2,
    ]:
        system_prompt = system_prompt.replace("[context]", context).replace(
            "[question]", ""
        ).replace("[hint]",str(evidence))
        prompt = f"### Task \nGenerate a SQL query to answer [QUESTION]{question}[/QUESTION]### Database Schema \nThe query will run on a database with the following schema:{system_prompt} ### Answer \nGiven the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION] \n[SQL]"
    else:
        prompt = system_prompt.replace("[context]", context).replace(
            "[question]", question
        ).replace("[hint]",str(evidence))

    return prompt


def initialize_system_prompt(instruction_size: int) -> str:
    INSTRUCTIONS_5 = """
    Adhere to the following Instuction:
    1. The answer generated must only be an SQL query ending with delimiter “;”
    2. Dedicate time to understand the database schema fully, identifying the relevant tables and columns that align with the query’s objectives.
    3. Utilize only the data from the specified tables in the provided database schema.
    4. Pay attention to case sensitivity in data and ensure the extraction of required information aligns precisely with the specified columns and tables.
    5. Analyze the query’s requirements to determine the appropriate use of GROUP BY, HAVING, and UNION clauses, ensuring they contribute to the accurate aggregation and segmentation of data.
    """
    INSTRUCTIONS_7 = """
    Adhere to the following Instuction:
    1. The answer generated must only be an SQL query ending with delimiter “;”
    2. Dedicate time to understand the database schema fully, identifying the relevant tables and columns that align with the query’s objectives.
    3. Utilize only the data from the specified tables in the provided database schema.
    4. Pay attention to case sensitivity in data and ensure the extraction of required information aligns precisely with the specified columns and tables.
    5. Analyze the query’s requirements to determine the appropriate use of GROUP BY, HAVING, and UNION clauses, ensuring they contribute to the accurate aggregation and segmentation of data.
    6. Pay careful attention to the primary keys, foreign keys present in the database schema to determine appropriate columns for JOIN operations.
    7. Apply WHERE clause conditions accurately to filter the dataset based on specified criteria.
    """
    INSTRUCTIONS_9 = """
    Adhere to the following Instuction:
    1. The answer generated must only be an SQL query ending with delimiter “;”
    2. Dedicate time to understand the database schema fully, identifying the relevant tables and columns that align with the query’s objectives.
    3. Utilize only the data from the specified tables in the provided database schema.
    4. Pay attention to case sensitivity in data and ensure the extraction of required information aligns precisely with the specified columns and tables.
    5. Analyze the query’s requirements to determine the appropriate use of GROUP BY, HAVING, and UNION clauses, ensuring they contribute to the accurate aggregation and segmentation of data.
    6. Pay careful attention to the primary keys, foreign keys present in the database schema to determine appropriate columns for JOIN operations.
    7. Apply WHERE clause conditions accurately to filter the dataset based on specified criteria.
    8. Apply WHERE clause conditions accurately to filter the dataset and use ASC or DESC in sorting results where specified.
    9. Assign meaningful aliases to tables and columns where necessary, especially in cases of grouping or joining, to enhance the clarity and maintainability of the SQL query.
    """
    INSTRUCTIONS_11 = """
    Adhere to the following Instuction:
    1. The answer generated must only be an SQL query ending with delimiter “;”
    2. Dedicate time to understand the database schema fully, identifying the relevant tables and columns that align with the query’s objectives.
    3. Utilize only the data from the specified tables in the provided database schema.
    4. Pay attention to case sensitivity in data and ensure the extraction of required information aligns precisely with the specified columns and tables.
    5. Analyze the query’s requirements to determine the appropriate use of GROUP BY, HAVING, and UNION clauses, ensuring they contribute to the accurate aggregation and segmentation of data.
    6. Pay careful attention to the primary keys, foreign keys present in the database schema to determine appropriate columns for JOIN operations.
    7. Apply WHERE clause conditions accurately to filter the dataset based on specified criteria.
    8. Apply WHERE clause conditions accurately to filter the dataset and use ASC or DESC in sorting results where specified.
    9. Assign meaningful aliases to tables and columns where necessary, especially in cases of grouping or joining, to enhance the clarity and maintainability of the SQL query.
    10. When multiple tables are involved, prioritize selecting appropriate columns based on context and handle null values properly in columns.
    11. Avoid any write operations (like modify, update, delete, or drop). Should the task demand such actions, respond with a polite refusal, stating, “I’m sorry, but I can’t assist with that.”
    """

    extra_instruction = []
    if instruction_size == 0:
        extra_instruction.append("")
    if instruction_size == 5:
        extra_instruction.append(INSTRUCTIONS_5)
    if instruction_size == 7:
        extra_instruction.append(INSTRUCTIONS_7)
    if instruction_size == 9:
        extra_instruction.append(INSTRUCTIONS_9)
    if instruction_size == 11:
        extra_instruction.append(INSTRUCTIONS_11)

    return """
    You are an SQL query generator. Given a question, you must generate a SQL query. If unsure do not assume the answer and give the default answer as "I don't know". Refer to the below context:
    {extra_instructions}
    
    [context]
    
    Hint: [hint]

    [question]

    [examples]
    """.format(extra_instructions="".join(extra_instruction))


def initialize_files(model_file_path: str) -> Tuple[str, str]:
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    log_file_path = f"{model_file_path}/execution-log.jsonl"

    output_file_path = f"{model_file_path}/predicted.txt"
    metrics_file_path = f"{model_file_path}/metrics.csv"
    metrics_file = open(metrics_file_path, "w", encoding="utf-8")
    metrics_file.write("response_time,llm_prompt_tokens,llm_response_tokens,hardness\n")

    open(output_file_path, "w", encoding="utf-8")
    open(log_file_path, "w", encoding="utf-8")
    return (output_file_path, metrics_file_path, log_file_path)


def generate_gold_file(gold_file_list: Tuple[Any, Any], model_file_path: str,inference_size: int) -> None:
    query_list, db_id_list = gold_file_list
    with open(f"{model_file_path}/gold.txt", "w") as f:
        for i in range(len(query_list)):
            f.write(f"{query_list[i]}\t{db_id_list[i]}\n\n")
    path_to_source_csv = f'../spider_data/bird_equal_split_{inference_size}.csv'
    df = pd.read_csv(path_to_source_csv)
    with open(f"{model_file_path}/dev_gold.sql", "w") as f:
        for i in range(len(query_list)):
            f.write(f"{query_list[i]}\t{db_id_list[i]}\n")
    data_list = df.to_dict(orient="records")
    with open(f"{model_file_path}/dev.json", "w") as json_file:
        json.dump(data_list, json_file, indent=4)
    
    with open(f"{model_file_path}/predicted.txt", "r",encoding='utf8') as file:
        sql_queries = file.readlines()
    with open(f"{model_file_path}/gold.txt", "r") as f:
        gold_query = f.readlines()
    queries_dict = {}
    count = 0
    for idx, query in enumerate(sql_queries):
        if query!="\n":
            db_id = gold_query[idx].split("\t")[1]
            queries_dict[count] = query.strip()+"\t----- bird -----\t"+db_id.replace('\n','')
            count = count + 1
    with open(f"{model_file_path}/predict_dev.json", "w") as json_file:
        json.dump(queries_dict, json_file, indent=4)


def get_elapsed_time(time_in_sec: int) -> None:
    minutes = time_in_sec // 60.0
    sec = time_in_sec % 60.0
    hr = minutes // 60.0
    minutes = minutes % 60.0
    return f"{math.trunc(hr)}hrs {math.trunc(minutes)}min {math.trunc(sec)}sec"


def write_to_file(
    output_file_path: str,
    metrics_file_path: str,
    output_file_text: str,
    metrics_file_text: str,
) -> None:
    with open(output_file_path, "a", encoding="utf-8") as file:
        file.write(output_file_text)
    with open(metrics_file_path, "a", encoding="utf-8") as file:
        file.write(metrics_file_text)


def log(log_text: str, data: Dict, log_file_path: str) -> None:
    data.update({"message": log_text})
    with open(log_file_path, "a") as json_file:
        json.dump(data, json_file)
        json_file.write("\n")


def get_parsed_args(supported_models: Dict, host_env: str) -> Tuple[Any, Dict]:
    parser = argparse.ArgumentParser(
        description=f"Run {host_env} specific NL-to-SQL LLM benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help=f"The model to use for this benchmark test. Supported models: {supported_models}",
    )
    parser.add_argument(
        "--model-instructions",
        "--model-inst",
        type=str,
        dest="model_instructions",
        default="",
        help=(
            "A slash('/') separated list of model specific instruction set to include in the results, e.g. "
            "cl-34=7,9,11/cl-70=5,7,9. The models specifed will run inferences for these instruction sets alone"
        ),
    )
    parser.add_argument(
        "--instructions",
        "--inst",
        dest="inst",
        type=str,
        default="0,5,7,9,11",
        help=(
            "A comma separated list of instructions set to include in the results, e.g. "
            "5,7,9. The models specifed will run inferences for these instruction sets alone"
        ),
    )
    parser.add_argument(
        "--inferences-length",
        "--il",
        dest="inf_length",
        type=str,
        default="360",
        help=(
            "A comma separated list of inferences to include for each results "
            "Example: 30,60,90,120,150,180,210,240,270,300,330,360. The models specifed will run inferences for these infernce-lengths alone.Make sure that there is a csv which as the name specified as bird_equal_split_{your inference length}"
        ),
    )
    parser.add_argument(
        "--shot-size",
        "--ss",
        dest="shot_size",
        type=str,
        default="0",
        help=(
            "A comma separated list of inferences to include for each results, currently supported shot size: "
            "2,4,6,8. The models specifed will run inferences for these infernce-lengths alone"
        ),
    )
    parser.add_argument(
        "--target-directory",
        type=str,
        dest="target_dir",
        default=CURRENT_FILE_PATH.parents[1],
        help="Name of the directory to store the compressed file",
    )

    parsed_args = parser.parse_args()
    parsed_args.target_dir = str(parsed_args.target_dir) + "/inference_results/"


    model_instructions = {}
    if parsed_args.model_instructions:
        for item in parsed_args.model_instructions.split("/"):
            model_key, inst_list_string = item.split("=")
            model_instructions[model_key] = [int(inst) for inst in inst_list_string.split(",")]
            key, value = item.split("=")
            model_instructions[key] = [int(inst) for inst in value.split(",")]

    return (parsed_args, model_instructions)


def initial_sql_match(sql_string: str) -> str:
    intial_regex = r"SELECT\b[^;]+"

    sql_string = (
        sql_string.replace("\n", " ")
        .replace("\\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("```", " ")
        .replace("...", " ")
    )
    match = re.findall(intial_regex, sql_string, re.IGNORECASE)
    if match:
        return match[0]
    else:
        return sql_string


def intermediate_sql_match(sql_string: str) -> Tuple[bool, str]:
    processed_str = initial_sql_match(sql_string)

    intermediate_regex = r"^(.*?)(?:\s*Explanation|\s*Caution|\s*This query|\s*The code|\s*Please|\s*The above query|\s*This SQL query|$)"
    match = re.findall(intermediate_regex, processed_str, re.IGNORECASE)
    if match:
        return (True, match[0])
    else:
        return (False, processed_str)

def sql_match(sql_string: str) -> Tuple[bool, str]:
    is_sql_match, processed_str = intermediate_sql_match(sql_string)
    if is_sql_match:
        return (True, processed_str)

    sql_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "ALTER",
        "DROP",
        "TRUNCATE",
        "HAVING",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "IN",
        "NOT IN",
        "BETWEEN",
        "LIKE",
        "IS NULL",
        "IS NOT NULL",
        "AND",
        "OR",
        "AS",
        "DISTINCT",
        "COUNT",
        "SUM",
        "AVG",
        "MAX",
        "MIN",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "ALL",
        "ANY",
        "EXISTS",
        "UNIQUE",
        "PRIMARY KEY",
        "FOREIGN KEY",
        "REFERENCES",
        "CHECK",
        "INDEX",
        "CASCADE",
        "CONSTRAINT",
        "DEFAULT",
        "AUTO_INCREMENT",
        "SEQUENCE",
        "VIEW",
        "TABLE",
        "DATABASE",
        "PROCEDURE",
        "FUNCTION",
        "TRIGGER",
        "GRANT",
        "REVOKE",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "START TRANSACTION",
        "LOCK",
        "UNLOCK",
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "ON",
        "GROUP BY",
        "ORDER BY",
        "LIMIT",
        "insert",
        "update",
        "delete",
        "create",
        "alter",
        "drop",
        "truncate",
        "having",
        "union",
        "intersect",
        "except",
        "in",
        "not in",
        "between",
        "like",
        "is null",
        "is not null",
        "and",
        "or",
        "as",
        "distinct",
        "count",
        "sum",
        "avg",
        "max",
        "min",
        "case",
        "when",
        "then",
        "else",
        "end",
        "all",
        "any",
        "exists",
        "unique",
        "primary key",
        "foreign key",
        "references",
        "check",
        "index",
        "cascade",
        "constraint",
        "default",
        "auto_increment",
        "sequence",
        "view",
        "table",
        "database",
        "procedure",
        "function",
        "trigger",
        "grant",
        "revoke",
        "commit",
        "rollback",
        "savepoint",
        "start transaction",
        "lock",
        "unlock",
        "select",
        "from",
        "where",
        "join",
        "on",
        "group by",
        "order by",
        "limit",
        "=",
        "*",
        "group",
        "GROUP",
        "by",
        "BY",
        ">",
        "<",
        ">=",
        "<=",
    ]
    sql_keyword_dict = {word: True for word in sql_keywords}
    answer = ""
    words = processed_str.split()
    for idx1 in range(len(words)):
        flag = 0
        if words[idx1].lower() == "from":
            for idx2 in range(idx1, len(words)):
                if words[idx2] in sql_keyword_dict:
                    answer += words[idx2] + " "

                else:
                    if (idx2 + 1) >= len(words):
                        answer += words[idx2] + " "
                        continue
                    if len(words[idx2 + 1]) == 1:
                        answer += words[idx2] + " "
                        continue
                    if words[idx2 + 1] in sql_keyword_dict:
                        answer += words[idx2] + " "
                    else:
                        if (idx2 + 2) >= len(words):
                            answer += words[idx2] + " "
                            continue
                        if len(words[idx2 + 2]) == 1:
                            answer += words[idx2] + " "
                            continue
                        if words[idx2 + 2] in sql_keyword_dict:
                            answer += words[idx2] + " "
                        else:
                            flag = 1
                            if words[idx2 - 1].lower() != "end":
                                answer += words[idx2] + " "
                            break
                if flag == 1:
                    break
            break
        else:
            answer += words[idx1]

    if answer == processed_str:
        return (False, answer)
    else:
        return (True, answer)
