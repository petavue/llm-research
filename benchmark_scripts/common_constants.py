class Defaults:
    INFERENCE_LENGTH_LIST = [360]
    INSTRUCTION_SIZE_LIST = [0,5,7,9,11]
    MAX_TOKENS_TO_GENERATE = 300
    anthropic_version_const = "bedrock-2023-05-31"


class Environments:
    OPEN_AI = "open-ai"
    ANYSCALE = "anyscale"
    AMZ_BEDROCK = "amazon-bedrock"
    SELF_HOSTED = "self-hosted"
    HUGGING_FACE = "hugging-face"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class BedrockModels:
    MODEL_META_LLAMA = "meta.llama2-70b-chat-v1"
    MODEL_ANTHROPIC_CLAUDE = "anthropic.claude-v2"
    MODEL_ANTHROPIC_CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    MODEL_ANTHROPIC_CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    MODEL_ANTHROPIC_MIXTRAL = "mistral.mixtral-8x7b-instruct-v0:1"
    MODEL_ANTHROPIC_MISTRAL_7B = "mistral.mistral-7b-instruct-v0:2"


class AnyscaleModels:
    MODEL_META_LLAMA = "meta-llama/Llama-2-70b-chat-hf"
    MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
    MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
    MODEL_MISTRALAI_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
    MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"

class SelfHostedModels:
    MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
    MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
    MODEL_MISTRALAI_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
    MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MODEL_WIZARDLM_WIZARD_CODER_33B = "WizardLM/WizardCoder-33B-V1.1"
    MODEL_DEFOG_SQLCODER_70B = "defog/sqlcoder-70b-alpha"
    MODEL_DEFOG_SQLCODER_7B_2 = "defog/sqlcoder-7b-2"
    MODEL_MISTRALAI_MISTRAL_7B_V2 = "mistralai/Mistral-7B-Instruct-v0.2"
    
class SelfHosted:
    MODEL_WEIGHTS_DIRECTORY = "../model-weights/"

class AnthropicModels:
    MODEL_ANTHROPIC_OPUS = "claude-3-opus-20240229"
    MODEL_ANTHROPIC_SONNET = "claude-3-sonnet-20240229"
    MODEL_ANTHROPIC_HAIKU = "claude-3-haiku-20240307"

class GeminiModels:
    MODEL_GEMINI_1_PRO = "models/gemini-1.0-pro-latest"

class OpenAIModels:
    MODEL_GPT_3 = "gpt-3.5-turbo-16k"
    MODEL_GPT_4 = "gpt-4-turbo-preview"