"""Factory classes for creating LLM, Embedder, and Database clients."""

from openai import AsyncAzureOpenAI

from config.schema import (
    DatabaseConfig,
    EmbedderConfig,
    LLMConfig,
)

# Try to import FalkorDriver if available
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: F401

    HAS_FALKOR = True
except ImportError:
    HAS_FALKOR = False

# Kuzu support removed - FalkorDB is now the default
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

# Try to import additional providers if available
try:
    from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

    HAS_AZURE_EMBEDDER = True
except ImportError:
    HAS_AZURE_EMBEDDER = False

try:
    from graphiti_core.embedder.gemini import GeminiEmbedder

    HAS_GEMINI_EMBEDDER = True
except ImportError:
    HAS_GEMINI_EMBEDDER = False

try:
    from graphiti_core.embedder.voyage import VoyageAIEmbedder

    HAS_VOYAGE_EMBEDDER = True
except ImportError:
    HAS_VOYAGE_EMBEDDER = False

try:
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

    HAS_AZURE_LLM = True
except ImportError:
    HAS_AZURE_LLM = False

try:
    from graphiti_core.llm_client.anthropic_client import AnthropicClient

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from graphiti_core.llm_client.gemini_client import GeminiClient

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from graphiti_core.llm_client.groq_client import GroqClient

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

    HAS_GEMINI_RERANKER = True
except ImportError:
    HAS_GEMINI_RERANKER = False

from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from utils.utils import create_azure_credential_token_provider


def _validate_api_key(provider_name: str, api_key: str | None, logger) -> str:
    """Validate API key is present.

    Args:
        provider_name: Name of the provider (e.g., 'OpenAI', 'Anthropic')
        api_key: The API key to validate
        logger: Logger instance for output

    Returns:
        The validated API key

    Raises:
        ValueError: If API key is None or empty
    """
    if not api_key:
        raise ValueError(
            f'{provider_name} API key is not configured. Please set the appropriate environment variable.'
        )

    logger.info(f'Creating {provider_name} client')

    return api_key


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create an LLM client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI', api_key, logger)

                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                # Determine appropriate small model based on main model type
                is_reasoning_model = (
                    config.model.startswith('gpt-5')
                    or config.model.startswith('o1')
                    or config.model.startswith('o3')
                )
                small_model = (
                    'gpt-5-nano' if is_reasoning_model else 'gpt-4.1-mini'
                )  # Use reasoning model for small tasks if main model is reasoning

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    small_model=small_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                # Only pass reasoning/verbosity parameters for reasoning models (gpt-5 family)
                if is_reasoning_model:
                    return OpenAIClient(config=llm_config, reasoning='minimal', verbosity='low')
                else:
                    # For non-reasoning models, explicitly pass None to disable these parameters
                    return OpenAIClient(config=llm_config, reasoning=None, verbosity=None)

            case 'azure_openai':
                if not HAS_AZURE_LLM:
                    raise ValueError(
                        'Azure OpenAI LLM client not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Handle Azure AD authentication if enabled
                api_key: str | None = None
                azure_ad_token_provider = None
                if azure_config.use_azure_ad:
                    logger.info('Creating Azure OpenAI LLM client with Azure AD authentication')
                    azure_ad_token_provider = create_azure_credential_token_provider()
                else:
                    api_key = azure_config.api_key
                    _validate_api_key('Azure OpenAI', api_key, logger)

                # Create the Azure OpenAI client first
                azure_client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_config.api_url,
                    api_version=azure_config.api_version,
                    azure_deployment=azure_config.deployment_name,
                    azure_ad_token_provider=azure_ad_token_provider,
                )

                # Then create the LLMConfig
                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    base_url=azure_config.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                return AzureOpenAILLMClient(
                    azure_client=azure_client,
                    config=llm_config,
                    max_tokens=config.max_tokens,
                )

            case 'anthropic':
                if not HAS_ANTHROPIC:
                    raise ValueError(
                        'Anthropic client not available in current graphiti-core version'
                    )
                if not config.providers.anthropic:
                    raise ValueError('Anthropic provider configuration not found')

                api_key = config.providers.anthropic.api_key
                _validate_api_key('Anthropic', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return AnthropicClient(config=llm_config)

            case 'gemini':
                if not HAS_GEMINI:
                    raise ValueError('Gemini client not available in current graphiti-core version')
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                # Configure thinking for Gemini models
                # - Gemini 3.x: use thinking_level ("minimal", "low", "medium", "high")
                # - Gemini 2.5: use thinking_budget (0=off, -1=dynamic)
                # For structured extraction, we want minimal thinking to reduce latency
                thinking_config = None
                model_name = config.model.lower()

                try:
                    from google.genai import types

                    is_gemini_3 = 'gemini-3' in model_name or 'gemini3' in model_name
                    is_gemini_25 = '2.5' in model_name or '2-5' in model_name

                    if is_gemini_3:
                        # Gemini 3 models: use thinking_level
                        # "minimal" is closest to no-thinking for Gemini 3 Flash
                        try:
                            thinking_config = types.ThinkingConfig(thinking_level='minimal')
                            logger.info(
                                'Using ThinkingConfig with thinking_level="minimal" for Gemini 3'
                            )
                        except Exception as e:
                            # Catch all exceptions including pydantic ValidationError
                            logger.info(
                                f'thinking_level not supported: {e}, trying without ThinkingConfig'
                            )
                            thinking_config = None
                    elif is_gemini_25:
                        # Gemini 2.5 models: use thinking_budget=0 to disable
                        try:
                            thinking_config = types.ThinkingConfig(thinking_budget=0)
                            logger.info(
                                'Using ThinkingConfig with thinking_budget=0 for Gemini 2.5'
                            )
                        except Exception as e:
                            # Catch all exceptions including pydantic ValidationError
                            logger.info(
                                f'thinking_budget not supported: {e}, trying without ThinkingConfig'
                            )
                            thinking_config = None
                    else:
                        # Other/older models: no ThinkingConfig needed
                        logger.info(f'Model {config.model} does not require ThinkingConfig')

                except ImportError:
                    logger.info('google.genai.types not available, skipping ThinkingConfig')
                except Exception as e:
                    logger.warning(f'Error configuring ThinkingConfig: {e}, proceeding without it')

                if thinking_config is not None:
                    return GeminiClient(config=llm_config, thinking_config=thinking_config)
                else:
                    return GeminiClient(config=llm_config)

            case 'groq':
                if not HAS_GROQ:
                    raise ValueError('Groq client not available in current graphiti-core version')
                if not config.providers.groq:
                    raise ValueError('Groq provider configuration not found')

                api_key = config.providers.groq.api_key
                _validate_api_key('Groq', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    base_url=config.providers.groq.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GroqClient(config=llm_config)

            case _:
                raise ValueError(f'Unsupported LLM provider: {provider}')


class EmbedderFactory:
    """Factory for creating Embedder clients based on configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        """Create an Embedder client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Embedder', api_key, logger)

                from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                )
                return OpenAIEmbedder(config=embedder_config)

            case 'azure_openai':
                if not HAS_AZURE_EMBEDDER:
                    raise ValueError(
                        'Azure OpenAI embedder not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Handle Azure AD authentication if enabled
                api_key: str | None = None
                azure_ad_token_provider = None
                if azure_config.use_azure_ad:
                    logger.info(
                        'Creating Azure OpenAI Embedder client with Azure AD authentication'
                    )
                    azure_ad_token_provider = create_azure_credential_token_provider()
                else:
                    api_key = azure_config.api_key
                    _validate_api_key('Azure OpenAI Embedder', api_key, logger)

                # Create the Azure OpenAI client first
                azure_client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_config.api_url,
                    api_version=azure_config.api_version,
                    azure_deployment=azure_config.deployment_name,
                    azure_ad_token_provider=azure_ad_token_provider,
                )

                return AzureOpenAIEmbedderClient(
                    azure_client=azure_client,
                    model=config.model or 'text-embedding-3-small',
                )

            case 'gemini':
                if not HAS_GEMINI_EMBEDDER:
                    raise ValueError(
                        'Gemini embedder not available in current graphiti-core version'
                    )
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini Embedder', api_key, logger)

                from graphiti_core.embedder.gemini import GeminiEmbedderConfig

                # Use Gemini default if model is unset or still using OpenAI default
                embedding_model = config.model
                if not embedding_model or embedding_model == 'text-embedding-3-small':
                    embedding_model = 'gemini-embedding-001'

                # Use appropriate dimensions for Gemini (default 768)
                dimensions = config.dimensions
                if not dimensions or dimensions == 1536:  # 1536 is OpenAI default
                    dimensions = 768

                gemini_config = GeminiEmbedderConfig(
                    api_key=api_key,
                    embedding_model=embedding_model,
                    embedding_dim=dimensions,
                )
                return GeminiEmbedder(config=gemini_config)

            case 'voyage':
                if not HAS_VOYAGE_EMBEDDER:
                    raise ValueError(
                        'Voyage embedder not available in current graphiti-core version'
                    )
                if not config.providers.voyage:
                    raise ValueError('Voyage provider configuration not found')

                api_key = config.providers.voyage.api_key
                _validate_api_key('Voyage Embedder', api_key, logger)

                from graphiti_core.embedder.voyage import VoyageAIEmbedderConfig

                # Use Voyage default if model is unset or still using OpenAI default
                embedding_model = config.model
                if not embedding_model or embedding_model == 'text-embedding-3-small':
                    embedding_model = 'voyage-3'

                # Use appropriate dimensions for Voyage (default 1024)
                dimensions = config.dimensions
                if not dimensions or dimensions == 1536:  # 1536 is OpenAI default
                    dimensions = 1024

                voyage_config = VoyageAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=embedding_model,
                    embedding_dim=dimensions,
                )
                return VoyageAIEmbedder(config=voyage_config)

            case _:
                raise ValueError(f'Unsupported Embedder provider: {provider}')


class DatabaseDriverFactory:
    """Factory for creating Database drivers based on configuration.

    Note: This returns configuration dictionaries that can be passed to Graphiti(),
    not driver instances directly, as the drivers require complex initialization.
    """

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        """Create database configuration dictionary based on the configured provider."""
        provider = config.provider.lower()

        match provider:
            case 'neo4j':
                # Use Neo4j config if provided, otherwise use defaults
                if config.providers.neo4j:
                    neo4j_config = config.providers.neo4j
                else:
                    # Create default Neo4j configuration
                    from config.schema import Neo4jProviderConfig

                    neo4j_config = Neo4jProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os

                uri = os.environ.get('NEO4J_URI', neo4j_config.uri)
                username = os.environ.get('NEO4J_USER', neo4j_config.username)
                password = os.environ.get('NEO4J_PASSWORD', neo4j_config.password)

                return {
                    'uri': uri,
                    'user': username,
                    'password': password,
                    # Note: database and use_parallel_runtime would need to be passed
                    # to the driver after initialization if supported
                }

            case 'falkordb':
                if not HAS_FALKOR:
                    raise ValueError(
                        'FalkorDB driver not available in current graphiti-core version'
                    )

                # Use FalkorDB config if provided, otherwise use defaults
                if config.providers.falkordb:
                    falkor_config = config.providers.falkordb
                else:
                    # Create default FalkorDB configuration
                    from config.schema import FalkorDBProviderConfig

                    falkor_config = FalkorDBProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os
                from urllib.parse import urlparse

                uri = os.environ.get('FALKORDB_URI', falkor_config.uri)
                password = os.environ.get('FALKORDB_PASSWORD', falkor_config.password)

                # Parse the URI to extract host and port
                parsed = urlparse(uri)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 6380

                return {
                    'driver': 'falkordb',
                    'host': host,
                    'port': port,
                    'password': password,
                    'database': falkor_config.database,
                }

            case _:
                raise ValueError(f'Unsupported Database provider: {provider}')


class CrossEncoderFactory:
    """Factory for creating CrossEncoder/Reranker clients based on LLM configuration."""

    @staticmethod
    def create(config: LLMConfig) -> CrossEncoderClient:
        """Create a CrossEncoder client based on the configured LLM provider.

        The cross-encoder/reranker should match the LLM provider to avoid
        requiring multiple API keys.
        """
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'gemini':
                if not HAS_GEMINI_RERANKER:
                    raise ValueError(
                        'Gemini reranker not available in current graphiti-core version'
                    )
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini Reranker', api_key, logger)

                reranker_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model='gemini-2.5-flash-lite',  # Use lite model for reranking
                )
                return GeminiRerankerClient(config=reranker_config)

            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Reranker', api_key, logger)

                reranker_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model='gpt-4.1-nano',  # Use small model for reranking
                )
                return OpenAIRerankerClient(config=reranker_config)

            case 'azure_openai':
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                api_key: str | None = None
                azure_ad_token_provider = None
                if azure_config.use_azure_ad:
                    logger.info('Creating Azure OpenAI Reranker with Azure AD authentication')
                    azure_ad_token_provider = create_azure_credential_token_provider()
                else:
                    api_key = azure_config.api_key
                    _validate_api_key('Azure OpenAI Reranker', api_key, logger)

                azure_client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_config.api_url,
                    api_version=azure_config.api_version,
                    azure_deployment=azure_config.deployment_name,
                    azure_ad_token_provider=azure_ad_token_provider,
                )

                reranker_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                )
                return OpenAIRerankerClient(config=reranker_config, client=azure_client)

            case _:
                # For other providers (anthropic, groq), fall back to OpenAI reranker
                # if OpenAI config is available, otherwise raise error
                if config.providers.openai and config.providers.openai.api_key:
                    api_key = config.providers.openai.api_key
                    logger.info(f'Using OpenAI reranker as fallback for {provider} LLM provider')
                    reranker_config = GraphitiLLMConfig(
                        api_key=api_key,
                        model='gpt-4.1-nano',
                    )
                    return OpenAIRerankerClient(config=reranker_config)
                else:
                    raise ValueError(
                        f'No reranker available for {provider}. '
                        f'Either use Gemini/OpenAI as LLM provider, or set OPENAI_API_KEY for reranking.'
                    )
