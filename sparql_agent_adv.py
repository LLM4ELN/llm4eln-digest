"""
Pydantic AI Agent Demo for SPARQL Query Generation and Validation
"""

import os
from typing import Any

import logfire
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from SPARQLWrapper import JSON, SPARQLWrapper

from llm_init_pydantic import llm

# Simple cache for classification results
_classification_cache: dict[tuple, Any] = {}
_MAX_CACHE_SIZE = 100

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

logfire.info("Hello from {name}!", name="Pydantic AI")

# ============================================================================
# 1. Define Schema Models for SPARQL Components
# ============================================================================


class SPARQLPrefix(BaseModel):
    """Represents a SPARQL prefix declaration"""

    prefix: str = Field(description="Prefix name (e.g., 'rdfs')")
    uri: str = Field(description="Full URI for the prefix")


class SPARQLSchema(BaseModel):
    """Schema knowledge for SPARQL query generation"""

    prefixes: list[SPARQLPrefix] = Field(description="Available prefixes and their URIs")
    properties: list[str] = Field(description="Valid property predicates (e.g., 'IsPartOf', 'HasName')")
    categories: list[str] = Field(
        description="Valid category instances (e.g., 'Category:OSW767ea23b639f5bcb9430a562cfc8be7c')"
    )
    data_types: list[str] = Field(
        default_factory=list, description="Valid data types for filtering (e.g., 'LREAL', 'REAL', 'BOOL')"
    )


class GeneratedSPARQLQuery(BaseModel):
    """Validated SPARQL query output"""

    query: str = Field(description="The complete SPARQL query string")
    explanation: str = Field(description="Brief explanation of what the query does")
    prefixes_used: list[str] = Field(description="List of prefixes used in the query")


# ============================================================================
# 2. Define Dependencies (Schema Knowledge + SPARQL Endpoint)
# ============================================================================


class SPARQLEndpointConfig(BaseModel):
    """Configuration for SPARQL endpoint connection"""

    model_config = {"frozen": True}

    endpoint_url: str = Field(description="SPARQL endpoint URL")
    username: str | None = Field(default=None, description="Optional username for authentication")
    password: str | None = Field(default=None, description="Optional password for authentication")


def create_sparql_wrapper(config: SPARQLEndpointConfig) -> SPARQLWrapper:
    """
    Create and configure a SPARQLWrapper instance with authentication.

    Args:
        config: SPARQL endpoint configuration

    Returns:
        Configured SPARQLWrapper instance
    """
    sparql = SPARQLWrapper(config.endpoint_url)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(30)

    if config.username and config.password:
        sparql.setHTTPAuth("BASIC")
        sparql.setCredentials(config.username, config.password)

    return sparql


class ImprovedQuestion(BaseModel):
    """Output from question improvement agent"""

    improved_question: str = Field(description="Clarified and improved version of the question")
    changes_made: list[str] = Field(description="List of improvements made to the original question")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")


class QuestionImprovementDeps(BaseModel):
    """Dependencies for question improvement agent"""

    original_question: str
    schema_context: str = Field(default="", description="Brief context about available schema")


class SchemaClassification(BaseModel):
    """Output from classification agent selecting relevant schema elements"""

    relevant_properties: list[str] = Field(description="List of property URIs relevant to the question")
    relevant_categories: list[str] = Field(description="List of category URIs relevant to the question")


class SchemaClassificationDeps(BaseModel):
    """Dependencies for the classification agent"""

    properties: list[dict] = Field(description="List of property dicts with uri, label, comment")
    categories: list[dict] = Field(description="List of category dicts with uri, label, comment")


improvement_agent = Agent(
    llm, output_type=ImprovedQuestion, deps_type=QuestionImprovementDeps
)


@improvement_agent.system_prompt
def improvement_prompt(ctx: RunContext[QuestionImprovementDeps]) -> str:
    """System prompt for question improvement"""
    return f"""You are a question improvement expert for SPARQL query generation.

Your task is to improve user questions to make them more suitable for generating SPARQL queries.

Improvements to make:
1. Clarify ambiguous terms and expand abbreviations
2. Add missing context (e.g., what properties/relationships to include)
3. Translate natural language to SPARQL query requirements
4. Validate completeness and suggest missing information
5. Make questions more specific and actionable

Guidelines:
- Preserve the user's intent
- Keep the improved question concise but complete
- Focus on what data to retrieve and how to structure it
- Consider SPARQL query patterns (SELECT, WHERE, FILTER, etc.)

Original question: {ctx.deps.original_question}

Provide an improved version that is clearer and more suitable for SPARQL query generation."""


async def improve_question(question: str) -> ImprovedQuestion:
    """
    Improve a user question using the improvement agent.

    Args:
        question: Original user question

    Returns:
        ImprovedQuestion with enhanced question and change notes
    """
    deps = QuestionImprovementDeps(original_question=question)
    result = await improvement_agent.run(question, deps=deps)
    return result.output


classifier_agent = Agent(llm, output_type=SchemaClassification, deps_type=SchemaClassificationDeps)


@classifier_agent.system_prompt
def classifier_prompt(ctx: RunContext[SchemaClassificationDeps]) -> str:
    """System prompt for the classification agent"""
    props = "\n".join(
        [
            f"- {p['uri']}" + (f" ({p.get('label')})" if p.get("label") else "")
            for p in ctx.deps.properties
        ]
    )
    cats = "\n".join(
        [
            f"- {c['uri']}" + (f" ({c.get('label')})" if c.get("label") else "")
            for c in ctx.deps.categories
        ]
    )
    return f"""Select 10-30 properties and 5-15 categories relevant to the question.

Properties:
{props}

Categories:
{cats}"""


async def classify(question: str, props: tuple, cats: tuple) -> SchemaClassification:
    """
    Use the classification agent to select relevant schema elements.
    Results are cached based on question and schema content.

    Args:
        question: User's question
        props: Tuple of property dicts (converted to tuple for caching)
        cats: Tuple of category dicts

    Returns:
        SchemaClassification with selected elements
    """
    # Create hashable cache key from URIs only (since dicts aren't hashable)
    prop_uris = tuple(sorted(p["uri"] for p in props))
    cat_uris = tuple(sorted(c["uri"] for c in cats))
    cache_key = (question, prop_uris, cat_uris)

    if cache_key in _classification_cache:
        return _classification_cache[cache_key]

    deps = SchemaClassificationDeps(properties=list(props), categories=list(cats))
    result = await classifier_agent.run(question, deps=deps)
    classification = result.output

    # Simple cache eviction if too large
    if len(_classification_cache) >= _MAX_CACHE_SIZE:
        _classification_cache.clear()
    _classification_cache[cache_key] = classification

    return classification


class SPARQLDependencies(BaseModel):
    """Dependencies injected into the agent"""

    schema: SPARQLSchema
    endpoint_config: SPARQLEndpointConfig
    properties: list[dict] = Field(description="Full property dicts with uri, label, comment")
    categories: list[dict] = Field(description="Full category dicts with uri, label, comment")
    selected_properties: list[dict] = Field(
        default_factory=list, description="Selected properties for current question"
    )
    selected_categories: list[dict] = Field(
        default_factory=list, description="Selected categories for current question"
    )

    def get_schema_context(self) -> str:
        """Format schema information using pre-selected elements"""
        # Use selected elements if available, otherwise use all
        props = self.selected_properties if self.selected_properties else self.properties
        cats = self.selected_categories if self.selected_categories else self.categories

        # Format schema context
        prefix_list = "\n".join([f"PREFIX {p.prefix}: <{p.uri}>" for p in self.schema.prefixes])

        props_text = "\n".join(
            [
                f"- {p['uri']}" + (f" ({p.get('label')})" if p.get("label") else "")
                for p in props
            ]
        )
        cats_text = "\n".join(
            [
                f"- {c['uri']}" + (f" ({c.get('label')})" if c.get("label") else "")
                for c in cats
            ]
        )

        schema_info = f"""
        Available SPARQL Prefixes:
        {prefix_list}

        Relevant Properties (selected for this question):
        {props_text}

        Relevant Categories (selected for this question):
        {cats_text}
        """

        if self.schema.data_types:
            dt_list = ", ".join(self.schema.data_types)
            schema_info += f"\nValid Data Types for Filtering:\n{dt_list}"

        return schema_info


async def prepare_deps_with_classification(
    question: str, deps: SPARQLDependencies, improve: bool = True
) -> tuple[str, SPARQLDependencies]:
    """
    Improve question, classify schema elements, and update dependencies with selected elements.

    Args:
        question: User's question
        deps: SPARQLDependencies instance
        improve: Whether to improve the question first (default: True)

    Returns:
        Tuple of (improved_question, updated_deps) where updated_deps has selected elements
    """
    # Step 1: Improve the question
    if improve:
        improved = await improve_question(question)
        question = improved.improved_question
        if improved.changes_made:
            print(f"💡 Question improvements: {', '.join(improved.changes_made)}")

    # Step 2: Classify schema elements
    cls = await classify(question, tuple(deps.properties), tuple(deps.categories))

    # Filter to selected URIs
    selected_props = [p for p in deps.properties if p["uri"] in cls.relevant_properties]
    selected_cats = [c for c in deps.categories if c["uri"] in cls.relevant_categories]

    # Create updated deps with selected elements
    updated_deps = SPARQLDependencies(
        schema=deps.schema,
        endpoint_config=deps.endpoint_config,
        properties=deps.properties,
        categories=deps.categories,
        selected_properties=selected_props,
        selected_categories=selected_cats,
    )

    return question, updated_deps


# ============================================================================
# 3. Create Pydantic AI Agent with Schema Awareness
# ============================================================================


sparql_agent = Agent(
    llm,
    output_type=GeneratedSPARQLQuery,
    deps_type=SPARQLDependencies,
    retries=2,
)


@sparql_agent.system_prompt
def create_system_prompt(ctx: RunContext[SPARQLDependencies]) -> str:
    """Dynamic system prompt with question-specific schema elements"""
    schema_context = ctx.deps.get_schema_context()

    return f"""You are a SPARQL query generation expert. Generate valid SPARQL queries
    based on user requests using ONLY the schema elements provided below.

    {schema_context}

    Rules:
    - Use only prefixes, properties, and categories from the schema above
    - The properties and categories shown were intelligently selected for this specific question
    - Generate syntactically correct SPARQL SELECT queries
    - Include appropriate FILTER clauses when filtering by data types
    - Use GROUP_CONCAT with separator when aggregating multiple values
    - Use OPTIONAL blocks only when asked explicitly to do so
    - Use property paths (e.g., IsPartOf+) for transitive relationships
    - Always add LIMIT clause (default: 100) to prevent timeout on large datasets
    - Be concise and to the point, avoid unnecessary complexity

    Workflow:
    1. Generate an initial SPARQL query based on the user's request
    2. Use the execute_query tool to test your query against the actual endpoint
    3. If the query returns no results or fails:
       - Analyze why it might have failed (wrong properties, wrong filters, syntax issues)
       - Generate a revised query addressing the issues
       - Test again using execute_query
    4. Iterate until you get successful results or determine the request cannot be satisfied
    5. Return your final working query with explanation
    """


# ============================================================================
# 4. Add Tool for Query Validation
# ============================================================================


@sparql_agent.tool
def validate_sparql_syntax(ctx: RunContext[SPARQLDependencies], query: str) -> dict[str, Any]:
    """
    Validate SPARQL query syntax by attempting to parse it.

    Args:
        ctx: Runtime context with dependencies
        query: SPARQL query string to validate

    Returns:
        Dictionary with validation status and any errors
    """
    try:
        sparql = create_sparql_wrapper(ctx.deps.endpoint_config)
        sparql.setQuery(query)
        # Validate query string is parseable
        _ = sparql.queryString
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": "Query has syntax errors",
        }
    else:
        return {
            "valid": True,
            "message": "Query syntax is valid",
        }


@sparql_agent.tool
def execute_query(ctx: RunContext[SPARQLDependencies], query: str) -> dict[str, Any]:
    """
    Execute a SPARQL query and return results summary.

    Args:
        ctx: Runtime context with dependencies
        query: SPARQL query string to execute

    Returns:
        Dictionary with execution results, row count, columns, and sample data

    Raises:
        ModelRetry: If the query returns no results, triggering the agent to try a different query
    """
    df = execute_sparql_query(ctx.deps.endpoint_config, query)

    if df.empty:
        msg = "Query returned no results. Try different properties, filters, or verify the data exists in the schema."
        raise ModelRetry(msg)

    return {
        "success": True,
        "row_count": len(df),
        "columns": df.columns.tolist(),
        "sample_rows": df.head(3).to_dict("records"),
        "message": f"Query successful, returned {len(df)} rows",
    }


# ============================================================================
# 5. Execute SPARQL Query
# ============================================================================


def execute_sparql_query(
    config: SPARQLEndpointConfig,
    query: str,
) -> pd.DataFrame:
    """
    Execute a SPARQL query and return results as DataFrame.

    Args:
        config: SPARQL endpoint configuration
        query: SPARQL query string

    Returns:
        DataFrame with query results
    """
    try:
        sparql = create_sparql_wrapper(config)
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if bindings:
            rows = []
            for binding in bindings:
                row = {k: v.get("value", "") for k, v in binding.items()}
                rows.append(row)
            return pd.DataFrame(rows)

        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error executing SPARQL query: {e}")
        return pd.DataFrame()


# ============================================================================
# 6. Introspect Schema from SPARQL Endpoint
# ============================================================================


def retrieve_sparql_entities(
    config: SPARQLEndpointConfig,
    entity_type: str,
    where_clause: str,
    result_key: str,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Generic function to retrieve entities from SPARQL endpoint.

    Args:
        config: SPARQL endpoint configuration
        entity_type: The SPARQL variable name to select (e.g., 'predicate', 'category')
        where_clause: The WHERE clause pattern (e.g., '?s ?predicate ?o', '?s a ?category')
        result_key: The key name for the results in the returned dict (e.g., 'predicates', 'categories')
        limit: Max number of distinct entities to retrieve

    Returns:
        Dictionary with discovered entities (as dicts with uri, label, comment) and count
    """
    query = f"""
    SELECT DISTINCT ?{entity_type} ?label ?comment
    WHERE {{
        {where_clause}
        OPTIONAL {{ ?{entity_type} <http://www.w3.org/2000/01/rdf-schema#label> ?label . }}
        OPTIONAL {{ ?{entity_type} <http://www.w3.org/2000/01/rdf-schema#comment> ?comment . }}
    }}
    LIMIT {limit}
    """

    try:
        sparql = create_sparql_wrapper(config)
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        entities = []
        for b in bindings:
            entity = {
                "uri": b[entity_type]["value"],
                "label": b.get("label", {}).get("value") if "label" in b else None,
                "comment": b.get("comment", {}).get("value") if "comment" in b else None,
            }
            entities.append(entity)
        return {result_key: entities, "count": len(entities)}
    except Exception as e:
        print(f"⚠️  {entity_type.capitalize()} introspection failed: {e}")
        return {result_key: [], "count": 0}


def retrieve_properties(
    config: SPARQLEndpointConfig,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Introspect SPARQL endpoint to discover available predicates.

    Args:
        config: SPARQL endpoint configuration
        limit: Max number of distinct predicates to retrieve

    Returns:
        List of property dicts with uri, label, comment
    """
    result = retrieve_sparql_entities(config, "predicate", "?s ?predicate ?o", "predicates", limit)
    return result["predicates"]


def retrieve_categories(
    config: SPARQLEndpointConfig,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Introspect SPARQL endpoint to discover available categories.

    Args:
        config: SPARQL endpoint configuration
        limit: Max number of distinct categories to retrieve

    Returns:
        List of category dicts with uri, label, comment
    """
    result = retrieve_sparql_entities(config, "category", "?s a ?category", "categories", limit)
    return result["categories"]


# ============================================================================
# 7. Main Demo Function
# ============================================================================


async def main():
    """Main demo showing agent usage"""

    print("🔧 Initializing SPARQL Agent with pre-configured LLM...")
    print(f"   SPARQL Endpoint: {os.getenv('BLAZEGRAPH_ENDPOINT')}\n")

    endpoint_config = SPARQLEndpointConfig(
        endpoint_url=os.getenv("BLAZEGRAPH_ENDPOINT"),
        username=os.getenv("BLAZEGRAPH_USER"),
        password=os.getenv("BLAZEGRAPH_PASSWORD"),
    )

    # Retrieve schema elements WITH descriptions
    print("📚 Fetching schema properties and categories with descriptions...")
    properties_with_desc = retrieve_properties(endpoint_config, limit=200)
    categories_with_desc = retrieve_categories(endpoint_config, limit=200)

    print(f"✅ Found {len(properties_with_desc)} properties and {len(categories_with_desc)} categories")
    print("🧠 Classification agent ready (will select relevant elements per question)")

    llm4eln_schema = SPARQLSchema(
        prefixes=[
            SPARQLPrefix(prefix="rdfs", uri="http://www.w3.org/2000/01/rdf-schema#"),
            SPARQLPrefix(prefix="owl", uri="http://www.w3.org/2002/07/owl#"),
            SPARQLPrefix(prefix="osl", uri="https://llm4eln.semos.dev/id/"),
            SPARQLPrefix(prefix="Property", uri="https://llm4eln.semos.dev/id/Property-3A"),
            SPARQLPrefix(prefix="Category", uri="https://llm4eln.semos.dev/id/Category-3A"),
            SPARQLPrefix(prefix="Item", uri="https://llm4eln.semos.dev/id/Item-3A"),
        ],
        properties=[p["uri"] for p in properties_with_desc],  # Store URIs for compatibility
        categories=[c["uri"] for c in categories_with_desc],
        data_types=[],
    )

    deps = SPARQLDependencies(
        schema=llm4eln_schema,
        endpoint_config=endpoint_config,
        properties=properties_with_desc,  # Full dicts with uri, label, comment
        categories=categories_with_desc,  # Full dicts with uri, label, comment
    )

    # Example 1: Query triples with display titles
    print("\n" + "=" * 80)
    print("Example 1: Query all triples for an entity with OSW ID")
    print("=" * 80)

    osw_id_example = "Item:OSW4dda5eac74284e6fab96ad97b73685b0"
    osw_id_example = "Category:OSWe4c3e517f2a5444597c29d6b82297f09"

    question1 = (
        f"Find all triples (subject, predicate, object) where either the subject or object "
        f"has OSW ID '{osw_id_example}'. Show display titles for all components."
    )
    improved_question1, deps1 = await prepare_deps_with_classification(question1, deps)
    result = await sparql_agent.run(improved_question1, deps=deps1)

    print(f"\n📝 Explanation: {result.output.explanation}\n")
    print(f"🔧 Prefixes used: {', '.join(result.output.prefixes_used)}\n")
    print("📋 Generated SPARQL Query:")
    print("-" * 80)
    print(result.output.query)
    print("-" * 80)

    # Example 2: Execute a simple query
    print("\n" + "=" * 80)
    print("Example 2: Execute query and show results")
    print("=" * 80)

    question2 = "Get the first 10 entities with their labels and OSW IDs"
    improved_question2, deps2 = await prepare_deps_with_classification(question2, deps)
    result = await sparql_agent.run(improved_question2, deps=deps2)

    print(f"\n📝 Explanation: {result.output.explanation}\n")
    print(f"🔧 Prefixes used: {', '.join(result.output.prefixes_used)}\n")
    print("📋 Generated SPARQL Query:")
    print("-" * 80)
    print(result.output.query)
    print("-" * 80)

    # Execute the query
    print("\n🚀 Executing query...")
    df = execute_sparql_query(deps.endpoint_config, result.output.query)

    if not df.empty:
        print(f"✅ Query returned {len(df)} rows\n")
        print(df.head(10).to_string(index=False))
    else:
        print("⚠️  Query returned no results")

    # Interactive mode: Ask questions in the terminal
    print("\n" + "=" * 80)
    print("Interactive Mode - Ask your own SPARQL questions!")
    print("=" * 80)
    print("Type your question below (or 'quit'/'exit' to stop)")
    print("The agent remembers previous questions and can reference them!\n")

    message_history = []

    while True:
        try:
            user_question = input("🔍 Your question: ").strip()

            if not user_question:
                continue

            if user_question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            print("\n🤖 Generating and testing query ...")
            improved_question, deps_classified = await prepare_deps_with_classification(
                user_question, deps
            )
            result = await sparql_agent.run(
                improved_question, message_history=message_history, deps=deps_classified
            )

            message_history = result.all_messages()

            print(f"\n📝 Explanation: {result.output.explanation}\n")
            print(f"🔧 Prefixes used: {', '.join(result.output.prefixes_used)}\n")
            print("📋 Generated SPARQL Query:")
            print("-" * 80)
            print(result.output.query)
            print("-" * 80)

            # Optionally execute and show results
            execute = input("\n▶️  Execute this query? (y/n): ").strip().lower()
            if execute == "y":
                print("\n🚀 Executing query...")
                df = execute_sparql_query(deps.endpoint_config, result.output.query)

                if not df.empty:
                    print(f"\n✅ Query returned {len(df)} rows\n")
                    print(df.head(20).to_string(index=False))
                else:
                    print("\n⚠️  Query returned no results")

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

"""
Questions:
find all triples where any item has OSW ID "Item:OSW4dda5eac74284e6fab96ad97b73685b0"
Search for Jane Doe
Can you tell me about her?
"""
