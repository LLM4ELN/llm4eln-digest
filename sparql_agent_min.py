"""
Pydantic AI Agent Demo for SPARQL Query Generation and Validation
"""

import os
from typing import Any

import logfire
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from SPARQLWrapper import JSON, SPARQLWrapper

from llm_init_pydantic import llm

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

    endpoint_url: str = Field(description="SPARQL endpoint URL")
    username: str | None = Field(default=None, description="Optional username for authentication")
    password: str | None = Field(default=None, description="Optional password for authentication")


class SPARQLDependencies(BaseModel):
    """Dependencies injected into the agent"""

    schema: SPARQLSchema
    endpoint_config: SPARQLEndpointConfig

    def get_schema_context(self) -> str:
        """Format schema information for the system prompt"""
        prefix_list = "\n".join([f"PREFIX {p.prefix}: <{p.uri}>" for p in self.schema.prefixes])
        props = "\n".join([f"- {p}" for p in self.schema.properties])
        cats = "\n".join([f"- {c}" for c in self.schema.categories]) if self.schema.categories else "N/A"

        schema_info = f"""
        Available SPARQL Prefixes:
        {prefix_list}

        Valid Properties (Predicates):
        {props}

        Valid Categories:
        {cats}
        """

        if self.schema.data_types:
            dt_list = ", ".join(self.schema.data_types)
            schema_info += f"\nValid Data Types for Filtering:\n{dt_list}"

        return schema_info


# ============================================================================
# 3. Create Pydantic AI Agent with Schema Awareness
# ============================================================================


sparql_agent = Agent(llm, output_type=GeneratedSPARQLQuery, deps_type=SPARQLDependencies)


@sparql_agent.system_prompt
def create_system_prompt(ctx: RunContext[SPARQLDependencies]) -> str:
    """Dynamic system prompt with schema information"""
    return f"""You are a SPARQL query generation expert. Generate valid SPARQL queries
    based on user requests using ONLY the schema elements provided below.

    {ctx.deps.get_schema_context()}

    Rules:
    - Use only prefixes, properties, and categories from the schema above
    - Generate syntactically correct SPARQL SELECT queries
    - Include appropriate FILTER clauses when filtering by data types
    - Use GROUP_CONCAT with separator when aggregating multiple values
    - Use OPTIONAL blocks only when asked explicitly to do so
    - Use property paths (e.g., IsPartOf+) for transitive relationships
    - Be concise and to the point, avoid unnecessary complexity
    """


# ============================================================================
# 4. Add Tool for Query Validation
# ============================================================================


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

    if config.username and config.password:
        sparql.setHTTPAuth("BASIC")
        sparql.setCredentials(config.username, config.password)

    return sparql


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


# ============================================================================
# 5. Introspect Schema from SPARQL Endpoint
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
        Dictionary with discovered entities and count
    """
    query = f"""
    SELECT DISTINCT ?{entity_type}
    WHERE {{
        {where_clause}
    }}
    LIMIT {limit}
    """

    try:
        sparql = create_sparql_wrapper(config)
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        entities = [b[entity_type]["value"] for b in bindings]
        return {result_key: entities, "count": len(entities)}
    except Exception as e:
        print(f"⚠️  {entity_type.capitalize()} introspection failed: {e}")
        return {result_key: [], "count": 0}


def retrieve_properties(
    config: SPARQLEndpointConfig,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Introspect SPARQL endpoint to discover available predicates.

    Args:
        config: SPARQL endpoint configuration
        limit: Max number of distinct predicates to retrieve

    Returns:
        Dictionary with discovered predicates
    """
    return retrieve_sparql_entities(config, "predicate", "?s ?predicate ?o", "predicates", limit)


def retrieve_categories(
    config: SPARQLEndpointConfig,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Introspect SPARQL endpoint to discover available categories.

    Args:
        config: SPARQL endpoint configuration
        limit: Max number of distinct categories to retrieve

    Returns:
        Dictionary with discovered categories
    """
    return retrieve_sparql_entities(config, "category", "?s a ?category", "categories", limit)


# ============================================================================
# 6. Main Demo Function
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

    llm4eln_schema = SPARQLSchema(
        prefixes=[
            SPARQLPrefix(prefix="rdfs", uri="http://www.w3.org/2000/01/rdf-schema#"),
            SPARQLPrefix(prefix="owl", uri="http://www.w3.org/2002/07/owl#"),
            SPARQLPrefix(prefix="osl", uri="https://llm4eln.semos.dev/id/"),
            SPARQLPrefix(prefix="Property", uri="https://llm4eln.semos.dev/id/Property-3A"),
            SPARQLPrefix(prefix="Category", uri="https://llm4eln.semos.dev/id/Category-3A"),
            SPARQLPrefix(prefix="Item", uri="https://llm4eln.semos.dev/id/Item-3A"),
        ],
        properties=retrieve_properties(endpoint_config, limit=200)["predicates"],
        categories=retrieve_categories(endpoint_config, limit=200)["categories"],
        data_types=[],
    )

    deps = SPARQLDependencies(
        schema=llm4eln_schema,
        endpoint_config=endpoint_config,
    )

    result = await sparql_agent.run(
        "Search for Jane Doe",
        deps=deps,
    )

    print(f"Explanation: {result.output.explanation}\n")
    print("📋 Generated SPARQL Query:\n" + "-" * 80 + "\n" + result.output.query + "\n" + "-" * 80)
    print("\nExecuting query...")
    df = execute_sparql_query(deps.endpoint_config, result.output.query)

    if not df.empty:
        print(f"✅ Query returned {len(df)} rows\n")
        print(df.head(3).to_string(index=False))
    else:
        print("⚠️  Query returned no results")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
