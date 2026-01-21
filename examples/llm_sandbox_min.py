from llm_sandbox import SandboxSession

# Create and use a sandbox session
with SandboxSession(lang="python") as session:
    result = session.run("""
print("Hello from LLM Sandbox!")
print("I'm running in a secure container.")
    """)
    print(result.stdout)
