"""Simple demonstration of robust copy functions.

Source: https://github.com/vndee/llm-sandbox/blob/9304f109e18ab596e2f5ce662b69efbbd7a98e17/examples/copy_demo.py

This script shows practical examples of:
- Copying files and directories to containers
- Extracting results from containers
- Consistent behavior across backends
- Error handling and robustness features

Usage:
    python examples/copy_demo.py [backend]

    backend: docker, podman, kubernetes (default: docker)
"""

import logging
from pathlib import Path

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
LOCAL_SANDBOX_DIR = DATA_DIR / "sandbox"
LOCAL_SANDBOX_INPUT_DIR = LOCAL_SANDBOX_DIR / "input"
LOCAL_SANDBOX_OUTPUT_DIR = LOCAL_SANDBOX_DIR / "output"

example_csv_path = LOCAL_SANDBOX_INPUT_DIR / "temp.csv"


def run_demo(backend_name: str = "docker") -> None:
    """Run the copy functions demonstration."""
    logger.info("🚀 Copy Functions Demo - %s Backend", backend_name.upper())
    logger.info("=" * 50)

    logger.info("📦 Creating %s session...", backend_name)

    try:
        with SandboxSession(
            lang="python",
            image="python:3.10-alpine",
            backend=SandboxBackend(backend_name),
            verbose=True,
            libraries=["pandas", "matplotlib"],
        ) as session:
            logger.info("✅ %s session ready", backend_name)

            # Demo 1: Copy Python script to container
            logger.info("\n📁 Step 1: Copying Python script to container")
            session.copy_to_runtime(src=str(LOCAL_SANDBOX_DIR / "plot_csv.py"), dest="/sandbox/plot_csv.py")
            logger.info("✅ Script copied successfully")

            # Demo 2: Copy input directory to container
            logger.info("\n📁 Step 2: Copying input data directory to container")
            session.copy_to_runtime(src=str(LOCAL_SANDBOX_INPUT_DIR), dest="/sandbox/input")
            logger.info("✅ Input data copied successfully")

            # Demo 3: Copy single config file
            logger.info("\n📁 Step 3: Copying configuration file")
            session.copy_to_runtime(src=str(LOCAL_SANDBOX_DIR / "config.txt"), dest="/sandbox/config.txt")
            logger.info("✅ Config file copied successfully")

            # Demo 4: Verify files are in container
            logger.info("\n🔍 Step 4: Verifying files in container")
            result = session.execute_command("ls -lah /sandbox")
            logger.info("📋 Files in container:")
            for line in result.stdout.strip().split("\n"):
                if line:
                    logger.info("   %s", line)

            # Demo 5: Execute the processing script
            logger.info("\n🏃 Step 5: Running data processing script")
            # Ensure output directory exists
            session.execute_command("mkdir -p /sandbox/output")
            result = session.execute_command("/sandbox/.sandbox-venv/bin/python /sandbox/plot_csv.py")

            logger.info("📤 Script output:")
            logger.info(result.stdout)

            # Demo 6: Copy results back from container
            logger.info("\n📁 Step 6: Copying results back to host")
            output_dir = LOCAL_SANDBOX_OUTPUT_DIR

            # Copy the entire output directory
            session.copy_from_runtime(src="/sandbox/output", dest=str(output_dir))
            logger.info("✅ Results copied back successfully")

            # Demo 8: Error handling demonstration
            logger.info("\n🛡️  Step 8: Demonstrating error handling")
            try:
                session.copy_to_runtime("/nonexistent/file.txt", "/sandbox/dummy.txt")
                logger.info("   ❌ Expected this to fail!")
            except FileNotFoundError:
                logger.info("   ✅ Correctly handled error")

            logger.info("\n🎉 Demo completed successfully with %s backend!", backend_name)
            logger.info("   All copy operations worked robustly and consistently.")

    except Exception:
        logger.exception("❌ Demo failed")
        raise


def main() -> None:
    """Execute main function."""
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "docker"

    logger.info("🧪 LLM Sandbox Copy Functions Demo")
    logger.info("==================================")
    logger.info("This demo shows how to reliably copy files and directories")
    logger.info("between your host system and sandbox containers.\n")

    try:
        run_demo(backend)

        logger.info("\n💡 Key Features Demonstrated:")
        logger.info("   • File and directory copying (both directions)")
        logger.info("   • Robust error handling")
        logger.info("   • Consistent behavior across backends")
        logger.info("   • Real-world data processing workflow")
        logger.info("   • Safe path handling")

    except Exception:
        logger.exception("❌ Demo failed")
        logger.info("💡 Make sure the selected backend is available and running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
