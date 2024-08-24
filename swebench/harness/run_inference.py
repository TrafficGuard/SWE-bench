from swebench.harness.docker_build import build_container, setup_logger, close_logger, build_base_images, build_env_images, get_test_specs_from_dataset
from swebench.harness.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset, str2bool
from typing import Tuple, List
from swebench.harness.docker_utils import (cleanup_container, copy_to_container, exec_run_with_timeout)
import docker
from pathlib import Path
import logging
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm

RUN_INFERENCE_LOG_DIR = Path("logs/run_inference")

# Inference is generating a solution for the TestSpec

def get_test_spec(instance_id: str, dataset_name: str, split: str) -> object:
    dataset = load_swebench_dataset(dataset_name, split)
    instance = next((i for i in dataset if i['instance_id'] == instance_id), None)
    if not instance:
        raise ValueError(f"Instance {instance_id} not found in dataset")
    return make_test_spec(instance)

def setup_container_for_inference(instance_id: str, test_spec: object, run_id: str):
    # Set up Docker client
    client = docker.from_env()

    # Set up logging
    log_dir = RUN_INFERENCE_LOG_DIR / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(instance_id, log_dir / "setup_container.log")

    try:
        # Build base and environment images
        build_base_images(client, [test_spec])
        build_env_images(client, [test_spec])

        # Build and start the container
        container = build_container(test_spec, client, run_id, logger, nocache=True, force_rebuild=True)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")
        # At this point, the container is running with the appropriate environment set up

        # Configure Nous
        copy_to_container(container, Path("~/local.env"), Path("/nous/variables/local.env"))
        # Get the latest version
        container.exec_run("git pull", workdir="/nous").output.decode("utf-8").strip()

        return container
    except Exception as e:
        logger.error(f"Error setting up container for {instance_id}: {e}")
        raise
    finally:
        close_logger(logger)

# Usage example (commented out)
# container = setup_container_for_inference("your_instance_id", "princeton-nlp/SWE-bench", "test", "your_run_id")

# When you're done with the container (commented out)
# cleanup_container(docker.from_env(), container, setup_logger("cleanup", "/path/to/cleanup.log"))



def run_inference(instance_id: str, dataset_name: str, split: str, run_id: str):
    # Get test_spec
    test_spec = get_test_spec(instance_id, dataset_name, split)
    
    # Build base and environment images
    client = docker.from_env()
    build_base_images(client, [test_spec])
    build_env_images(client, [test_spec])
    
    container = setup_container_for_inference(instance_id, test_spec, run_id)
    try:
        # Run inference
        output = container.exec_run(f"npm run swebench --fs=/testbed {instance_id}", workdir="/nous").output.decode("utf-8").strip()
        
        # Get git diff
        git_diff = container.exec_run(f"git diff HEAD {test_spec.base_commit}", workdir="/testbed").output.decode("utf-8").strip()
        
        return instance_id, git_diff
    finally:
        # Clean up the container
        cleanup_logger = setup_logger(f"cleanup_{instance_id}", RUN_INFERENCE_LOG_DIR / run_id / instance_id / "cleanup.log")
        cleanup_container(client, container, cleanup_logger)
        close_logger(cleanup_logger)

def main(
        dataset_name: str,
        split: str,
        instance_ids: List[str],
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
#        open_file_limit: int,
        run_id: str,
        timeout: int,
    ):
    client = docker.from_env()
    
    # Load dataset
    full_dataset = load_swebench_dataset(dataset_name, split)
    
    # Filter dataset based on instance_ids if provided
    if instance_ids:
        dataset = [i for i in full_dataset if i['instance_id'] in instance_ids]
    else:
        dataset = full_dataset

    # Create TestSpec objects for the filtered dataset
    test_specs = get_test_specs_from_dataset(dataset)
    
    # Build base images only for the filtered dataset
    build_base_images(client, test_specs, force_rebuild)
    
    # Build environment images only for the filtered dataset
    build_env_images(client, test_specs, force_rebuild, max_workers)
    
    # Set up the thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each instance
        futures = {executor.submit(run_inference, instance['instance_id'], dataset_name, split, run_id): instance['instance_id'] for instance in dataset}
        
        # Process the results as they complete
        results = []
        with tqdm(total=len(dataset), desc="Running inference") as pbar:
            for future in as_completed(futures):
                instance_id, git_diff = future.result()
                results.append({"instance_id": instance_id, "model_patch": git_diff})
                pbar.update(1)
        
        # Write the results to the predictions file
        predictions_path = f"{run_id}_predictions.jsonl"
        with open(predictions_path, 'w') as f:
            for result in results:
                json.dump(result, f)
                f.write('\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Verified", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
        )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
