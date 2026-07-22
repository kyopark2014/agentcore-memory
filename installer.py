#!/usr/bin/env python3
"""
AgentCore Memory Installer
- Creates IAM role for AgentCore Memory
- Creates AgentCore Memory
- Generates application/config.json
"""

import boto3
import json
import os
import time
import logging
import argparse
from typing import Dict, Optional
from botocore.exceptions import ClientError
from bedrock_agentcore.memory import MemoryClient

project_name = "agentcore-memory"
region = "us-west-2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_aws_clients(region_name: str):
    sts = boto3.client("sts", region_name=region_name)
    iam = boto3.client("iam", region_name=region_name)
    account_id = sts.get_caller_identity()["Account"]
    return iam, account_id


def create_iam_role(iam_client, role_name: str, assume_role_policy: Dict) -> str:
    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description=f"Role for {role_name}",
        )
        role_arn = response["Role"]["Arn"]
        logger.info(f"IAM role created: {role_name}")
        return role_arn

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            logger.warning(f"IAM role already exists: {role_name}")
            response = iam_client.get_role(RoleName=role_name)
            return response["Role"]["Arn"]
        logger.error(f"Failed to create IAM role {role_name}: {e}")
        raise


def attach_inline_policy(iam_client, role_name: str, policy_name: str, policy_document: Dict):
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document),
        )
        logger.info(f"Inline policy '{policy_name}' attached to {role_name}")
    except ClientError as e:
        logger.error(f"Error attaching policy {policy_name}: {e}")
        raise


USER_PREFERENCE_PROMPT = (
    "You are tasked with analyzing conversations to extract the user's preferences. You'll be analyzing two sets of data:\n"
    "<past_conversation>\n"
    "[Past conversations between the user and system will be placed here for context]\n"
    "</past_conversation>\n"
    "<current_conversation>\n"
    "[The current conversation between the user and system will be placed here]\n"
    "</current_conversation>\n"
    "Your job is to identify and categorize the user's preferences into two main types:\n"
    "- Explicit preferences: Directly stated preferences by the user.\n"
    "- Implicit preferences: Inferred from patterns, repeated inquiries, or contextual clues. Take a close look at user's request for implicit preferences.\n"
    "For explicit preference, extract only preference that the user has explicitly shared. Do not infer user's preference.\n"
    "For implicit preference, it is allowed to infer user's preference, but only the ones with strong signals, such as requesting something multiple times.\n"
    "Use Korean.\n"
)


SUMMARY_PROMPT = (
    "You will be given a text block and a list of summaries you previously generated when available.\n"
    "<task>\n"
    "- When the previously generated is not available, your goal is to summarize the given text block.\n"
    "- When there is existing summary, your goal is to extend summary by taking into account the given text block.\n"
    "- If there are queries/topics specified in the text block, your generated summary need to cover those queries/topics.\n"
    "- If there are instructions in the text block **guiding you how to generate summary**, you MUST follow them.\n"
    "</task>\n"
    "Use Korean.\n"
)

SEMANTIC_PROMPT = (
    "You are a long-term memory extraction agent supporting a lifelong learning system.\n"
    "Your task is to identify and extract meaningful information about the users from a given list of messages.\n"
    "Analyze the conversation and extract structured information about the user according to the schema below.\n"
    "Only include details that are explicitly stated or can be logically inferred from the conversation.\n"
    "- Extract information ONLY from the user messages. You should use assistant messages only as supporting context.\n"
    "- If the conversation contains no relevant or noteworthy information, return an empty list.\n"
    "- Do NOT extract anything from prior conversation history, even if provided. Use it solely for context.\n"
    "- Do NOT incorporate external knowledge.\n"
    "- Avoid duplicate extractions.\n"
    "Use Korean.\n"
)

SEMANTIC_CONSOLIDATION_PROMPT = (
    "You consolidate newly extracted facts with existing long-term semantic memories.\n"
    "- Merge duplicates; keep the most specific and recent facts.\n"
    "- Do not invent facts that were not extracted.\n"
    "- Prefer clear, atomic statements in Korean.\n"
    "Use Korean.\n"
)

MEMORY_EXTRACTION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _shared_memory_strategies() -> list:
    """UserPreference + Summary + Semantic (one of each kind per memory_id)."""
    return [
        {
            "customMemoryStrategy": {
                "name": "UserPreference",
                "namespaces": ["/users/{actorId}/preferences"],
                "configuration": {
                    "userPreferenceOverride": {
                        "extraction": {
                            "modelId": MEMORY_EXTRACTION_MODEL_ID,
                            "appendToPrompt": USER_PREFERENCE_PROMPT,
                        }
                    }
                },
            }
        },
        {
            "customMemoryStrategy": {
                "name": "Summary",
                "namespaces": ["/users/{actorId}/sessions/{sessionId}"],
                "configuration": {
                    "summaryOverride": {
                        "consolidation": {
                            "modelId": MEMORY_EXTRACTION_MODEL_ID,
                            "appendToPrompt": SUMMARY_PROMPT,
                        }
                    }
                },
            }
        },
        {
            "customMemoryStrategy": {
                "name": "Semantic",
                "namespaces": ["/users/{actorId}/facts"],
                "configuration": {
                    "semanticOverride": {
                        "extraction": {
                            "modelId": MEMORY_EXTRACTION_MODEL_ID,
                            "appendToPrompt": SEMANTIC_PROMPT,
                        },
                        "consolidation": {
                            "modelId": MEMORY_EXTRACTION_MODEL_ID,
                            "appendToPrompt": SEMANTIC_CONSOLIDATION_PROMPT,
                        },
                    }
                },
            }
        },
    ]


def create_agentcore_memory_role(iam_client, proj_name: str, rgn: str) -> str:
    """Create AgentCore Memory IAM role."""
    logger.info("[2/10] Creating AgentCore Memory IAM role")
    account_id = boto3.client("sts", region_name=rgn).get_caller_identity()["Account"]
    role_name = f"role-agentcore-memory-for-{proj_name}-{rgn}"

    # Trust must include aws:SourceAccount / aws:SourceArn; CreateMemory rejects otherwise.
    # https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/long-term-configuring-custom-strategies.html
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "MemoryAssumeRolePolicy",
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock-agentcore.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": account_id
                    },
                    "ArnLike": {
                        "aws:SourceArn": (
                            f"arn:aws:bedrock-agentcore:{rgn}:{account_id}:*"
                        )
                    },
                },
            }
        ],
    }

    role_arn = create_iam_role(iam_client, role_name, assume_role_policy)

    memory_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/*",
                    "arn:aws:bedrock:*:*:inference-profile/*",
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": account_id
                    }
                },
            }
        ],
    }
    attach_inline_policy(
        iam_client, role_name, f"agentcore-memory-policy-for-{proj_name}", memory_policy
    )

    # IAM eventual consistency: CreateMemory validates trust immediately after role create/update
    logger.info("  Waiting for IAM role trust policy to propagate...")
    time.sleep(10)

    return role_arn


def create_agentcore_memory(role_arn: str, user_id: str = "installer") -> str:
    """
    Create AgentCore Memory with shared UserPreference / Summary / Semantic strategies.

    user_id is unused for strategy naming — kept for call-site compatibility.
    User isolation uses {actorId}/{sessionId} namespace templates from CreateEvent.
    """
    logger.info("[2/10] Creating AgentCore Memory")

    memory_client = MemoryClient(region_name=region)
    memory_name = project_name.replace("-", "_")

    memories = memory_client.list_memories()
    for memory in memories:
        if memory.get("id", "").split("-")[0] == memory_name:
            memory_id = memory.get("id")
            logger.info(f"  Memory already exists: {memory_id}")
            return memory_id

    strategies = _shared_memory_strategies()
    result = memory_client.create_memory_and_wait(
        name=memory_name,
        description=f"Memory for {project_name}",
        event_expiry_days=365,
        strategies=strategies,
        memory_execution_role_arn=role_arn,
    )
    memory_id = result.get("id")
    names = [s["customMemoryStrategy"]["name"] for s in strategies]
    logger.info(f"  ✓ Memory created: {memory_id} (strategies={names})")
    return memory_id

def save_config(config_path: str, config_data: Dict):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    existing = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                existing = json.load(f)
            logger.info(f"Loaded existing {config_path}")
        except Exception as e:
            logger.warning(f"Could not read existing {config_path}: {e}")

    existing.update(config_data)

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info(f"Config saved to {config_path}")


def main():
    global project_name, region

    parser = argparse.ArgumentParser(description="AgentCore Memory Installer")
    parser.add_argument("--project-name", default=project_name, help="Project name (default: %(default)s)")
    parser.add_argument("--region", default=region, help="AWS region (default: %(default)s)")
    args = parser.parse_args()

    proj_name = args.project_name
    rgn = args.region

    project_name = proj_name
    region = rgn

    logger.info("=" * 60)
    logger.info("AgentCore Memory Installer")
    logger.info("=" * 60)
    logger.info(f"Project: {proj_name}")
    logger.info(f"Region:  {rgn}")
    logger.info("=" * 60)

    iam_client, account_id = get_aws_clients(rgn)
    logger.info(f"Account ID: {account_id}")

    # 1. Create AgentCore Memory IAM Role
    agentcore_memory_role_arn = create_agentcore_memory_role(iam_client, proj_name, rgn)
    logger.info(f"AgentCore Memory Role ARN: {agentcore_memory_role_arn}")

    # 2. Create AgentCore Memory
    memory_id = create_agentcore_memory(agentcore_memory_role_arn)
    logger.info(f"Memory ID: {memory_id}")

    # 3. Generate application/config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "application", "config.json")

    config_data = {
        "projectName": proj_name,
        "accountId": account_id,
        "region": rgn,
        "agentcore_memory_role": agentcore_memory_role_arn,
        "memory_id": memory_id,
    }
    save_config(config_path, config_data)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Installation Completed Successfully!")
    logger.info("=" * 60)
    logger.info(f"  AgentCore Memory Role: {agentcore_memory_role_arn}")
    logger.info(f"  Memory ID: {memory_id}")
    logger.info(f"  Config: {config_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
