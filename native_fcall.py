# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to use tools with the Converse API and the Cohere Command R model.
"""

import logging
import json
import boto3


from botocore.exceptions import ClientError



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc(a,b: float) -> float:
    """Calculates the sum of two numbers.
    Args:
        a (float): first operand.
        b (float): second operand.

    Returns:
        response (float): the sum of a and b.
    """

    return a + b


def generate_text(bedrock_client, model_id, tool_config, input_text):
    """Generates text using the supplied Amazon Bedrock model. If necessary,
    the function handles tool use requests and sends the result to the model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The Amazon Bedrock model ID.
        tool_config (dict): The tool configuration.
        input_text (str): The input text.
    Returns:
        Nothing.
    """

    logger.info("Generating text with model %s", model_id)

   # Create the initial message from the user input.
    messages = [{
        "role": "user",
        "content": [{"text": input_text}]
    }]

    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        toolConfig=tool_config
    )

    output_message = response['output']['message']
    messages.append(output_message)
    stop_reason = response['stopReason']

    if stop_reason == 'tool_use':
        # Tool use requested. Call the tool and send the result to the model.
        tool_requests = response['output']['message']['content']
        for tool_request in tool_requests:
            if 'toolUse' in tool_request:
                tool = tool_request['toolUse']
                logger.info("Requesting tool %s. Request: %s",
                            tool['name'], tool['toolUseId'])

                if tool['name'] == 'calc':
                    tool_result = {}
                    result = calc(tool['input']['a'],tool['input']['b'])
                    tool_result = {
                        "toolUseId": tool['toolUseId'],
                        "content": [{"json": {"operation": result}}]
                    }
                    logger.info(f"Tool use id {tool['toolUseId']} returned {result}")
                    tool_result_message = {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": tool_result

                            }
                        ]
                    }
                    messages.append(tool_result_message)

                    # Send the tool result to the model.
                    response = bedrock_client.converse(
                        modelId=model_id,
                        messages=messages,
                        toolConfig=tool_config
                    )
                    output_message = response['output']['message']

    # print the final response from the model.
    for content in output_message['content']:
        print(json.dumps(content, indent=4))


def main():
    """
    Entrypoint for tool use example.
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    input_text = "Após uma operação matématica com 3 e 2, diga se o resultado é par ou ímpar"

    tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "calc",
                "description": "Exceute a math operation using two numbers.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "first operand."
                            },
                            "b": {
                                "type": "number",
                                "description": "second operand."
                            }
                        },
                        "required": [
                            "a","b"
                        ]
                    }
                }
            }
        }
    ]
}
    bedrock_client = boto3.client(service_name='bedrock-runtime')


    try:
        print(f"Question: {input_text}")
        generate_text(bedrock_client, model_id, tool_config, input_text)

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(
            f"Finished generating text with model {model_id}.")


if __name__ == "__main__":
    main()
