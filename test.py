import unittest
import boto3

from  llm_fcall import FCall

client = boto3.client("bedrock-runtime", region_name="us-east-1")

class Test_FCall (unittest.TestCase):
    @staticmethod
    def add_wrong_math (a,b: float) -> float:
        """
        Soma dois numeros

        Args:
            a: primeiro numero
            b: segundo numero
        """
        return a+b+1

    def test_invoke_llm_success (self):
        fcall = FCall()
        fcall.tools = [self.add_wrong_math]

        request = {
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": """
                          Encontre a soma entre 6 e 12 e diga se é par ou impar
                          """}],
            "anthropic_version": "bedrock-2023-05-31"
        }

        result = fcall.invoke_bedrock_client_and_tools(client,"anthropic.claude-3-5-sonnet-20240620-v1:0",request)
        self.assertIsNotNone(result["tool_info"])
        self.assertIsNotNone(result["invoke_response"])
        self.assertTrue(result["tool_info"]["result"] %2 != 0)
        content = result["invoke_response"]["content"][0]["text"]
        print (content)
        self.assertTrue("ímpar" in content)

    def test_invoke_llm_fail (self):
        fcall = FCall()
        fcall.tools = [self.add_wrong_math]

        request = {
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": """
                          Encontre a divisão entre 12 e 6
                          Depois junte com a palavra banana
                          """}],
            "anthropic_version": "bedrock-2023-05-31"
        }

        result = fcall.invoke_bedrock_client_and_tools(client,"anthropic.claude-3-5-sonnet-20240620-v1:0",request)
        self.assertIsNone(result["tool_info"]["result"])
        
    def test_tools_docs (self):
        fcall = FCall()
        fcall.tools = [self.add_wrong_math]

        result = fcall._FCall__get_tools_docs()

        self.assertIsNotNone(result)
        
    def test_fcall_valid (self):
        fcall = FCall()
        fcall.tools = [self.add_wrong_math]

        result = fcall._FCall__parse_and_call("add_wrong_math(2,3)")
        self.assertEquals(result, 6)

    def test_fcalls_invalid (self):
        fcall = FCall()
        fcall.tools = [self.add_wrong_math]

        with self.assertRaises(Exception) as ex:
            fcall._FCall__parse_and_call("multiply(2,3)")


