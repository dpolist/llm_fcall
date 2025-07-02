import ast
import json

class FCall:
    tools = []
    
    # Given a function call in a string, parse the string and securely execute the function
    def __parse_and_call(self, function_call: str) -> any:
        # Parse the string as Python code
        tree = ast.parse(function_call, mode="eval")

        # Validate it's a function call
        if not (isinstance(tree, ast.Expression) and isinstance(tree.body, ast.Call)):
            raise ValueError("Invalid function call syntax")

        # Extract function name
        if isinstance(tree.body.func, ast.Name):
            func_name = tree.body.func.id
        else:
            raise ValueError("Only simple function names are allowed")

        # Ensure the function is allowed
        allowed_func_names = [func.__name__ for func in self.tools]
        if func_name not in allowed_func_names:
            raise ValueError(f"Function '{func_name}' not registered")

        # Find the function object
        func = next(func for func in self.tools if func.__name__ == func_name)

        # Extract arguments
        args = [ast.literal_eval(arg) for arg in tree.body.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}

        # Call the function with the extracted arguments
        result = func(*args, **kwargs)
        return result
    
    def invoke_bedrock_client_and_tools (self,bedrock_client,modelId,request):
        """
            Invoke a bedrock client and execute tools, if any
        """
        docs = self.__get_tools_docs()
        content = request["messages"][len(request["messages"])-1]["content"]
        prompt = f"""
        Considering the available functions below and only the functions below:\n{docs}
        Generate a single function call in Python that supports part of the folowing prompt (do not add any explanation, simply show the function call in Python):
        {content}
        If there are no avaliable functions to support the prompt, return simply 'not_found()'
        """
        new_request = {
            "max_tokens": request["max_tokens"],
            "messages":[{"role":"user","content":prompt}],
            "anthropic_version": request["anthropic_version"]
        }

        response = bedrock_client.invoke_model(
            modelId=modelId,
            body = json.dumps(new_request),
            contentType='application/json'
        ) 

        response = json.loads(response['body'].read().decode('utf-8'))
        fcall =response ["content"][0]["text"]
        if fcall == "not_found()":
            result = None
        else:
            result = self.__parse_and_call(fcall)
            user_message = dict(request["messages"][len(request["messages"])-1])
            prompt= f"""
                Considering, despite the information being right or wrong, that {fcall} = {result}:\n
                Simply answer without any comment regarding the fact the it's right or wrong:\n
            """+user_message["content"]

            new_request = {
                "max_tokens": request["max_tokens"],
                "messages":[{"role":"user","content":prompt}],
                "anthropic_version": request["anthropic_version"]
            }

            response = bedrock_client.invoke_model(
                modelId=modelId,
                body = json.dumps(new_request),
                contentType='application/json'
            ) 

            response = json.loads(response['body'].read().decode('utf-8'))

        return {
            "invoke_response":response,
            "tool_info": {
                "tool_used": fcall,
                "result" : result
            }
        }
    
    def __get_tools_docs(self):
        return "\n".join(['Function "' + tool.__name__ +'":\n'+tool.__doc__ for tool in self.tools])

    