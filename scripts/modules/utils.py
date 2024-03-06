import json

def get_parameter(parameter_name):
    with open("./parameters.json", "r") as file:
        parameters = json.loads(file.read())
        return parameters[parameter_name]