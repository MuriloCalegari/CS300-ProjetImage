import json

def get_parameters():
    with open("./parameters.json", "r") as file:
        return json.loads(file.read())

def get_parameter(parameter_name):
    with open("./parameters.json", "r") as file:
        parameters = json.loads(file.read())
        return parameters[parameter_name]