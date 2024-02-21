def format_prompt(caption,output=None):
    intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    intro_data = """### Instruction:\nCategorize relation between the image and caption into one of the 6 categories:
            \n\nTrue NEWS\nSatire\nMisleading Content\nManipulated Content\nFalse Connection\nImposter Content\n\n"""
    _input = f"USER: <image>\n{caption}"
    dummy = ""
    _output = f"ASSISTANT:\n{output if output else dummy}"
    
    return "\n\n".join([intro,intro_data,_input,_output])

