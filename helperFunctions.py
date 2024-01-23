import os
import json
import gradio as gr


def replaceCharName(input_text :str, char_name :str, user_name):
    input_text = input_text.replace("{{user}}", user_name)
    input_text = input_text.replace("{{char}}", char_name)
    return input_text

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def updateSystemPrompt(system_prompt:str, JailBreak_prompt:str, add_system_prompt, add_char_desc, add_jailbreak, char_description:str, char_name, user_name):
    # temp_description = char_description.split('\n')
    # char_description = "".join(temp_description)
    # char_description = char_description.strip()
    JailBreak_prompt = JailBreak_prompt.strip()
    system_prompt = system_prompt.strip()
    char_description = char_description.strip()
    startPrompt = ""
    if add_system_prompt:
        startPrompt += system_prompt
        if add_jailbreak:
            startPrompt += "\n" + JailBreak_prompt
        if add_char_desc:
            startPrompt += '\n\n' + char_description
        return replaceCharName(startPrompt, char_name, user_name)

    if not add_system_prompt:
        if add_jailbreak:
            startPrompt += JailBreak_prompt
            if add_char_desc:
                startPrompt += '\n\n' + char_description
            return replaceCharName(startPrompt, char_name, user_name)
        if add_char_desc:
            startPrompt+=char_description
            return replaceCharName(startPrompt, char_name, user_name)

def updateSDinstructPrompt(SD_instruct_prompt:str, char_name, user_name):
    return replaceCharName(SD_instruct_prompt, char_name, user_name)
    
def UpdateBoolean(input):
    if input == True:
        return False
    else:
        return True

def updateUser(fileName, IO_path):
    with open(IO_path+"users/"+fileName) as json_file:
        user_sheet = json.load(json_file)
        userDescriptionbool = False
        userNamebool = False
    if 'name' in user_sheet and user_sheet['name'] != "":
        userNamebool = True
        userName = user_sheet['name']
    if 'char_name' in user_sheet and user_sheet['char_name'] != "" and userNamebool == False:
        userName = user_sheet['char_name']
    if 'description' in user_sheet and user_sheet['description'] != "":
        userDescriptionbool = True
        userDescripion =  user_sheet['description'] + '\n'
    if 'char_persona' in user_sheet and userDescriptionbool == False and user_sheet['char_persona'] != "":
        userDescripion = user_sheet['char_persona'] + '\n'
    if 'personality' in user_sheet and user_sheet['personality'] != "":
        userPersonality = user_sheet['personality'] + '\n'
    userPersonality = userPersonality.replace("{{char}}", userName)
    userDescripion = userDescripion.replace("{{char}}", userName)
    userPersonality = userPersonality.replace("{{user}}","{{char}}")
    userDescripion = userDescripion.replace("{{user}}","{{char}}")
    return userName, userDescripion, userPersonality

#   add support for chatML prompt format
#   make instruction prompt toggleable in case of instruction present in the charcter_sheet
def updateCharacter(fileName, firstMsg, userName, userPersonality, userDescription, instructPrompt, includeUser,IO_path):
    char_desc = "### Instruction:\n" + instructPrompt + "\n\n### Character sheet:\n"
    worldInforAfter = False
    charDescription = False
    example_dialogue = False
    scenario = False
    charNamebool = False
    first_mes = False
    with open(IO_path+"characters/"+fileName) as json_file:
        char_sheet = json.load(json_file) 
    if 'wiBefore' in char_sheet and char_sheet['wiBefore'] != "":
        char_desc += "world info: " + char_sheet['wiBefore'] + '\n'
    if 'name' in char_sheet and char_sheet['name'] != "":
        charNamebool = True
        charName = char_sheet['name']
    if 'char_name' in char_sheet and char_sheet['char_name'] != "" and charNamebool == False:
        charName = char_sheet['char_name']
    if 'description' in char_sheet and char_sheet['description'] != "":
        charDescription = True
        char_desc += "description of {{char}}: " + char_sheet['description'] + '\n'
    if 'char_persona' in char_sheet and charDescription == False and char_sheet['char_persona'] != "":
        char_desc += "description of {{char}}: " + char_sheet['char_persona'] + '\n'
    if 'personality' in char_sheet and char_sheet['personality'] != "":
        char_desc += "personality of {{char}}: " + char_sheet['personality'] + '\n'
    if 'userName' in char_sheet and char_sheet['userName'] != "" and userName=="":
        userName = char_sheet['userName']
    if includeUser == True and ('userDescription' in char_sheet and char_sheet['userDescription'] != ""):
        char_desc += "description of {{user}}: " + char_sheet['userDescription'] + '\n'
    if includeUser == True and ('userDescription' not in char_sheet and userDescription != ""):
        char_desc += "description of {{user}}: " + userDescription + '\n'
    if includeUser == True and ('userPersonality' in char_sheet and char_sheet['userPersonality'] != ""):
        char_desc += "personality of {{user}}: " + char_sheet['userPersonality'] + '\n'
    if includeUser == True and ('userPersonality' not in char_sheet and userPersonality != ""):
        char_desc += "personality of {{user}}: " + userPersonality + '\n'
    if 'example_dialogue' in char_sheet and char_sheet['example_dialogue'] != "":
        example_dialogue = True
        char_desc += char_sheet['example_dialogue'].replace('<START>', '### Example:')
    if 'mes_example' in char_sheet and example_dialogue == False and char_sheet['mes_example'] != "":
        char_desc += char_sheet['mes_example'].replace('<START>', '### Example:')
    if 'scenario' in char_sheet and char_sheet['scenario'] != "":
        scenario = True
        char_desc += "scenario: " + char_sheet['scenario'] + '\n'
    if 'world_scenario' in char_sheet and scenario is False and char_sheet['world_scenario'] != "":
        char_desc += "scenario: " + char_sheet['world_scenario'] + '\n'
    if 'wiAfter' in char_sheet and char_sheet['wiAfter'] != "":#should be implemented in the chat function
        worldInforAfter = True
        char_desc += char_sheet['wiAfter'] + '\n'
    # wiAfter/post_history seem to be creator's notes injected at intervals to mainting character coherency.
    # should not be implemented here :)     
    if 'post_history_instructions' in char_sheet and worldInforAfter == False and char_sheet['post_history_instructions'] != "":#should be implemented in the chat function
        char_desc += char_sheet['post_history_instructions']
    char_desc += "Taking the above information into consideration, you must engage in a roleplay conversation with {{user}} below this line. Do not write {{user}}'s dialogue lines in your responses.\n" 
    if firstMsg != "":
        char_desc+= "### {{char}}:\n" + firstMsg + "\n\n"
    return charName, replaceCharName(char_desc, charName, userName)

def updateGreeting(greeting):
    return greeting

def updateLLMinstructPrompt(SD_instruct_prompt:str):
    return SD_instruct_prompt#replaceCharName(SD_instruct_prompt, char_name, user_name)

def resetChatBot(z):
    return [], []

def updateFirstMsgList(fileName, IO_path):
    greetingList = []
    greetingList.append('')
    with open(IO_path+"characters/"+fileName) as json_file:
        char_sheet = json.load(json_file)
    if char_sheet['first_mes']:
        greetingList.append(char_sheet['first_mes'])
    if len(greetingList) == 0 and char_sheet['char_greeting']:
        greetingList.append(char_sheet['char_greeting'])
    if char_sheet['alternate_greetings']:
        for i in char_sheet['alternate_greetings']:
            greetingList.append(i)
    return gr.update(choices=greetingList)

def updateLLMmodelList(LLM_path):    
    return gr.update(choices=[f.name for f in os.scandir(LLM_path) if f.is_dir()])

def updateSDmodelList(SD_path):  
    SD_model_path = SD_path+"models/"
    return gr.update(choices=[f for f in os.listdir(SD_model_path) if ".safetensors" in f])

def updateSystemPromptList(IO_path): 
    sys_prompt_path = IO_path+"prompts/system/"   
    return gr.update(choices=[open(f, encoding="utf-8").read() for f in os.scandir(sys_prompt_path)])

def updateSDPromptList(IO_path):
    SD_prompt_path = IO_path+"prompts/SD/"
    return gr.update(choices=[open(f, encoding="utf-8").read() for f in os.scandir(SD_prompt_path)])

def updateCharList(IO_path):
    char_path = IO_path+"characters/"
    return gr.update([f for f in os.listdir(char_path) if ".json" in f])

def updateUserList(IO_path):
    user_path = IO_path+"users/"
    return gr.update([f for f in os.listdir(user_path) if ".json" in f])

def updateInstructPromptList(IO_path):
    instruct_path = IO_path+"prompts/instructions/"
    return gr.update(choices=[open(f, encoding="utf-8").read() for f in os.scandir(instruct_path)])

def initializeFirstMsg(fileName, IO_path):
    greetingList = []
    greetingList.append('')
    with open(IO_path+"characters/"+fileName) as json_file:
        char_sheet = json.load(json_file)
    if char_sheet['first_mes']:
        greetingList.append(char_sheet['first_mes'])
    if len(greetingList) == 0 and char_sheet['char_greeting']:
        greetingList.append(char_sheet['char_greeting'])
    if char_sheet['alternate_greetings']:
        for i in char_sheet['alternate_greetings']:
            greetingList.append(i)
    return greetingList

def initializeJailBreakPromptList(IO_path):
    return [open(f, encoding="utf-8").read() for f in os.scandir(IO_path+"prompts/JailBreakPrompt/")]

def initializeCharList(IO_path):
    return [f for f in os.listdir(IO_path+"characters/") if ".json" in f]

def initializeUserList(IO_path):
    return [f for f in os.listdir(IO_path+"users/") if ".json" in f]

def initializeLLMList(baseLLM_path):
    return [f.name for f in os.scandir(baseLLM_path) if f.is_dir()]

def initializeSDList(baseSD_path):
    return [f for f in os.listdir(baseSD_path) if ".safetensors" in f]

def initializeSDPromptList(IO_path):
    return [open(f, encoding="utf-8").read() for f in os.scandir(IO_path+"prompts/SD/")]

def initializeSysPromptList(IO_path):
    return [open(f, encoding="utf-8").read() for f in os.scandir(IO_path+"prompts/system/")]

def initializeInstructPromptList(IO_path):
    return [open(f, encoding="utf-8").read() for f in os.scandir(IO_path+"prompts/instructions/")]


