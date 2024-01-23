import torch
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, StableDiffusionPipeline, DEISMultistepScheduler
from compel import Compel, ReturnedEmbeddingsType
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from threading import Thread
import threading
import gradio as gr
import os
import weakref
from PIL import Image 
import time
from helperFunctions import *
    
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequences):
        self.eos_sequences = eos_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -max(len(seq) for seq in self.eos_sequences):].tolist()
        return any(any(seq == last_ids[i][-len(seq):] for i in range(len(last_ids))) for seq in self.eos_sequences)

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)
    
################TODO######################
# add dynamic prompt and chat template support per model.
# look into SD prompt models
        # some RP models do not listen to instructions
        # requiring another model to create the image prompt
# look into TTS(is it possible to stream output into TTS model?)
    # and what will TTS vram peak usage be?
# allow uploading character files and system/instruct prompts from UI.
# add support for different quant methods (GGUF/bitsandbytes/exllama(v2)/GPTQ)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NeuralChat Chat template is correct if i understand it
# the BOS token should be added before system prompt
# the PAD token(PAD is set to equal EOS token) after the end of every sequence?
        # almost certainly wrong, EOS token should only be placed if the 
        # interraction has come to an end entirely 
        # possibly used on scene changes? 
# sequence being a (user input, assistant response) pair
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   HelperFunction:
#       updateCharacter:
#           add support for chatML prompt format and others?
#           make instruction prompt toggleable in case of instruction present in the charcter_sheet

def initializePaths():
    base_path = os.path.dirname(__file__)
    if not os.path.exists(base_path+"/LLM/"):
        os.mkdir(base_path+"/LLM/")
    if not os.path.exists(base_path+"/SD/"):
        os.mkdir(base_path+"/SD/")
        os.mkdir(base_path+"/SD/models/")
        os.mkdir(base_path+"/SD/lora/")
        os.mkdir(base_path+"/SD/nonXL/")
        os.mkdir(base_path+"/SD/upscaler/")
    if not os.path.exists(base_path+"/InOut/"):
        os.mkdir(base_path+"/InOut/")
        os.mkdir(base_path+"/InOut/users/")
        os.mkdir(base_path+"/InOut/characters")
        os.mkdir(base_path+"/InOut/prompts/")
        os.mkdir(base_path+"/InOut/images/")
        os.mkdir(base_path+"/InOut/prompts/instructions/")
        os.mkdir(base_path+"/InOut/prompts/JailBreakprompt/")
        os.mkdir(base_path+"/InOut/prompts/SD/")
        os.mkdir(base_path+"/InOut/prompts/system/")
    return base_path+"/LLM/", base_path+"/SD/", base_path+"/InOut/"

LLM_path, SD_path, IO_path = initializePaths() 
SD_model_path = SD_path + "models/"
currentLLM = None
currentSD = None
LLMmodel = None
tokenizer = None
LLMstreamer = None
SDstreamer = None
pipe = None
compel = None 
token_id_tuple_list = []

# special token_id_tuple_list could be used to aid model agnostic templating down the line
def LLMLoadModel(LLMmodel_name):
    global LLM_path, currentLLM, LLMmodel, tokenizer, LLMstreamer, SDstreamer
    global LLMmodelRef, tokenizerRef, LLMstreamerRef, SDstreamerRef, token_id_tuple_list
    if currentLLM is None or currentLLM != LLMmodel_name:
        del LLMmodel, tokenizer, LLMstreamer, SDstreamer
        torch.cuda.empty_cache()
        currentLLM = LLMmodel_name
        print(f"cache cleared and currentLLM = {currentLLM}")
        
    tokenizer = AutoTokenizer.from_pretrained(LLM_path + LLMmodel_name, trust_remote_code=False, 
                                        token=True, local_files_only=True
                                        )
    # special_token_list = tokenizer.all_special_tokens
    # token_special_ids = tokenizer.all_special_ids
    # for i in token_special_ids:
    #     token_id_tuple_list.append((token_special_ids[i], special_token_list[i]))
    # token_id_tuple_list.sort()
    # print(f"token_id_tuple_list = {token_id_tuple_list}")
    LLMmodel = AutoModelForCausalLM.from_pretrained(LLM_path + LLMmodel_name,
                                            device_map="auto",
                                            trust_remote_code=True
                                            )
    LLMstreamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_token=False, clean_up_tokenization_spaces=True, timeout=10)
    SDstreamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_token=False, clean_up_tokenization_spaces=True, timeout=10)
    LLMmodelRef = weakref.proxy(LLMmodel)
    tokenizerRef = weakref.proxy(tokenizer)
    LLMstreamerRef = weakref.proxy(LLMstreamer)
    SDstreamerRef = weakref.proxy(SDstreamer)
    # SpecialTokenNameIDList = weakref.proxy(token_id_tuple_list)
    return LLMmodel_name

def LLMTokenize(tokenizer, prompt):
    prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    return prompt

def createStoppingList(stopString):
    global tokenizerRef
    NLstopString = "\n"+stopString
    NLstopTokens = LLMTokenize(tokenizerRef, NLstopString)[:, 2:]
    stopTokens = LLMTokenize(tokenizerRef, stopString)
    my_eos_sequences = [NLstopTokens[0].tolist(), stopTokens[0].tolist()]
    stopping_criteria = EosListStoppingCriteria(my_eos_sequences)
    stopping_criteria_list = StoppingCriteriaList()
    stopping_criteria_list.append(stopping_criteria)
    return stopping_criteria_list

def LLMDetokenize(tokenizer, tokens):
    return tokenizer.decode(tokens[0])

def LLMSetKwargs(tokens, streamer, max_new_tokens, do_sample, temperature, 
                eta_cutoff, repetition_penalty, stop_string,
                exponential_decay_length_penalty=None):
    stopping_criteria_list = createStoppingList(stop_string)
    kwargs = dict(inputs=tokens, streamer=streamer, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature, 
                eta_cutoff=eta_cutoff, repetition_penalty=repetition_penalty,
                stopping_criteria=stopping_criteria_list,
                exponential_decay_length_penalty=exponential_decay_length_penalty)
    return kwargs

def SDTemplateTokenizer(tokenizer, system_prompt, history, char_desc, incl_char):
    prompt = ""
    if incl_char==True:
        prompt += char_desc
    if isinstance(history, str):
        prompt += history
    if isinstance(history, list):
        conversation = ""
        for input, output in history:
            conversation += input + "\n" + output
        prompt += conversation
    
    in_prompt = f'''<s>
### Instruction:
{system_prompt}
### Input:
{prompt}
### Response:
'''
    # print(f"\n<<TO BE SUMMARIZED FOR SD PROMPT>>\n\n{in_prompt}\n\n<<END OF TO BE SUMMARIZED FOR SD PROMPT>>")
    return LLMTokenize(tokenizer, in_prompt)

def ChatTemplateTokenizer(system_prompt, current_msg, history, tokenizer, char_name, user_name, stop_string):
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in history:
        user=user.strip()
        assistant=assistant.strip()
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant.removesuffix(stop_string)}])
    conversation.append({"role": "user", "content": current_msg})
    templatedConv = tokenizer.apply_chat_template(conversation, return_tensors="pt", tokenize=False)
    inToks = LLMTokenize(tokenizer, replaceCharName(templatedConv, char_name, user_name))
    # print(f"\n<THIS IS WHAT WE SEND TO THE MODEL>\n{LLMDetokenize(tokenizer, inToks)}\n<THIS IS THE END OF WHAT WE SEND TO THE MODEL>\n")
    return inToks

def SDLoadModel(model_name):
    global SD_model_path, currentSD, pipe, compel, SDmodel_size
    global pipeRef, compelRef
    if currentSD is None or currentSD != model_name:
        SDmodel_size=os.path.getsize(SD_model_path+model_name)
        del pipe, compel
        torch.cuda.empty_cache()
        currentSD = model_name
        print(f"cache cleared and currentSD = {currentSD}")
    if SDmodel_size > 6000000000:
        pipe = StableDiffusionXLPipeline.from_single_file(SD_model_path + model_name, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True])
        compelRef = weakref.proxy(compel)
    if SDmodel_size < 6000000000:
        pipe = StableDiffusionPipeline.from_single_file(SD_model_path + model_name, torch_dtype=torch.float16, variant="fp16",
                    load_safety_checker=False)
        # pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, solver_oder=2)
        pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipe.enable_vae_tiling()
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipeRef = weakref.proxy(pipe)
    return model_name

def SDXLGenerate(prompt, steps, width, height, cfg):
    global pipeRef, compelRef
    generator  = torch.manual_seed(torch.seed())
    conditioning, pooled = compelRef(prompt)
    image = pipeRef(prompt_embeds=conditioning, pooled_prompt_embeds=pooled,
                num_inference_steps=steps, guidance_scale=cfg,
                width=width, height=height, generator=generator).images[0]
    curTime = str(time.strftime("%Y-%j-%H:%M:%S",time.localtime(time.time())))
    image.save(fp=IO_path + "images/" + curTime +".jpg")
    print(f"maximum GB used: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())}")
    return image

def SDGenerate(prompt, steps, width, height, cfg):
    global pipeRef
    generator = torch.manual_seed(torch.seed())
    max_length = pipeRef.tokenizer.model_max_length
    input_ids = pipeRef.tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    negative_ids = pipeRef.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
    negative_ids = negative_ids.to("cuda")
    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipeRef.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeRef.text_encoder(negative_ids[:, i: i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    image = pipeRef(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                    generator=generator, num_inference_steps=steps,
                    width=width, height=height, guidance_scale=cfg).images[0] 
    
    curTime = str(time.strftime("%Y-%j-%H:%M:%S",time.localtime(time.time()))) 
    image.save(fp=IO_path + "images/" + curTime +".jpg")
    print(f"maximum GB used: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())}")
    return image

def Chat(current_msg, history, max_tokens, temp, repPen, etaCut, system_prompt, char_name, user_name, stop_string):
    global LLMmodelRef, tokenizerRef, LLMstreamerRef
    inToks = ChatTemplateTokenizer(system_prompt, current_msg, history, tokenizerRef, char_name, user_name, stop_string)
    KWargs = LLMSetKwargs(inToks, LLMstreamerRef, max_tokens, True, temp, etaCut, repPen, stop_string)
    LLMThread = ThreadWithResult(target=LLMmodelRef.generate, kwargs=KWargs)
    LLMThread.start()
    outputs = []
    for text in LLMstreamerRef:
        text=text.replace("</s>","")
        outputs.append(text)
        yield "".join(outputs).replace(stop_string,"")
    LLMThread.join()
    print(f"maximum GB used: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())}")
    # torch.set_printoptions(profile="full")
    # print(LLMThread.result)
    # print(LLMDetokenize(tokenizer, LLMThread.result))

def SDGenInput(history, SD_instruct_prompt, incl_char, char_description, stop_string):
    global LLMmodelRef, tokenizerRef, SDstreamerRef

    inToks = SDTemplateTokenizer(tokenizerRef, SD_instruct_prompt, history, char_description, incl_char)

    KWargs = LLMSetKwargs(inToks, SDstreamerRef, 300, True, 0.3, 0.001, 1.01, stop_string=stop_string, exponential_decay_length_penalty=(200, 1.05))

    SDsummaryThread = Thread(target=LLMmodelRef.generate, kwargs=KWargs)
    SDsummaryThread.start()
    
    PromptSummary = ""
    for i in SDstreamerRef:
        PromptSummary+=i
    PromptSummary=PromptSummary.replace(stop_string, "")
    PromptSummary=PromptSummary.replace("</s>", "")
    SDsummaryThread.join()
    return PromptSummary

def Pic(history, steps, width, height, cfg, direct_mode, incl_hist, SD_instruct_prompt, incl_char, char_description, stop_string):
    global SDmodel_size
    # print(f"SDmodel size = {SDmodel_size}")
    if direct_mode == True and SDmodel_size > 6000000000:
        # print(history[-1][1])
        return(SDXLGenerate(history[-1][1], steps, width, height, cfg))
    if direct_mode == True and SDmodel_size < 6000000000:
        # print(history[-1][1])
        return(SDGenerate(history[-1][1], steps, width, height, cfg))
    if incl_hist == True:
        PromptSummary = SDGenInput(history, SD_instruct_prompt, incl_char, char_description, stop_string)
    else:  
        PromptSummary = SDGenInput(history[-1][1], SD_instruct_prompt, incl_char, char_description, stop_string)
    # print(f"\n SUMMARY: {PromptSummary}\n")
    if SDmodel_size > 6000000000:
        picture = SDXLGenerate(PromptSummary, steps, width, height, cfg)
    if SDmodel_size < 6000000000:
        picture = SDGenerate(PromptSummary, steps, width, height, cfg)

    return picture

# set paths to model files, create (name, path) tuple list for gradio dropdown
LLMList = initializeLLMList(LLM_path)
SDList = initializeSDList(SD_model_path)

for i in LLMList:
    print(f"Langue model: {i}")
for i in SDList:
    print(f"SDmodel: {i}")

currentLLM = LLMLoadModel(LLMList[2])
currentSD = SDLoadModel(SDList[3])
SDmodel_size=os.path.getsize(SD_model_path+currentSD)

SDPromptList = initializeSDPromptList(IO_path)
SysPromptList = initializeSysPromptList(IO_path)
JailBreakPromptList = initializeJailBreakPromptList(IO_path)
InstructionPromptList = initializeInstructPromptList(IO_path)
CharList = initializeCharList(IO_path)
UserList = initializeUserList(IO_path)

system_prompt = SysPromptList[0]
SD_instruct_prompt = SDPromptList[0]
instruct_prompt = InstructionPromptList[0]
user_file = UserList[1]
user_name, user_description, user_personality = updateUser(user_file, IO_path)
char_file = CharList[0]
greeting_list = initializeFirstMsg(char_file, IO_path)
char_name, char_description = updateCharacter(char_file, 
                                greeting_list[0] ,user_name, user_personality, 
                                user_description, instruct_prompt, True, IO_path)

generateTab, LLMparamTab, SDparamTab, CharTab = gr.Blocks(), gr.Blocks(), gr.Blocks(), gr.Blocks()
chat_bot = gr.Chatbot(height=500)
typingBox= gr.Textbox(lines=5, autofocus=True)
b = gr.Button(value="Send", variant="primary",scale=0,min_width=50)

with CharTab:
    with gr.Group():
        with gr.Group():
            InstructDropdown=gr.Dropdown(choices=InstructionPromptList, label="select instruction", value=InstructionPromptList[0], show_label=True)
            refreshInstructions=gr.Button("refresh instruction")
        with gr.Group():
            UserDropdown=gr.Dropdown(label="select user", choices=UserList, value=UserList[1])
            refreshUserButton=gr.Button("refresh User list")
            UserNameBox=gr.Textbox(label="what's your name", value=user_name)
            UserPersonalityBox=gr.Textbox(label="user's personality", value=user_personality)
            userDescripionBox=gr.Textbox(label="user's description", value=user_description)
            includeUserDescCheckbox=gr.Checkbox(label="include user description and personality in the prompt", value=False)
        with gr.Group():
            CharDropdown=gr.Dropdown(choices=CharList, label="select character", value=CharList[0],show_label=True)
            refreshCharButton=gr.Button("refresh Char list")
            CharNameBox=gr.Textbox(label='char name', value=char_name, interactive=False)
            CharGreetingDropdown=gr.Dropdown(choices=greeting_list, label="the first message", value=greeting_list[0],show_label=True)
            CharDescBox=gr.Textbox(label="char description", value=char_description, lines=25, max_lines=40, interactive=False)
            CharUpdateButton=gr.Button("update character selection")
    IOPathBox=gr.Textbox(label="the path to your characters, prompts and where the images are saved",value=IO_path)
        
with SDparamTab:
    with gr.Group():
        selectSDDropdown=gr.Dropdown(choices=SDList, label="select SD model", value=currentSD, show_label=True)
        refreshSDModelList = gr.Button("refresh SD list")
    with gr.Group():
        widthSlider=gr.Slider(label="width",minimum=512, maximum=2048,value=680, step=8)
        heightSlider=gr.Slider(label="height",minimum=512, maximum=2048,value=1024, step=8)
        stepsSlider=gr.Slider(label="steps",minimum=2, maximum=50,value=10, step=1)
        cfgSlider=gr.Slider(label="cfg",minimum=0.0, maximum=10.0,value=0.0, step=0.1)
        incl_histCheckbox=gr.Checkbox(label="include chat history for image generation", value=True)
        incl_charCheckbox=gr.Checkbox(label="include the character description for image generation", value=True)
    with gr.Group("a prompt for summarizing the text and extracting details relevant for image generation"):
        SDSystemPromptBox=gr.Textbox(label="SD prompt", value=SD_instruct_prompt, lines=8, max_lines=8, interactive=True)
        ChangeSDPromptButton=gr.Button("submit prompt")
        SDpromptDropdown=gr.Dropdown(choices=SDPromptList, value=SDPromptList[0], show_label=True)
        refreshSDprompt=gr.Button("refresh SD prompt list")
    SDPathBox=gr.Textbox(label="the path where your SD files are located", value=SD_path)

with LLMparamTab:  
    with gr.Group():
        with gr.Group():
            selectLLMDropdown = gr.Dropdown(choices=LLMList, label="select LLM model", value=currentLLM)
            refreshLLMbutton = gr.Button("refresh LLM list")
        maxTokenSlider = gr.Slider(label="max new tokens",minimum=50, maximum=1024,value=512, step=1)
        tempSlider = gr.Slider(label="temperature",minimum=0, maximum=1.5,value=0.7, step=0.05)
        repPenSlider = gr.Slider(label="repetition penalty",minimum=1.0, maximum=2.0,value=1.05, step=0.05)
        etaCutSlider = gr.Slider(label="eta_cutoff",minimum=0.0001, maximum=0.005, value=0.0005, step=0.0001)
        stopStringBox=gr.Textbox(label="string to stop generation once it's been generated", value="###")
    with gr.Group():
        systemPromptBox=gr.Textbox(label="sytem prompt", value=system_prompt, lines=20, max_lines=20)
        submitSysPromptButton=gr.Button("submit system prompt")
        systemPromptDropdown=gr.Dropdown(label="System Prompt",choices=SysPromptList, value=SysPromptList[0],show_label=True)
        refreshSysPromptButton=gr.Button("refresh SysPrompt List")
        JailBreakPromptDropdown=gr.Dropdown(label="JailBreak/Jailbreak prompt", choices=JailBreakPromptList, value=JailBreakPromptList[0],show_label=True)
        JailBreakCheckBox=gr.Checkbox(label="add JailBreak/Jailbreak prompt to the system prompt", value=False)
        AddCharCheckBox=gr.Checkbox(label="add character description to prompt", value=True)
        addSysPromptCheckBox=gr.Checkbox(label="check this box if you want to add a system prompt", value=False)
    LLMPathBox=gr.Textbox(label="the path where your LLM files are located", value=LLM_path)

with generateTab:   
    with gr.Row():
        with gr.Column("generate image of output text"):
            directModeCheckBox=gr.Checkbox(label="apply SD prompt to directly generate images")
            chat_interface = gr.ChatInterface(fn=Chat,chatbot=chat_bot, submit_btn=b, textbox=typingBox, additional_inputs=[
                maxTokenSlider, tempSlider, repPenSlider, etaCutSlider, systemPromptBox, CharNameBox, UserNameBox, stopStringBox])
            genBut = gr.Button("generate image of output text") 
        with gr.Column():
            img = gr.Image(interactive=False, width=720, height=920) 

with gr.Blocks() as demo:
    with gr.Tab("generate"): 
        generateTab.render()
    with gr.Tab("LLM paramaters"):     
        LLMparamTab.render()
    with gr.Tab("SD paramater"):
        SDparamTab.render()
    with gr.Tab("CharList"):
        CharTab.render()


    UserDropdown.change(fn=updateUser, 
        inputs=[UserDropdown, IOPathBox],
        outputs=[UserNameBox, userDescripionBox, UserPersonalityBox]
            ).then(fn=updateCharacter, 
                inputs=[CharDropdown, CharGreetingDropdown, UserNameBox, UserPersonalityBox, userDescripionBox, InstructDropdown, includeUserDescCheckbox, IOPathBox], 
                outputs=[CharNameBox, CharDescBox]
                    ).then(fn=updateSystemPrompt, 
                        inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
                        outputs=systemPromptBox
                            ).then(fn=resetChatBot, 
                                inputs=CharDropdown, 
                                outputs=[chat_bot, chat_interface.chatbot_state]
                                )
    InstructDropdown.change(fn=updateCharacter, 
        inputs=[CharDropdown, CharGreetingDropdown, UserNameBox, UserPersonalityBox, userDescripionBox, InstructDropdown, includeUserDescCheckbox, IOPathBox], 
        outputs=[CharNameBox, CharDescBox]
            ).then(fn=updateSystemPrompt, 
                inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
                outputs=systemPromptBox
                    ).then(fn=resetChatBot, 
                        inputs=CharDropdown, 
                        outputs=[chat_bot, chat_interface.chatbot_state]
                        )  
    CharGreetingDropdown.change(fn=updateCharacter, 
        inputs=[CharDropdown, CharGreetingDropdown, UserNameBox, UserPersonalityBox, userDescripionBox, InstructDropdown, includeUserDescCheckbox, IOPathBox], 
        outputs=[CharNameBox, CharDescBox]
            ).then(fn=updateSystemPrompt, 
                inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
                outputs=systemPromptBox
                    ).then(fn=resetChatBot, 
                        inputs=CharDropdown, 
                        outputs=[chat_bot, chat_interface.chatbot_state]
                        )  
    CharDropdown.change(fn=updateFirstMsgList,
        inputs=[CharDropdown, IOPathBox],
        outputs=CharGreetingDropdown
            ).then(fn=updateCharacter, 
                inputs=[CharDropdown, CharGreetingDropdown, UserNameBox, UserPersonalityBox, userDescripionBox, InstructDropdown, includeUserDescCheckbox, IOPathBox], 
                outputs=[CharNameBox, CharDescBox]
                ).then(fn=updateSystemPrompt, 
                    inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
                    outputs=systemPromptBox
                        ).then(fn=resetChatBot, 
                            inputs=CharDropdown, 
                            outputs=[chat_bot, chat_interface.chatbot_state]
                            )  
    CharUpdateButton.click(fn=updateCharacter, 
        inputs=[CharDropdown, CharGreetingDropdown, UserNameBox, UserPersonalityBox, userDescripionBox, InstructDropdown, includeUserDescCheckbox, IOPathBox], 
        outputs=[CharNameBox, CharDescBox]
            ).then(fn=updateSystemPrompt, 
                inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
                outputs=systemPromptBox
                    ).then(fn=resetChatBot, 
                        inputs=CharDropdown, 
                        outputs=[chat_bot, chat_interface.chatbot_state]
                        )
    refreshInstructions.click(fn=updateInstructPromptList, 
        inputs=IOPathBox, 
        outputs=InstructDropdown
        )
    refreshCharButton.click(fn=updateCharList, 
        inputs=IOPathBox, 
        outputs=CharDropdown
        )
    refreshUserButton.click(fn=updateUserList, 
        inputs=IOPathBox, 
        outputs=UserDropdown
        )

    SDpromptDropdown.change(fn=updateSDinstructPrompt, 
        inputs=[SDpromptDropdown, CharNameBox, UserNameBox], 
        outputs=SDSystemPromptBox
        )
    refreshSDprompt.click(fn=updateSDPromptList, 
        inputs=IOPathBox, 
        outputs=SDpromptDropdown
        )
    ChangeSDPromptButton.click(fn=updateSDinstructPrompt, 
        inputs=[SDSystemPromptBox, CharNameBox, UserNameBox], 
        outputs=SDSystemPromptBox
        )

    JailBreakPromptDropdown.change(fn=updateSystemPrompt, 
        inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
        outputs=systemPromptBox
        )
    systemPromptDropdown.change(fn=updateSystemPrompt, 
        inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
        outputs=systemPromptBox
        )
    refreshSysPromptButton.click(fn=updateSystemPromptList, 
        inputs=IOPathBox, 
        outputs=systemPromptDropdown
        )
    submitSysPromptButton.click(fn=resetChatBot, 
        inputs=submitSysPromptButton, 
        outputs=[chat_bot, chat_interface.chatbot_state]
            ).then(fn=updateSystemPrompt, 
                inputs=[systemPromptDropdown, JailBreakPromptDropdown, addSysPromptCheckBox, AddCharCheckBox, JailBreakCheckBox, CharDescBox, CharNameBox, UserNameBox], 
                outputs=systemPromptBox
                )

    genBut.click(fn=Pic, 
        inputs=[chat_interface.chatbot, stepsSlider, widthSlider, heightSlider, cfgSlider, directModeCheckBox, incl_histCheckbox, SDSystemPromptBox, incl_charCheckbox, CharDescBox, stopStringBox], 
        outputs=img
        )
    
    selectLLMDropdown.change(fn=resetChatBot, 
        inputs=selectLLMDropdown, 
        outputs=[chat_bot, chat_interface.chatbot_state]
            ).then(fn=LLMLoadModel, 
                inputs=selectLLMDropdown, 
                outputs=selectLLMDropdown
                )
    refreshLLMbutton.click(fn=updateLLMmodelList, 
        inputs=LLMPathBox, 
        outputs=selectLLMDropdown
        )

    selectSDDropdown.change(fn=SDLoadModel, 
        inputs=selectSDDropdown, 
        outputs=selectSDDropdown
        )
    refreshSDModelList.click(fn=updateSDmodelList, 
        inputs=SDPathBox, 
        outputs=selectSDDropdown
        )


demo.queue().launch()
