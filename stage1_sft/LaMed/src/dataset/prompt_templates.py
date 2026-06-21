Caption_templates = [
            "Can you provide a caption consists of findings for this medical image?",
            "Describe the findings of the medical image you see.",
            "Please caption this medical scan with findings.",
            "What is the findings of this image?",
            "Describe this medical scan with findings.",
            "Please write a caption consists of findings for this image.",
            "Can you summarize with findings the images presented?",
            "Please caption this scan with findings.",
            "Please provide a caption consists of findings for this medical image.",
            "Can you provide a summary consists of findings of this radiograph?",
            "What are the findings presented in this medical scan?",
            "Please write a caption consists of findings for this scan.",
            "Can you provide a description consists of findings of this medical scan?",
            "Please caption this medical scan with findings.",
            "Can you provide a caption consists of findings for this medical scan?",
            "Please generate a medical report based on this image.",
            "Can you generate a diagnose report from this image.",
            "Could you analyze and provide a caption for the findings in this medical image?",
            "Please describe the observations depicted in this medical scan.",
            "Can you summarize the findings of this image in a caption?",
            "What are the significant findings in this medical image?",
            "Please provide a detailed caption outlining the findings of this image.",
            "Could you interpret and describe the findings shown in this medical scan?",
            "What conclusions can you draw from the observations in this image?",
            "Please write a descriptive caption based on the findings in this scan.",
            "What key findings can you identify from examining this medical image?",
            "Could you generate a detailed report based on the observations in this image?",
            "Can you provide a diagnosis based on the findings in this image?",
            "Please generate a comprehensive report summarizing the findings in this image.",
            "Caption the findings in this medical image?",
            "Describe the findings you see.",
            "Caption this medical scan's findings.",
            "What are the findings here?",
            "Describe these findings.",
            "Summarize the findings in these images.",
            "Caption this scan's findings.",
            "Provide a caption for this medical image's findings.",
            "Summarize the findings of this radiograph.",
            "What findings are presented in this scan?",
            "Describe this scan's findings.",
            "Generate a medical report based on this image.",
            "Can you provide a diagnosis based on this image?",
]


import random

# 角色设定
REPORT_ROLES = [
    "You are a medical AI assistant that analyzes CT scans and generates radiology reports.",
    "You are an AI radiologist designed to describe CT scan to radiology reports.",
    "As an expert system in radiology, your job is to interpret CT images into structured reports.",
    "You specialize in transforming visual scan data into medically meaningful reports.",
]

# 任务指令（findings）
FINDING_INSTRUCTIONS = [
    "Provide detailed imaging observations enclosed in <finding>...</finding>.",
    "Describe all visible findings precisely within the <finding>...</finding> section.",
    "Report objective and detailed CT scan findings inside <finding>...</finding> tags.",
    "List all visual abnormalities and relevant features within a <finding>...</finding> block.",
    "Document clear and comprehensive radiological observations using <finding>...</finding> tags.",
]


# 任务指令（impressions）
IMPRESSION_INSTRUCTIONS = [
    "Provide concise diagnostic impressions enclosed in <impression>...</impression>.",
    "Summarize your clinical conclusions in the <impression>...</impression> section.",
    "Write a clear and focused impression using <impression>...</impression> tags.",
    "Express your diagnostic reasoning briefly within a <impression>...</impression> block.",
    "Present the overall interpretation and clinical takeaway inside <impression>...</impression> tags.",
]


# 保底指令（逻辑性、结构性）
GENERAL_REQUIREMENTS = [
    "Ensure your response is medically accurate, logically consistent, and well-structured.",
    "The report should be coherent, clinically valid, and follow standard formatting.",
    "Be concise, medically precise, and maintain a clear structure.",
    "Output must be medically sound, organized, and aligned with radiology reporting norms.",
]

def generate_report_prompt(finding_list=None, impression_list=None):
    role = random.choice(REPORT_ROLES)
    lines = []

    if finding_list:
        lines.append(random.choice(FINDING_INSTRUCTIONS))
    if impression_list:
        lines.append(random.choice(IMPRESSION_INSTRUCTIONS))
    
    lines.append(random.choice(GENERAL_REQUIREMENTS))

    # 按编号排序
    task_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    return f"{role}\n{task_block}"


# Caption_templates = [
#     "Can you generate a medical report based on this image?",
#     "Please write a medical report for this scan.",
#     "Create a full report based on this medical image.",
#     "What does this medical image show? Please provide a report.",
#     "Write a diagnostic report for this image.",
#     "Please provide a complete report based on this scan.",
#     "Generate a detailed medical report from this image.",
#     "What is your interpretation of this medical scan?",
#     "Could you analyze this image and write a report?",
#     "Examine the image and produce a medical report.",
#     "Write a clinical report based on this radiograph.",
#     "Please create a report for this medical scan.",
#     "Can you produce a diagnostic report for this image?",
#     "Evaluate this scan and provide a report.",
#     "What would a radiologist report based on this image?",
#     "Generate a radiological report from this scan.",
#     "Please interpret this scan and write a report.",
#     "Based on this image, write a complete report.",
#     "Can you describe this image in a medical report?",
#     "Please generate a clinical report for this scan."
# ]

Finding_templates = [
    "Can you describe the observable findings in this image?",
    "Please write the radiological findings for this scan.",
    "Create a findings section based on this medical image.",
    "What visible findings can be observed in this scan?",
    "Write the objective findings observed in this image.",
    "Please provide only the radiological findings from this scan.",
    "Generate a detailed findings section from this image.",
    "What are the visible features or abnormalities in this scan?",
    "Could you analyze this image and write only the findings?",
    "Examine the image and produce the findings section.",
    "Write the findings based on this radiograph.",
    "Please describe the anatomical and pathological findings seen in this scan.",
    "Can you produce only the findings section for this image?",
    "Evaluate this scan and provide the observable findings.",
    "What would a radiologist describe as findings from this image?",
    "Generate radiological findings from this scan.",
    "Please interpret this scan and write the findings.",
    "Based on this image, write only the radiological findings.",
    "Can you describe the visual findings in this medical image?",
    "Please generate the objective findings for this scan."
]


Impression_template = [
    "What is the diagnosis based on this CT scan?",
    "Based on this scan, what is the most likely diagnosis?",
    "What condition does this CT scan indicate?",
    "Please provide a diagnosis for the observed abnormalities.",
    "What disease is suggested by this CT scan?",
    "Can you determine the diagnosis from this scan?",
    "Identify the most probable diagnosis given this scan.",
    "What pathology is visible in this CT image?",
    "What is the clinical interpretation of this CT scan?",
    "What diagnosis can be made from these findings?",
    "What is the underlying abnormality shown in this scan?",
    "From the CT scan, what condition can be diagnosed?",
    "What medical condition is evident in this image?",
    "Interpret the scan and provide a diagnosis.",
    "What do the radiological findings suggest?",
    "Please identify the disease based on this scan.",
    "According to the findings, what is the likely diagnosis?",
    "What does this CT scan reveal diagnostically?",
    "Provide a diagnostic assessment for this scan.",
    "What is the diagnostic impression from this image?"
]



MMVQA_roles = [
    "You are a radiology expert.",
    "You are a medical imaging specialist.",
    "As a chest CT interpretation expert, your role is to assist diagnosis.",
    "You specialize in interpreting CT scans for diagnostic purposes.",
    "As an expert in chest imaging, you are tasked with providing accurate answers.",
    "You are a consultant radiologist focusing on thoracic imaging.",
    "You are a diagnostic imaging expert working on clinical case analysis."
]

MMVQA_task_descriptions = [
    "Analyze the CT scan carefully and answer the multiple-choice question based on visible findings.",
    "Solve the multiple-choice question by interpreting only the visual information from the chest CT.",
    "Determine which option best matches the scan findings after careful observation.",
    "Review the chest CT image and select the answer that most accurately reflects the visible abnormalities.",
    "Evaluate the CT scan findings and choose the best matching option among those provided.",
    "Identify the correct answer based solely on what can be observed in the CT scan.",
    "Carefully inspect the CT scan and determine the choice that aligns best with your observations.",
    "Use only the visible evidence in the CT scan to answer the multiple-choice question.",
]

MMVQA_think_options = [
    "Optionally, explain your reasoning process inside a <think>...</think> tag before finalizing your answer.",
    "You should document your thought process in a <think>...</think> tag.",
    "Provide the reasoning, which should be enclosed in a <think>...</think> block.",
    "Before deciding, you may reason through your observations in a <think>...</think> section.",
    "You can first summarize your analysis inside a <think>...</think> block.",
    "Think through the options carefully and record your logic in a <think>...</think> block if appropriate.",
    "Provide your thought process optionally in a <think>...</think> tag.",
    "Explain your reasoning step-by-step in a <think>...</think> tag."
]



MMVQA_answer_requirement = [
    "Provide your final answer by stating the option letter (A, B, C, or D) directly.",
    "Your final choice should be expressed clearly using only the option letter (A, B, C, or D).",
    "Output the final answer as a single letter: A, B, C, or D.",
    "State your final answer by choosing one option: A, B, C, or D.",
    "Indicate the selected answer with only the letter (A, B, C, or D).",
    "Finalize your response by writing just the correct option letter: A, B, C, or D.",
    "Clearly state your selected choice as one of the letters: A, B, C, or D.",
    "Select the best matching option and output only its letter.",
    "Write the final answer as a single uppercase letter corresponding to your choice.",
    "Give your answer by selecting and stating A, B, C, or D, without additional tags or formatting."
]



import random

def generate_mmvaq_prompt(use_think=True):
    role = random.choice(MMVQA_roles)
    task = random.choice(MMVQA_task_descriptions)
    answer_instruction = random.choice(MMVQA_answer_requirement)
    
    if use_think:
        think_instruction = random.choice(MMVQA_think_options)
        prompt = f"{role} {task} {think_instruction} {answer_instruction}"
    else:
        prompt = f"{role} {task} {answer_instruction}"
    
    return prompt


# MMVQA_answer_requirement = [
#     "Provide your final answer inside an <answer>...</answer> block using the option letter (A, B, C, or D).",
#     "Your final choice must be enclosed within an <answer>...</answer> tag, stating only the option letter (A, B, C, or D).",
#     "Output the final answer using an <answer>...</answer> tag, indicating the option letter (A, B, C, or D).",
#     "State your final answer inside <answer>...</answer>, selecting from options A, B, C, or D.",
#     "Wrap the selected answer option (A, B, C, or D) within an <answer>...</answer> tag.",
#     "Finalize your response by outputting only the option letter (A, B, C, or D) inside an <answer>...</answer> block.",
#     "Provide the selected option inside an <answer>...</answer> tag.",
#     "Clearly state your selected choice, using <answer>...</answer> to enclose only the final answer letter.",
#     "Record your final selection by placing the corresponding letter (A, B, C, or D) within an <answer>...</answer> block.",
#     "Choose the best matching option and present it inside an <answer>...</answer> tag."
#     "Select the best option and output the final answer in an <answer>...</answer> tag using the option letter (e.g., A, B, C, or D)."
# ]



YESNO_roles = [
    "You are an expert in medical imaging.",
    "You are a specialist in chest CT interpretation.",
    "You are a radiologist tasked with analyzing CT scans for diagnostic purposes.",
    "As a thoracic imaging expert, your role is to assist in binary decision-making based on CT findings.",
    "You specialize in evaluating chest CT images to answer Yes/No diagnostic questions.",
    "You are a clinical imaging consultant focusing on thoracic abnormalities."
]


YESNO_task_descriptions = [
    "Based on the chest CT scan, answer the Yes/No question strictly using visual evidence.",
    "Analyze the CT image to determine the correct Yes or No answer to the given question.",
    "Examine the CT scan carefully and conclude with a Yes or No based only on what you observe.",
    "Your task is to assess the chest CT and respond to the Yes/No question using only visible features.",
    "Interpret the CT scan and judge whether the given statement is supported by the image.",
    "Evaluate the CT image and provide a definitive Yes or No based on observed findings.",
    "Review the CT scan findings and answer the Yes/No question without introducing external information."
]

YESNO_think_options = [
    "Detail your reasoning inside a <think>...</think> block before providing your final answer.",
    "Optionally, explain your thought process in a <think>...</think> section before concluding.",
    "You may document your visual analysis in a <think>...</think> tag prior to stating Yes or No.",
    "Think through the evidence carefully and, if needed, record your logic within a <think>...</think> block.",
    "Before deciding, you can reason through the visual findings inside a <think>...</think> section.",
    "Summarize your observational reasoning inside a <think>...</think> block.",
    "Provide your reasoning step-by-step inside a <think>...</think> tag."

]

# YESNO_answer_requirement = [
#     "Provide your final answer clearly inside an <answer>Yes</answer> or <answer>No</answer> block.",
#     "Conclude by stating your answer using <answer>Yes</answer> or <answer>No</answer> tags.",
#     "Output your final decision by enclosing Yes or No within an <answer>...</answer> tag.",
#     "Record your answer inside an <answer>Yes</answer> or <answer>No</answer> block based on your conclusion.",
#     "Respond with either <answer>Yes</answer> or <answer>No</answer>, enclosing your choice appropriately.",
#     "Output your answer in an <answer>Yes</answer> or <answer>No</answer> format."
# ]

YESNO_answer_requirement = [
    "Provide your final answer clearly as either Yes or No.",
    "Conclude by stating your answer with a simple Yes or No.",
    "Output your final decision directly: Yes or No.",
    "Respond with either Yes or No based on your reasoning.",
    "State your final answer plainly as Yes or No, without extra formatting.",
    "Give your answer by choosing one word: Yes or No."
]


def generate_yesno_prompt(use_think=True):
    role = random.choice(YESNO_roles)
    task = random.choice(YESNO_task_descriptions)
    answer_instruction = random.choice(YESNO_answer_requirement)
    
    if use_think:
        think_instruction = random.choice(YESNO_think_options)
        prompt = f"{role} {task} {think_instruction} {answer_instruction}"
    else:
        prompt = f"{role} {task} {answer_instruction}"
    
    return prompt


REGION_roles = [
    "You are an expert in thoracic imaging analysis.",
    "You are a radiology specialist focusing on regional abnormality detection.",
    "You are a chest CT imaging consultant tasked with assessing localized regions.",
    "As a medical imaging expert, your role is to evaluate specific anatomical areas in CT scans.",
    "You specialize in interpreting localized findings in chest CT imaging.",
    "You are a diagnostic radiologist analyzing regional areas for abnormalities."
]


REGION_task_descriptions = [
    "Focus only on the specified anatomical region mentioned in the question and identify any visible abnormalities.",
    "Examine the designated region of the CT scan carefully and describe observed abnormalities based on visual features.",
    "Your task is to assess a specific anatomical area in the chest CT and summarize any abnormal findings.",
    "Only consider the anatomical region indicated in the question, and detect abnormalities solely within that area.",
    "Review the specified region in the CT scan and document any visible pathology.",
    "Concentrate on the highlighted anatomical location and evaluate it for abnormal changes.",
    "Analyze only the given region and identify any distinct pathological findings."
]


REGION_think_options = [
    "You may reason through your observations inside a <think>...</think> block before summarizing the finding.",
    "Optionally, explain your visual analysis in a <think>...</think> tag prior to stating the abnormality.",
    "Structure the reasoning inside a <think>...</think> block before giving the final description.",
    "Document your thought process inside a <think>...</think> block, reasoning about the abnormality observed.",
    "Provide step-by-step reasoning inside a <think>...</think> section.",
    "Before concluding, you can summarize your observations inside a <think>...</think> block.",
    "Identify any visible abnormality in that area and explain your reasoning in a <think>...</think> tag."
]

# REGION_answer_requirement = [
#     "Provide your final abnormality description inside an <answer>...</answer> block.",
#     "State the observed abnormality clearly within an <answer>...</answer> tag.",
#     "Summarize your findings inside an <answer>...</answer> section.",
#     "Output the abnormality you observed by enclosing it within an <answer>...</answer> block.",
#     "Present your conclusion inside an <answer>...</answer> tag.",
#     "Identify any visible abnormality in that area and explain your reasoning in a <think>...</think> tag."
# ]

REGION_answer_requirement = [
    "Provide your final abnormality description in plain text.",
    "State the observed abnormality clearly.",
    "Summarize your findings as a direct description of the abnormal region.",
    "Output the abnormality you observed using natural language only.",
    "Present your conclusion about the abnormal region in clear descriptive form.",
    "Identify any visible abnormality in the specified area."
]


def generate_region_prompt(use_think=True):
    role = random.choice(REGION_roles)
    task = random.choice(REGION_task_descriptions)
    answer_instruction = random.choice(REGION_answer_requirement)
    
    if use_think:
        think_instruction = random.choice(REGION_think_options)
        prompt = f"{role} {task} {think_instruction} {answer_instruction}"
    else:
        prompt = f"{role} {task} {answer_instruction}"
    
    return prompt


OPEN_roles = [
    "You are an expert in diagnostic radiology.",
    "You are a chest CT imaging specialist tasked with visual diagnosis.",
    "You are a medical imaging expert focusing on disease identification in CT scans.",
    "As a radiologist, your role is to diagnose abnormalities based solely on imaging findings.",
    "You specialize in analyzing chest CT scans to derive diagnostic conclusions.",
    "You are a clinical imaging consultant providing diagnoses based on CT findings."
]

OPEN_task_descriptions = [
    "Analyze the CT image carefully and identify the most likely diagnosis based solely on visible findings.",
    "Your task is to deduce the diagnosis using only the visual information available in the chest CT scan.",
    "Study the CT scan attentively and derive the most plausible diagnosis from the observed features.",
    "Interpret the CT image and determine the diagnosis that best matches the visible abnormalities.",
    "Review the CT scan carefully and conclude with the diagnosis supported by the image evidence.",
    "Evaluate the visual features of the CT scan to arrive at a diagnosis without external information.",
    "Assess the CT scan findings and identify the condition that the image most likely represents."
]

OPEN_think_options = [
    "You may explain your diagnostic reasoning inside a <think>...</think> block before stating your diagnosis.",
    "Optionally, structure your reasoning process within a <think>...</think> tag.",
    "If included, document your step-by-step diagnostic logic inside a <think>...</think> section.",
    "Summarize the visual evidence and reasoning inside a <think>...</think> block prior to the final diagnosis.",
    "Think through the abnormalities carefully and, record your thought process in a <think>...</think> block.",
    "Before outputting the diagnosis, you can reason through your observations inside a <think>...</think> tag.",
    "Explain the reasoning leading to your diagnosis inside a <think>...</think> tag."
]

# OPEN_answer_requirement = [
#     "Present the final diagnosis clearly inside an <answer>...</answer> block.",
#     "State your diagnosis by enclosing it within an <answer>...</answer> tag.",
#     "Output the most probable diagnosis inside an <answer>...</answer> block.",
#     "Provide your diagnostic conclusion in an <answer>...</answer> section.",
#     "Record the final diagnosis within an <answer>...</answer> tag based on the CT findings.",
#     "Provide the final diagnosis inside an <answer>...</answer> tag."
# ]

OPEN_answer_requirement = [
    "Present the final diagnosis clearly in plain text.",
    "State your diagnosis directly without using any tags.",
    "Output the most probable diagnosis as a concise sentence.",
    "Provide your diagnostic conclusion in natural language form.",
    "Record the final diagnosis based on the CT findings.",
    "Clearly state the diagnosis using only descriptive text."
]


def generate_open_prompt(use_think=True):
    role = random.choice(OPEN_roles)
    task = random.choice(OPEN_task_descriptions)
    answer_instruction = random.choice(OPEN_answer_requirement)
    
    if use_think:
        think_instruction = random.choice(OPEN_think_options)
        prompt = f"{role} {task} {think_instruction} {answer_instruction}"
    else:
        prompt = f"{role} {task} {answer_instruction}"
    
    return prompt




VQA_roles = [
    "You are a medical visual question answering (VQA) assistant.",
    "You are an AI assistant trained to answer clinical questions based on CT images.",
    "You are a visual reasoning expert for medical CT image question answering tasks.",
    "You are a medical AI assistant tasked with interpreting CT scans and answering related questions.",
    "You specialize in answering medically relevant visual questions using CT image understanding.",
    "As a medical imaging assistant, your job is to provide answers grounded in the visible evidence of CT scans.",
    "You are a CT imaging AI trained to provide clinical reasoning based on visible abnormalities.",
    "You are an automated assistant designed to answer diagnostic questions about CT scans.",
    "As a radiology VQA system, you are responsible for answering clinical inquiries from CT image analysis.",
    "You are a computer-aided diagnostic assistant focused on interpreting CT scans for question answering.",
    "You are a vision-based medical AI designed to provide factual answers from CT images.",
    "You act as a clinical support assistant, answering visual questions by analyzing CT scan data only."
]


VQA_task_descriptions = [
    "Answer naturally based strictly on what is visible in the image.",
    "A CT image and a clinical question will be presented. Provide a concise and accurate answer using only the image.",
    "Use the CT image to understand and respond directly to the user's question without adding external information.",
    "Your task is to answer the user's question based solely on the CT image.",
    "Given a CT image and a visual question, infer the correct answer based entirely on observed features.",
    "Formulate a direct answer to the user's question using only the visible clues present in the CT scan.",
    "Base your answer exclusively on visual findings from the provided CT image.",
    "Evaluate the CT scan carefully and provide an answer rooted only in what can be visually confirmed.",
    "Focus on observable evidence from the CT scan when responding to the clinical question.",
    "Answer the user question by extracting only the visible information from the CT image without speculating.",
    "Review the CT scan findings to answer the question directly, relying purely on the image content.",
    "Determine the best possible answer based on your analysis of the CT scan, ignoring external assumptions.",
    "Analyze the CT scan, focus on objective findings, and answer the question with clear clinical reasoning.",
    "Strictly use the observed data in the CT scan to formulate your response to the question."
]



VQA_think_options = [
    "You may reason step-by-step inside a <think>...</think> block before answering.",
    "Write your visual reasoning in a <think>...</think> section before giving the answer.",
    "Structure your thought process inside a <think>...</think> tag to show step-by-step logic.",
    "You can first describe your reasoning in a <think>...</think> tag before providing the answer.",
    "To clarify your decision, you may use a <think>...</think> block to explain your observations.",
    "Summarize your logic inside a <think>...</think> section for transparency."
]

# VQA_answer_requirement = [
#     "Provide the final answer in an <answer>...</answer> block.",
#     "State your response inside an <answer>...</answer> tag.",
#     "Output your answer by placing it inside <answer>...</answer>.",
#     "Write your conclusion in the <answer>...</answer> section.",
#     "Respond with the answer enclosed in <answer>...</answer>.",
#     "Record your final answer within an <answer>...</answer> block."
# ]

VQA_answer_requirement = [
    "Provide the final answer directly in plain text.",
    "State your response clearly without using any tags.",
    "Output your answer as a simple, natural language statement.",
    "Write your conclusion using clear descriptive text.",
    "Respond with the answer in plain language only.",
    "Record your final answer using clear descriptive text."
]


def generate_vqa_prompt(use_think=True):
    role = random.choice(VQA_roles)
    task = random.choice(VQA_task_descriptions)
    answer_instruction = random.choice(VQA_answer_requirement)
    
    if use_think:
        think_instruction = random.choice(VQA_think_options)
        prompt = f"{role} {task} {think_instruction} {answer_instruction}"
    else:
        prompt = f"{role} {task} {answer_instruction}"
    
    return prompt


def generate_vqa_prompt_user(use_think=True):
    task = random.choice(VQA_task_descriptions)
    answer_instruction = random.choice(VQA_answer_requirement)

    if use_think:
        think_instruction = random.choice(VQA_think_options)
        prompt = f"\nYour task: {task} {think_instruction} {answer_instruction}"
    else:
        prompt = f"\nYour task: {task} {answer_instruction}"
    
    return prompt

def generate_vqa_roles():
    role = random.choice(VQA_roles)
    return role




def generate_vqa_wo_tag_prompt():
    role = random.choice(VQA_roles)
    task = random.choice(VQA_task_descriptions)
    prompt = f"{role} {task}"

    return prompt


def generate_vqa_wo_tag_prompt_user():
    task = random.choice(VQA_task_descriptions)
    prompt = f"\nYour task: {task}"

    return prompt

def generate_vqa_wo_tag_role():
    role = random.choice(VQA_roles)

    return role



FINDING_roles = [
    "You are a medical AI assistant that analyzes CT scans and asked to answer questions about it.",
    "You are a clinical imaging assistant specialized in creating radiology findings from CT images.",
    "You are an AI model trained to generate detailed radiology findings based on CT scan observations.",
    "As a radiology AI assistant, your task is to describe the findings observed in CT scans.",
    "You are a diagnostic imaging assistant focused on reporting CT findings only.",
    "You specialize in interpreting CT scans and producing radiology findings for clinical reports."
]

FINDING_task_descriptions = [
    "Analyze the provided CT scan carefully and generate a detailed description of the radiological findings.",
    "Carefully review the CT image and document all visible findings.",
    "Based solely on the CT scan, produce a thorough list of observed radiological abnormalities and features.",
    "Write only the findings section, focusing strictly on what is visible in the CT scan.",
    "Examine the CT scan and generate a findings report describing all relevant anatomical and pathological observations.",
    "Observe the CT image meticulously and create a findings-only report.",
    "Identify and describe the visible radiologic features from the CT scan."
]


def generate_finding_prompt():
    role = random.choice(FINDING_roles)
    task = random.choice(FINDING_task_descriptions)
    
    prompt = f"{role} {task}"
    
    return prompt


def generate_finding_prompt_user():
    task = random.choice(FINDING_task_descriptions)
    prompt = f"\nYour task: {task}"
    return prompt


def generate_finding_prompt_role():
    role = random.choice(FINDING_roles)
    return role

Think_finding_prefix = [
    # 通用型
    "Based on the CT image, the following findings are observed:",
    "After reviewing the CT scan, the visible findings include:",
    "Upon inspection of the CT image, the radiological findings are as follows:",
    "Visual analysis of the scan reveals the following findings:",
    "The CT scan shows the following radiological observations:",
    "Following a detailed review of the image, the findings are:",
    "From the CT scan, we can observe:",
    "On examining the CT scan, the following features are present:",
    "According to the visual evidence in the scan, the findings are:",
    "Observation of the CT image suggests the following findings:",
    "The imaging results reveal the following features:",
    "Here are the visible findings based on the CT scan:",

    # 更专业化表达
    "The CT scan demonstrate the following findings:",
    "Image interpretation reveals the following radiologic details:",
    "Notable features observed in the CT scan include:",
    "Upon axial and coronal review, the following abnormalities are noted:",
    "Assessment of the scan reveals:",
    "Evaluation of the pulmonary fields shows:",
    "On review of the mediastinal structures, we note:",
    "Imaging findings include:",

    # 稍微更口语 / 科研风格
    "The scan appears to show the following:",
    "Reviewing the scan reveals these observations:",
    "The observed features in this case are as follows:",
    "As seen in the CT image, the following anomalies are present:",
    "Upon radiological review, the following findings stand out:",
    "The visible pathology can be summarized as follows:",
    "The image contains the following diagnostically relevant features:",
    "These are the findings directly inferred from the image:",
    "Key radiological findings observed include:",
    "This CT scan presents the following radiologic characteristics:"
]



PosREC_templates = {
"cls_questions": [
    "Can you find the {} in this image? Give coordinates.",
    "Can you find {} in this image? Please output the coordinates.",
    "Please bounding the {} by box in this image.",
    "Where is {} in this image? Please respond with a bounding box.",
    "Where is {} in this image? Please output the box.",
    "Can you locate the {} in this image? Please output its coordinates.",
    "Could you mark the {} by bounding box in this image?",
    "Where can I find the {} in this image? Please provide its bounding box.",
    "Identify the indicated {} in this image. Please provide the coordinates of its bounding box.",
    "Can you locate the {} in this image? Please provide coordinates.",
    "Where is the {} in this image? Please output its coordinates.",
    "Please outline the {} with a bounding box in this image.",
    "Identify the location of {} in this image. Provide the bounding box coordinates.",
    "Where can I find the {} in this image? Please give its bounding box.",
    "Locate the {} in this image. Output its coordinates, please.",
    "Could you mark the {} with a bounding box in this image?",
    "Spot the {} in this image and provide its bounding box coordinates.",
    "Please pinpoint the {} in this image. Output its coordinates.",
    "Identify the {} indicated in this image. Provide coordinates for its bounding box.",
],

"des_questions": [
    "Description: {} Please answer and find it by box based on the above description.",
    "Definition: {} Please answer and show the bounding box based on the above definition.",
    "Description: {} Can you answer and find it by coordinates based on the above description.",
    "Definition: {} Please output the bounding box and answer based on the above definition.",
    "Description: {} Respond and locate it using a bounding box according to the description.",
    "Definition: {} Please provide an answer and display the bounding box according to the given definition.",
    "Description: {} Can you identify and locate it by coordinates, following the provided description or definition?",
    "Definition: {} Please output the bounding box and provide an answer based on the provided definition.",
    "Based on the description or definition, please respond to {} and indicate its location with a bounding box.",
    "{} Please answer and find it by box based on the above description.",
    "{} Please answer and show the bounding box based on the above definition.",
    "{} Can you answer and find it by coordinates based on the above description.",
    "{} Please output the bounding box and answer based on the above definition.",
    "{} Respond and locate it using a bounding box according to the description.",
    "{} Please provide an answer and display the bounding box according to the given definition.",
    "{} Can you identify and locate it by coordinates, following the provided description or definition?",
    "{} Please output the bounding box and provide an answer based on the provided definition.",
    "Please answer and find it by box based on the description. {}",
    "Please answer and show the bounding box based on the definition. {}",
    "Can you answer and find it by coordinates based on the description. {}",
    "Please output the bounding box and answer based on the definition. {}",
    "Respond and locate it using a bounding box according to the description. {}",
    "Please provide an answer and display the bounding box according to the given definition. {}",
    "Can you identify and locate it by coordinates, following the provided description or definition? {}",
    "Please output the bounding box and provide an answer based on the provided definition. {}",
    "Description: {} Find and mark it with a bounding box.",
    "Definition: {} Show the bounding box and provide the answer.",
    "Describe: {} Identify and locate it by coordinates.",
    "Define: {} Output the bounding box and respond.",
    "Description: {} Locate it with a bounding box.",
    "Definition: {} Provide the answer with the bounding box.",
    "Description: {} Identify and locate it by coordinates.",
    "Definition: {} Output the bounding box with the answer.",
    "{} Answer and locate with a bounding box.",
    "{} Show the bounding box based on the definition.",
    "{} Locate it by coordinates according to the description.",
    "{} Output the bounding box and answer.",
    "{} Respond and locate with a bounding box.",
    "{} Provide the answer with the bounding box.",
    "{} Identify and locate by coordinates.",
    "{} Output the bounding box with the answer.",
],

"cls_answers": [
    "Coordinates are {}.",
    "Sure, {}.",
    "Sure, it is {}.",
    "Sure, the bounding box is {}.",
    "{}.",
    "Here are the coordinates: {}.",
    "Of course, it's located at {}.",
    "The bounding box is given by {}.",
    "The box is {}.",
    "Coordinates: {}.",
    "{}.",
    "Yes, {}.",
    "{} are the coordinates.",
    "Sure, it's at {}.",
    "The bounding box: {}.",
    "{} - here are the coordinates.",
    "{} - it's located at these coordinates.",
    "The box: {}.",
    "Coordinates: {}.",
    "Certainly, it's at {}.",
    "Yes, it is located at {}.",
    "Sure, the bounding box: {}.",
    "It's located at {}.",
    "Here are the coordinates: {}.",
    "Of course, it's positioned at {}.",
    "The bounding box can be described by: {}.",
    "The box is specified by: {}.",
    "Yes, the coordinates are {}.",
    "Sure, it's located around {}.",
    "Yes, it's within the bounding box: {}.",
    "Absolutely, the coordinates: {}.",
    "Yes, it's within the box: {}.",
],

"des_answers": [
    "The target is {} and the coordinates is {}.",
    "The category is {} and the bounding box is {}.",
    "It is {}, {}.",
    "{}, {}",
    "The target is identified as {} and its coordinates are {}.",
    "The category is {}, the bounding box is provided as {}.",
    "It is characterized by {}, with coordinates {}.",
    "The identified attributes are {}, {}.",
    "Describing it as {}, the corresponding box is {}.",
    "The target is {} and its coordinates are {}.",
    "Categorized as {}, with a bounding box of {}.",
    "It's {}, located at {}.",
    "It's characterized as {}, with coordinates {}.",
    "Identified as {}, with a bounding box: {}.",
    "Belongs to the category of {}, with coordinates {}.",
    "Characterized by {}, with a bounding box given as {}.",
    "Describing it as {}, with coordinates {}.",
    "Identified attributes: {}, with a bounding box of {}.",
    "The category is {}, and the coordinates are {}.",
    "Describing it as {}, its bounding box is {}.",
    "It falls under the category of {}, with coordinates {}.",
],

"cls_no_answers": [
    "Sorry, there is no {}.",
    "No, we can not see {}.",
    "{} is not here.",
    "Sorry, there's no {} in sight.",
    "Unfortunately, {} is not visible here.",
    "No, {} is not present in this image.",
    "Apologies, but there's no sign of {}.",
    "We couldn't find {} in this image.",
    "It seems {} is not included in this image.",
    "Sorry, no {} found.",
    "No {} visible.",
    "Not seeing {} here.",
    "{} absent.",
    "No {} detected.",
    "Can't find {} in this image.",
],

"des_no_answers": [
    "This is {}, but not here.",
    "This is {}, however we can not see it.",
    "This is {}, but it's not present here.",
    "This is {}, but unfortunately, it's not visible in this context.",
    "While this is {}, it doesn't seem to be here.",
    "This is {}, but it's not depicted in this image.",
    "While this is {}, it's not currently observable in this scene.",
    "{} mentioned, but not here.",
    "{} described, but not visible.",
    "This is {}, but it's not here.",
    "This is {}, but it's not visible.",
    "{} mentioned, but not depicted here.",
    "{} described, but not present.",
]
}


PosREG_templates = {
"cls_questions": [
            "What target is present within the coordinates {} ?",
            "Does the bounding box {} contain any target?",
            "Within the specified region {}, what target is present?",
            "Do you know what it is in the bounding box {}?",
            "What is it in this region {}?",
            "What object is located within the coordinates {}?",
            "Within the specified area {}, what object can be found?",
            "Can you identify the object within the bounding box {}?",
            "What object is present in this region {}?",
            "What's within coordinates {}?",
            "Does the box {} contain any target?",
            "What's in the region {}?",
            "What's within the box {}?",
            "What object is in this region {}?",
            "What's located within coordinates {}?",
            "What's within the area {}?",
            "Can you identify the object within box {}?",
            "What object is present in this area {}?",
            "What target can be found within coordinates {}?",
            "Is there any target within the bounding box {}?",
            "Within the specified region {}, what target is present?",
            "Can you identify what's within the bounding box {}?",
            "What object is present in this region {}?",
            "What's located within coordinates {}?",
            "Within the specified area {}, what object can be found?",
            "Could you identify the object within the bounding box {}?",
            "What object is present in this region {}?",
            "What's located at coordinates {}?",
            "Is there a target within the box {}?",
            "What's situated in the region {}?",
            "What's inside the box {}?",
            "What object can be found in this region {}?",
            "What's situated within coordinates {}?",
            "What's within the specified area {}?",
            "Can you identify any object within box {}?",
            "What object is present in this area {}?",
        ],

"des_questions": [
            "Please describe the target and its function based on the box {} in the image.",
            "Do you know what is it in this bounding box {}? Answer and explain it.",
            "What's the target in the bounding box {}? What function does it have?",
            "What is the area marked with a box {} in the image? Can you explain it?",
            "Could you describe the object and its purpose within the bounding box {} in the image?",
            "Can you identify and describe the object within this bounding box {}? Please explain.",
            "What is the object located in the bounding box {}? Could you explain its function?",
            "Could you describe the area outlined by the box {} in the image? Please explain its significance.",
            "Describe the target and its function in box {}.",
            "What's in this box {}? Describe and explain.",
            "What's in the box {}? Explain its function.",
            "What's inside the box {}? Can you describe it?",
            "Describe the object in box {} and its purpose.",
            "Identify and describe the object in box {}. Please explain.",
            "What's the object in box {}? Describe its function.",
            "Describe the area outlined by box {} and its significance.",
            "What's in the marked area {}? Please explain.",
        ],

"cls_answers": [
            "The target is {}.",
            "Sure, the bounding box contains {}.",
            "Sure, it is {}.",
            "Sure, {} is in the bounding box.",
            "{}.",
            "The object is {}.",
            "Of course, it's {}.",
            "Certainly, {} can be found in the bounding box.",
            "Yes, the bounding box includes {}.",
            "The target is identified as {}.",
            "Affirmative, the bounding box contains {}.",
            "Yes, it is indeed {}.",
            "Yes, {} is within the bounding box.",
            "The object in question is {}.",
            "Of course, it's indeed {}.",
            "Certainly, {} can be found within the bounding box.",
            "Yes, the bounding box includes {}.",
            "Indeed, {} is present within the bounding box.",
            "Confirmed, the object is {}.",
            "Absolutely, it's {}.",
        ],

"des_answers": [
            "Sure, it is {}. {}",
            "The category is {}. {}",
            "It is {}, {}",
            "{}, {}",
            "The target is identified as {} and its description is {}",
            "The category is {}. Description: {}",
            "It is characterized by {}, {}",
            "The identified attributes are {}, {}",
            "Sure, it is {}. Describing it as {}",
            "Certainly, it is {}. {}",
            "The category is identified as {}. {}",
            "It's labeled as {}, {}",
            "Identified as {}, {}",
            "The target is recognized as {} with the following description: {}",
            "The category is {}. Here's the description: {}",
            "Characterized by {}, {}",
            "Identified attributes: {}, {}",
            "Sure, it's {}. Described as {}",
        ],

"cls_no_answers": [
    "Sorry, there is no {}.",
    "No, we can not see {}.",
    "{} is not here.",
    "Sorry, we couldn't find any {}.",
    "Nope, {} isn't visible here.",
    "{} seems to be absent.",
    "Unfortunately, {} is not present.",
    "It appears there's no {} in sight.",
    "Regrettably, we couldn't detect any {}.",
],

"des_no_answers": [
    "This is {}, but not here.",
    "This is {}, however we can not see it.",
    "This is {}, but it's not present here.",
    "This is {}, however, it's not visible in this context.",
    "The answer is {}, but it is not here.",
    "The object is {}, but it is absent in the image.",
]
}


Seg_templates = {
"cls_questions": [
            "Can you segment the {} in this image?",
            "Can you segment {} in this image? Please output the mask.",
            "Please segment the {} in this image.",
            "What is {} in this image? Please respond with segmentation mask.",
            "What is {} in this image? Please output segmentation mask.",
            "Could you provide a segmentation for the {}?",
            "I need the {} segmented from this image.",
            "Segment {} from this image and provide the mask, please.",
            "Please provide a segmentation mask for the {} in this image.",
            "Can you identify and segment the {} in this image?",
            "Could you segment the {} here and share the mask?",
            "I'm looking for segmentation of {} in this image.",
            "Can you segment the object labeled as {}?",
            "Segment the {} in this image, please.",
            "Please segment {} in this image and provide the mask.",
            "What's the segmentation for {} in this image?",
            "Segment the {} in this image and share the mask.",
            "Please provide segmentation for {} in this image.",
            "Could you segment the {} here and share the mask?",
            "Segment the object labeled as {} and provide the mask.",
            "I need the {} segmented from this image. Can you help?",
            "Can you identify and segment the {} in this image?",
            "Can you segment {} here and output the mask?",
            "Please segment the {} in this image and share the mask.",
            "Segment {} from this image and provide the mask, please.",
            "Can you segment the {} in this image?",
        ],

"des_questions": [
            "{} Please answer and segment.",
            "{} Please output segmentation mask and answer.",
            "{} Please answer and segment based on the above description.",
            "{} Please answer and segment based on the above definition.",
            "{} Can you answer and segment it based on the above description or definition.",
            "{} Please output segmentation mask and answer based on the above description or definition.",
            "{} Please segment accordingly.",
            "{} Please provide segmentation and answer according to it.",
            "{} Now, segment it and provide your answer.",
            "{} Please segment and provide your response.",
            "{} Can you segment it accordingly?",
            "Description: {} Please answer and segment based on the above description.",
            "Definition: {} Please answer and segment based on the above definition.",
            "Description: {} Can you answer and segment it based on the above description or definition.",
            "Definition: {} Please output segmentation mask and answer based on the above description or definition.",
            "Provided description: {} Please segment accordingly.",
            "Given definition: {} Please provide segmentation and answer according to it.",
            "The description provided is: {} Now, segment it and provide your answer.",
            "Based on the provided definition: {} Please segment and provide your response.",
            "Describing the object as: {} Can you segment it accordingly?",
            "Defining it as: {} Now, segment and provide your answer.",
        ],

"cls_answers": [
            "It is [SEG].",
            "Sure, [SEG].",
            "Sure, it is [SEG].",
            "Sure, the segmentation result is [SEG].",
            "[SEG].",
            "The segmentation indicates [SEG].",
            "According to the segmentation, it is [SEG].",
            "The segmentation reveals [SEG].",
            "The segmentation suggests [SEG].",
            "From the segmentation, it appears to be [SEG].",
            "The target is [SEG].",
            "The segmentation mask is [SEG].",
            "The mask is [SEG].",
],

"des_answers": [
            "The target is {} and the segmentation mask is [SEG].",
            "The category is {} and the mask is [SEG].",
            "It is {}, [SEG].",
            "{}, [SEG]",
            "Identified as {}, here is the segmentation: [SEG].",
            "Categorized as {}, the segmentation is: [SEG].",
            "The class is {}, and the corresponding segmentation is: [SEG].",
            "Regarding the classification, it is {}, and the segmentation is: [SEG].",
            "Classified as {}, here's the segmentation: [SEG].",
            "The label assigned is {}, and the associated segmentation is: [SEG].",
            "Category: {}, segmentation: [SEG].",
            "It's classified as {}, with the segmentation: [SEG].",
        ],

"cls_no_answers": [
            "Sorry, there is no {}.",
            "No, we cannot see {}.",
            "{} is not present.",
            "There's no sign of {} in this image.",
            "Unfortunately, {} is not visible in this image.",
            "We cannot detect {} in this image.",
            "There's no indication of {} here.",
            "Regrettably, {} cannot be observed in this image.",
            "Sorry, {} isn't here.",
            "{} is absent.",
            "Nope, {} is missing.",
            "We're not seeing any {}.",
        ],

"des_no_answers": [
    "This is {}, but not here.",
    "This is {}, however we can not see it.",
    "While this is {}, it's not within view.",
    "This appears to be {}, but it's not in this vicinity.",
    "Although this is {}, it's not visible here.",
    "Though this is {}, it's not captured in this scene.",
    "This is indeed {}, but it's not here in the image.",
    "While we've identified {}, it's not within this image.",
    "This seems to be {}, but it's not depicted here.",
    "While this describes {}, it's not shown in this image.",
]
}
