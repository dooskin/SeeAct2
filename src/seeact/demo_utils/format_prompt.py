import re
from typing import Any, List
import shlex

def format_choices(elements, include_dom: bool = False):
    converted_elements = []
    elements_w_descriptions = []
    for element in elements:
        if "description" in element and "=" in element["description"] and "'" not in element["description"] and "\"" not in element["description"]:
            description_dict = [] 
            for sub in shlex.split(element["description"]): 
                if '=' in sub:
                    description_dict.append(map(str.strip, sub.split('=', 1)))
            element.update(dict(description_dict))
        elements_w_descriptions.append(element)

    converted_elements = []
    for i, element in enumerate(elements_w_descriptions):
        converted = ""
        if element['tag']!="select":
            converted += f'{element["center_point"]} <{element["tag_with_role"]}>'
            converted += (
                element["description"]
                if len(element["description"].split()) < 30
                else " ".join(element["description"].split()[:30]) + "..."
            )
            converted += f"</{element['tag']}>"
        else:
            converted += f'{element["center_point"]} <{element["tag_with_role"]}>'
            converted += (
                element["description"]
            )
            converted += f"</{element['tag']}>"
        if include_dom and element.get("outer_html"):
            # Append a compact DOM snippet
            converted += f" | DOM: {element['outer_html']}"
        converted_elements.append(converted)

    return converted_elements

def postprocess_action_lmm(text):
    text = text.strip()
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:",
        "")
    text = text.replace("The uppercase letter of your choice based on your analysis is:\n\n", "")
    text = text.replace("The uppercase letter of your choice based on your analysis is:\n", "")
    text = text.replace("The uppercase letter of your choice based on your analysis is:", "")
    text = text.replace("The uppercase letter of my choice is \n\n", "")
    text = text.replace("The uppercase letter of my choice is \n", "")
    text = text.replace("The uppercase letter of my choice is ", "")
    text = text.replace("The uppercase letter of your choice is \n\n", "")
    text = text.replace("The uppercase letter of your choice is \n", "")
    text = text.replace("The uppercase letter of your choice is ", "")
    text = text.replace("The uppercase letter of your choice.\n\n", "")
    text = text.replace("The uppercase letter of your choice.\n", "")
    text = text.replace("The uppercase letter of your choice.", "")
    text = text.replace("The uppercase letter of your choice based on my analysis is:\n\n", "")
    text = text.replace("The uppercase letter of your choice based on my analysis is:\n", "")
    text = text.replace("The uppercase letter of your choice based on my analysis is:", "")
    text = text.replace("The correct choice based on the analysis would be:\n\n", "")
    text = text.replace("The correct choice based on the analysis would be:\n", "")
    text = text.replace("The correct choice based on the analysis would be :", "")
    text = text.replace("The correct choice based on the analysis would be ", "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:",
        "")
    text = text.replace("The uppercase letter of your choice.\n\n", "")
    text = text.replace("The uppercase letter of your choice.\n", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is:\n\n", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is:\n", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is:", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is ", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is:\n\n", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is:\n", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is:", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is ", "")
    text = text.replace("The correct element to select would be:\n\n", "")
    text = text.replace("The correct element to select would be:\n", "")
    text = text.replace("The correct element to select would be:", "")
    text = text.replace("The correct element to select would be ", "")
    text = text.replace("The uppercase letter of my choice is:\n\n", "")
    text = text.replace("The uppercase letter of my choice is:\n", "")
    text = text.replace("The uppercase letter of my choice is:", "")
    text = text.replace("The uppercase letter of my choice is ", "")
    text = text.replace("Choose an action from {CLICK, TYPE, SELECT}.\n\n", "")
    text = text.replace("Choose an action from {CLICK, TYPE, SELECT}.\n", "")
    text = text.replace("Choose an action from {CLICK, TYPE, SELECT}.", "")
    text = text.replace("Provide additional input based on ACTION.\n\n", "")
    text = text.replace("Provide additional input based on ACTION.\n", "")
    text = text.replace("Provide additional input based on ACTION.", "")
    selected_option = re.findall(r"ELEMENT: ([A-Z]{2}|[A-Z])", text)

    if selected_option:
        selected_option = (
            selected_option[0]
        )
    else:
        selected_option = "Invalid"

    action = re.search(
        r"ACTION: (CLICK|SELECT|TYPE|HOVER|PRESS ENTER|SCROLL UP|SCROLL DOWN|PRESS HOME|PRESS END|PRESS PAGEUP|PRESS PAGEDOWN|NEW TAB|CLOSE TAB|GO BACK|GO FORWARD|TERMINATE|NONE|GOTO|SAY|MEMORIZE)",
        text
    )


    if action:
        action = action.group(1)
        start = text.find(f"ACTION: {action}")
        for probing_length in range(15, 160, 10):
            selected_option_from_action = re.findall(
                r"ELEMENT: ([A-Z]{2}|[A-Z])",
                text[max(start - probing_length, 0):start])
            # print("text span:",text[max(start-probing_length,0):start])
            # print("finded group:",selected_option__)
            if selected_option_from_action:
                selected_option = selected_option_from_action[0]
                break
    else:
        action = "None"

    value = re.search(r"VALUE: (.*)$", text, re.MULTILINE)
    value = value.group(1) if value is not None else ""
    
    reason = "None"
    if 'REASON:' in text:
        reason = text.split('REASON:')[1].split('ACTION:')[0].strip()
        
    
    return selected_option, action.strip(), process_string(process_string(value.strip())), reason


def format_ranking_input(elements: List[Any], task: str, previous_actions: List[str]) -> List[List[str]]:
    """Build [query, doc] pairs for ranking candidates.

    - elements: list entries like [center_point, description, tag_head, box_model, selector, real_tag_name]
    - task: confirmed_task string
    - previous_actions: list of action strings (may be empty)
    """
    # Build a compact query with last few actions
    prev = "; ".join(previous_actions[-3:]) if previous_actions else "None"
    query = f"task: {task}\nprevious: {prev}"
    model_input: List[List[str]] = []
    for el in elements:
        try:
            desc = str(el[1])
            tag = str(el[-1]) if len(el) >= 6 else "element"
            doc = f"<{tag}> {desc} </{tag}>"
            # truncate overly long docs to keep ranking efficient
            if len(doc) > 400:
                doc = doc[:397] + "..."
            model_input.append([query, doc])
        except Exception:
            continue
    return model_input




def postprocess_action_lmm_pixel(text):
    text = text.strip()
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:",
        "")
    text = text.replace("The uppercase letter of your choice based on your analysis is:\n\n", "")
    text = text.replace("The uppercase letter of your choice based on your analysis is:\n", "")
    text = text.replace("The uppercase letter of your choice based on your analysis is:", "")
    text = text.replace("The uppercase letter of my choice is \n\n", "")
    text = text.replace("The uppercase letter of my choice is \n", "")
    text = text.replace("The uppercase letter of my choice is ", "")
    text = text.replace("The uppercase letter of your choice is \n\n", "")
    text = text.replace("The uppercase letter of your choice is \n", "")
    text = text.replace("The uppercase letter of your choice is ", "")
    text = text.replace("The uppercase letter of your choice.\n\n", "")
    text = text.replace("The uppercase letter of your choice.\n", "")
    text = text.replace("The uppercase letter of your choice.", "")
    text = text.replace("The uppercase letter of your choice based on my analysis is:\n\n", "")
    text = text.replace("The uppercase letter of your choice based on my analysis is:\n", "")
    text = text.replace("The uppercase letter of your choice based on my analysis is:", "")
    text = text.replace("The correct choice based on the analysis would be:\n\n", "")
    text = text.replace("The correct choice based on the analysis would be:\n", "")
    text = text.replace("The correct choice based on the analysis would be :", "")
    text = text.replace("The correct choice based on the analysis would be ", "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n",
        "")
    text = text.replace(
        "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:",
        "")
    text = text.replace("The uppercase letter of your choice.\n\n", "")
    text = text.replace("The uppercase letter of your choice.\n", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is:\n\n", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is:\n", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is:", "")
    text = text.replace("The uppercase letter of your choice based on the analysis is ", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is:\n\n", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is:\n", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is:", "")
    text = text.replace("The uppercase letter of my choice based on the analysis is ", "")
    text = text.replace("The correct element to select would be:\n\n", "")
    text = text.replace("The correct element to select would be:\n", "")
    text = text.replace("The correct element to select would be:", "")
    text = text.replace("The correct element to select would be ", "")
    text = text.replace("The uppercase letter of my choice is:\n\n", "")
    text = text.replace("The uppercase letter of my choice is:\n", "")
    text = text.replace("The uppercase letter of my choice is:", "")
    text = text.replace("The uppercase letter of my choice is ", "")
    text = text.replace("Choose an action from {CLICK, TYPE, SELECT}.\n\n", "")
    text = text.replace("Choose an action from {CLICK, TYPE, SELECT}.\n", "")
    text = text.replace("Choose an action from {CLICK, TYPE, SELECT}.", "")
    text = text.replace("Provide additional input based on ACTION.\n\n", "")
    text = text.replace("Provide additional input based on ACTION.\n", "")
    text = text.replace("Provide additional input based on ACTION.", "")

    action = re.search(
        r"ACTION: (CLICK|SELECT|TYPE|HOVER|PRESS ENTER|SCROLL UP|SCROLL DOWN|PRESS HOME|PRESS END|PRESS PAGEUP|PRESS PAGEDOWN|NEW TAB|CLOSE TAB|GO BACK|GO FORWARD|TERMINATE|NONE|GOTO|SAY|MEMORIZE)",
        text
    )


    if action:
        action = action.group(1)
        start = text.find(f"ACTION: {action}")
        for probing_length in range(15, 160, 10):
            selected_option_from_action = re.findall(
                r"ELEMENT: ([A-Z]{2}|[A-Z])",
                text[max(start - probing_length, 0):start])
            # print("text span:",text[max(start-probing_length,0):start])
            # print("finded group:",selected_option__)
            if selected_option_from_action:
                selected_option = selected_option_from_action[0]
                break
    else:
        action = "None"

    selected_option = re.search(r"ELEMENT: (.*)$", text, re.MULTILINE)
    selected_option = selected_option.group(1) if selected_option is not None else ""

    value = re.search(r"VALUE: (.*)$", text, re.MULTILINE)
    value = value.group(1) if value is not None else ""
    return selected_option, action.strip(), process_string(process_string(value.strip()))

def process_string(input_string):
    if input_string.startswith('"') and input_string.endswith('"'):
        input_string = input_string[1:-1]
    if input_string.endswith('.'):
        input_string = input_string[:-1]
    return input_string






