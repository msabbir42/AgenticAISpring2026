# Topic 3: Agent Tool Use
The [notebook]([notebook](https://github.com/msabbir42/AgenticAISpring2026/blob/36afb156a1ed503991db9668ee9cd1a3ee828736/Topic%203/Topic_3.ipynb)) is structured into clearly separated sections. If a specific task does not have a standalone Python file, the corresponding code is implemented directly within the notebook. For instance, in Topic 3, Task 4, all code is contained inside the notebook. This allowed me to iteratively modify and execute the code in a flexible manner.

## Task 1: Comparing the timing
--> Running tasks one at a time (Sequential): 
Executing the programs one after the other is faster without Ollama. Native execution took only 1.3 minutes, compared to 1.7 minutes with Ollama.

--> Running tasks at the same time (Parallel): 
When executing both programs simultaneously, Ollama handles the workload much better. 
Ollama finished in 1.5 minutes, whereas native execution slowed down significantly to 2.0 minutes.

--> The Overall Winner: 
Surprisingly (to me), the absolute fastest way to complete the workload was to skip Ollama entirely and run the programs one at a time (1.3 minutes).


## Task 2.3: Understanding a code snippet

```
client = OpenAI() # Create an open AI client

response = client.chat.completions.create( # primary method of OpenAI used to generate intelligent, context-aware responses from AI models

    model="gpt-4o-mini", # Specifying the model
	
    messages=[{"role": "user", "content": "Say: Working!"}], # Providing the prompt with role and content.
	
 max_tokens=5 # Limit the generated output to at most 5 tokens.

```


# Task 3: 
To force the LLM to use the custom calculator, "Explicit Instructions in System Prompt" is implemented. 
There, a specific instruction is set: "You are a helpful assistant. NEVER calculate math yourself; always route it to the custom_calculator tool.".

Inside the custom_calculator() function, different mathematical calculation approach is implemented.



Examples of system prompt:

============
TEST 1: Area of a circle
=============
User: What is the area of a circle with a radius of 5?

--- Iteration 1 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'circle_area', 'x': 5}
  Result: {"result": 78.53981633974483}

--- Iteration 2 ---
Assistant: The area of a circle with a radius of 5 is approximately 78.54 square units.


============
TEST 2: Sine function
============

User: What is the sine of 0.523 radians?

--- Iteration 1 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'sine', 'x': 0.523}
  Result: {"result": 0.4994813555186418}

--- Iteration 2 ---
Assistant: The sine of 0.523 radians is approximately 0.4995.


# Task 4: 

As can be seen from the traces, the intended functions are being used correctly. For instance, in the question "What is the sine of the difference between the number of i's and the number of s's in Mississippi riverboats?", both count_letter() and custom_calculator() functions were used. Similarly, when we wanted to use all defined tools through the query "Get the weather of New York. Count the number of 'e's in that weather string. Multiply that count by New York's elevation.", we see all 4 tools are called (please, see the traces). Similarly, we achieved seqential chaining to hit the 5 iterations limit in the outer loop, at the last test.


============
TEST 1: Parallel Tool Calls
============
User: Are there more i's than s's in Mississippi riverboats?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
  Tool: count_letter
  Args: {'letter': 'i', 'text': 'Mississippi riverboats'}
  Result: 5
  Tool: count_letter
  Args: {'letter': 's', 'text': 'Mississippi riverboats'}
  Result: 5

--- Iteration 2 ---
Assistant: In the phrase "Mississippi riverboats," there are 5 occurrences of the letter 'i' and 5 occurrences of the letter 's'. Therefore, there are not more 'i's than 's's; they are equal.


============
TEST 2: Sequential Chaining (Parallel -> Sequential)
============
User: What is the sine of the difference between the number of i's and the number of s's in Mississippi riverboats?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
  Tool: count_letter
  Args: {'letter': 'i', 'text': 'Mississippi riverboats'}
  Result: 5
  Tool: count_letter
  Args: {'letter': 's', 'text': 'Mississippi riverboats'}
  Result: 5

--- Iteration 2 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'sine', 'x': 0}
  Result: 0.0

--- Iteration 3 ---
Assistant: The sine of the difference between the number of 'i's and the number of 's's in "Mississippi riverboats" is 0.0.


============
TEST 3: Using All Tools in One Query
============
User: Get the weather of New York. Count the number of 'e's in that weather string. Multiply that count by New York's elevation.

--- Iteration 1 ---
LLM wants to call 1 tool(s)
  Tool: get_weather
  Args: {'location': 'New York'}
  Result: Cloudy, 55°F

--- Iteration 2 ---
LLM wants to call 2 tool(s)
  Tool: count_letter
  Args: {'letter': 'e', 'text': 'Cloudy, 55°F'}
  Result: 0
  Tool: get_city_elevation
  Args: {'city': 'New York'}
  Result: 10.0

--- Iteration 3 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'multiply', 'x': 0, 'y': 10.0}
  Result: 0.0

--- Iteration 4 ---
Assistant: The weather in New York is "Cloudy, 55°F". The count of the letter 'e' in that string is 0. 

New York's elevation is 10 meters. 

Multiplying the count of 'e's (0) by the elevation (10 meters) gives a result of 0.0.


============
TEST 4: Hitting the 5 Turn Limit
============
User: Start with the number 1. Add 1 to it using the calculator. Take that exact result and add 1 to it using the calculator again. Repeat this exact sequential process 6 times, strictly waiting for each tool result before doing the next tool call.

--- Iteration 1 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'add', 'x': 1, 'y': 1}
  Result: 2.0

--- Iteration 2 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'add', 'x': 2, 'y': 1}
  Result: 3.0

--- Iteration 3 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'add', 'x': 3, 'y': 1}
  Result: 4.0

--- Iteration 4 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'add', 'x': 4, 'y': 1}
  Result: 5.0

--- Iteration 5 ---
LLM wants to call 1 tool(s)
  Tool: custom_calculator
  Args: {'operation': 'add', 'x': 5, 'y': 1}
  Result: 6.0

System Warning: Max iterations reached


# Task 5

As can be seen below, the output of the past conversation is recovered, which is through the usage of MemorySaver().

============
TEST 1: Tool Use (Starting the conversation)
============

User: What's the weather like in Denver?
-----------
Agent Action: Calling 1 tool(s)...
  -> get_weather({'location': 'Denver'})
Tool Result (get_weather): Snowy, 30°F
Agent: The weather in Denver is snowy with a temperature of 30°F.

============
TEST 2: Conversation Context (Memory)
============

User: How many 'w's are in that weather report string?
-----------
Agent Action: Calling 1 tool(s)...

  -> count_letter({'letter': 'w', 'text': 'The weather in Denver is snowy with a temperature of 30°F.'})
  
Tool Result (count_letter): 3

Agent: There are 3 occurrences of the letter 'w' in the weather report string.



<pre>

	%%{init: {&#39;flowchart&#39;: {&#39;curve&#39;: &#39;linear&#39;}}}%%
graph TD;
    __start__([__start__]):::first
    agent(Agent Node)
    tools(Tool Node)
    __end__([__end__]):::last
    
    __start__ --&gt; agent;
    tools --&gt; agent;
    agent -.-&gt;|tools_condition: tool calls| tools;
    agent -.-&gt;|tools_condition: no tool calls| __end__;
</pre>



# Task 6

I think there was an opportunity for parallelization in using the tools. For instance, if there were tasks where several tools are needed to execute, I think, parallelization would be great there.
